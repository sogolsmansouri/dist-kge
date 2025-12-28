import os
import time
import logging
import warnings

import psutil
from signal import signal, SIGINT
from typing import Dict, Optional
import threading
from kge import Config, Dataset
from kge.job import Job
from kge.distributed.parameter_server import init_torch_server, init_lapse_scheduler
from kge.distributed.parameter_client import KgeParameterClient
from kge.distributed.worker_process import WorkerProcessPool
from kge.distributed.work_scheduler import WorkScheduler
from kge.distributed.work_scheduler import LocalSchedulerClient
from kge.distributed.misc import get_num_keys, get_optimizer_dim, get_min_rank

import torch
from torch import multiprocessing as mp


def monitor_hardware(folder, interval=1):
    def bytes_to_mb(bytes_amount):
        return round(bytes_amount / 1024 / 1024, 2)

    logger = logging.getLogger("hardware_monitor")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(folder, "hardware_monitor.log"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # let's monitor the default connection between OUR two servers
    # todo: just monitor all interfaces later on
    interface = "enp130s0f0"
    while True:
        time.sleep(interval)
        cpu_percentage = psutil.cpu_percent()
        memory_percentage = psutil.virtual_memory().percent
        network_info = psutil.net_io_counters()
        bytes_sent = network_info.bytes_sent
        bytes_recv = network_info.bytes_recv
        # timestamp;cpu%;mem%;net_sent;net_recvm

        msg = f"{time.time()};{cpu_percentage};{memory_percentage};{bytes_to_mb(bytes_sent)};{bytes_to_mb(bytes_recv)}"
        network_info = psutil.net_io_counters(pernic=True)
        if interface in network_info.keys():
            bytes_sent = network_info[interface].bytes_sent
            bytes_recv = network_info[interface].bytes_recv
            msg += f";{bytes_to_mb(bytes_sent)};{bytes_to_mb(bytes_recv)}"
        logger.info(
            msg=msg
        )


def monitor_gpus(folder, interval=1):
    try:
        from py3nvml.py3nvml import (
            nvmlInit,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetComputeRunningProcesses,
            nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetMemoryInfo,
            nvmlShutdown,
        )
    except Exception:
        print("py3nvml not available. GPU monitoring disabled.")
        return
    try:
        nvmlInit()
    except Exception:
        print("could not initialize GPU monitor")
        return
    device_count = nvmlDeviceGetCount()
    if device_count == 0:
        return
    logger = logging.getLogger("gpu_monitor")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(folder, "gpu_monitor.log"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    while True:
        time.sleep(interval)
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            proc_res = nvmlDeviceGetComputeRunningProcesses(handle)
            mem_per_process = list(
                map(lambda obj: (obj.pid, obj.usedGpuMemory), proc_res)
            )
            res = nvmlDeviceGetUtilizationRates(handle)
            mem_res = nvmlDeviceGetMemoryInfo(handle)
            # timestamp;device_id;gpu_util;gpu_mem_util;gpu_temp;mem_per_process
            logger.info(
                f"{time.time()};{i};{res.gpu};{round((mem_res.used/mem_res.total)*100)};{mem_per_process}"
            )
    nvmlShutdown()

def update_config_for_distributed(config: Config):
    # setting num eval workers to 1 if < 1
    if config.get("job.distributed.num_eval_workers") < 1:
        warnings.warn("Need to have at least one worker for evaluation. "
                      "Setting job.distributed.num_eval_workers to 1")
        config.set("job.distributed.num_eval_workers", 1)
    # setting num workers to 1 if < 1
    if config.get("job.distributed.num_workers") < 1:
        warnings.warn("Need to have at least one worker for training. "
                      "Setting job.distribtued.num_workers to 1")
        config.set("job.distributed.num_workers", 1)
    # setting num workers per machine to num workers if < 0
    if config.get("job.distributed.num_workers_machine") <= 0:
        warnings.warn("Number of workers for this specific machine not defined. "
                      "Using default floor(num_workers / num_machines).")
        config.set("job.distributed.num_workers_machine", int(config.get("job.distributed.num_workers")/config.get("job.distributed.num_machines")))
    # setting already initialized workers if < 0
    if config.get("job.distributed.already_init_workers") < 0:
        config.set("job.distributed.already_init_workers",
                   config.get("job.distributed.machine_id") * config.get("job.distributed.num_workers_machine"))
    # specific settings for valid only jobs
    if config.get("job.type") in ["valid", "test", "eval"]:
        config.set("job.distributed.parameter_server", "shared")
        num_eval_workers = config.get("job.distributed.num_eval_workers")
        config.set("job.distributed.num_workers", num_eval_workers)
        config.set("job.distributed.num_workers_machine", num_eval_workers)
        config.set("job.distributed.num_machines", 1)
        config.set("job.distributed.gloo_socket_ifname", "lo")
        config.set("job.distributed.master_ip", "127.0.0.1")
        config.set(f"{config.get('model')}.create_eval", True)
    return config


def create_and_run_distributed(
    config: Config, dataset: Optional[Dataset] = None, checkpoint: Optional[Dict] = None
):
    config = update_config_for_distributed(config)

    if _should_run_single_process(config):
        return _run_single_process_distributed(config, dataset, checkpoint)

    os.environ["OMP_NUM_THREADS"] = str(
        config.get("job.distributed.num_threads_per_process")
    )
    os.environ["GLOO_SOCKET_IFNAME"] = config.get("job.distributed.gloo_socket_ifname")

    if (
        config.get("job.distributed.repartition_epoch")
        and config.get("job.distributed.partition_type") == "stratification"
    ):
        # with stratificaton we have a lot of open files that need to be shared
        # between processes. Some servers don't allow that. Therefore set sharing
        # strategy to file_system to avoid too many open files error
        torch.multiprocessing.set_sharing_strategy("file_system")

    # catch interrupt (to shut down lapse and other processes)
    processes = []
    monitoring_processes = []
    worker_process_pool = None

    def kill_processes(signal_received, frame):
        print("\nSIGINT or CTRL-C detected. Shutting down all processes and exiting...")
        for process in processes:
            if process is not None:
                try:
                    process.kill()
                except AttributeError:
                    print("process already killed")
        for process in monitoring_processes:
            if process is not None:
                process.kill()
        if worker_process_pool is not None:
            worker_process_pool.kill()
        exit(0)
    signal(SIGINT, kill_processes)

    job_device = str(config.get("job.device"))
    use_gpu = job_device.startswith("cuda")

    if config.get("job.type") == "train":
        # start hardware monitoring
        monitor_process = mp.Process(
            target=monitor_hardware, args=(config.folder, 0.5), daemon=True
        )
        monitoring_processes.append(monitor_process)
        monitor_process.start()
        if use_gpu:
            gpu_monitor_process = mp.Process(
                target=monitor_gpus, args=(config.folder, 1), daemon=True
            )
            monitoring_processes.append(gpu_monitor_process)
            gpu_monitor_process.start()

    worker_pool_holder = {"pool": None, "error": None}

    def _spawn_workers():
        try:
            worker_pool_holder["pool"] = WorkerProcessPool(
                config,
                dataset,
                checkpoint,
            )
        except Exception as exc:
            worker_pool_holder["error"] = exc

    worker_thread = threading.Thread(
        target=_spawn_workers, name="worker-pool-spawn", daemon=True
    )
    worker_thread.start()

    if config.get("job.distributed.machine_id") == 0:
        num_keys = get_num_keys(config, dataset)
        if config.get("job.distributed.parameter_server") == "lapse":
            p = mp.Process(
                target=init_lapse_scheduler,
                args=(
                    config,
                    num_keys,
                ),
                daemon=True,
            )
            processes.append(p)
            p.start()
        elif config.get("job.distributed.parameter_server") == "torch":
            p = mp.Process(
                target=init_torch_server,
                args=(
                    config,
                    num_keys,
                ),
                daemon=True,
            )
            processes.append(p)
            p.start()

        # create a work scheduler
        print("init scheduler")
        scheduler_init_time = time.time()
        scheduler = WorkScheduler.create(config=config, dataset=dataset)
        config.log(f"scheduler initialized after: {time.time()-scheduler_init_time}")
        print("start scheduler")
        scheduler_start_time = time.time()
        processes.append(scheduler)
        scheduler.start()
        config.log(f"scheduler start took: {time.time()-scheduler_start_time}")

    worker_thread.join()
    if worker_pool_holder["error"] is not None:
        raise worker_pool_holder["error"]
    worker_process_pool = worker_pool_holder["pool"]
    valid_trace = worker_process_pool.join()
    for p in processes:
        p.join()

    if config.get("job.type") == "train":
        monitor_process.terminate()
        monitor_process.join(timeout=1.0)
        if use_gpu:
            gpu_monitor_process.terminate()
            gpu_monitor_process.join(timeout=1.0)
    return valid_trace


def _should_run_single_process(config: Config) -> bool:
    if not config.get("job.distributed.single_process"):
        return False
    try:
        num_workers = int(config.get("job.distributed.num_workers"))
    except (TypeError, ValueError):
        return False
    if num_workers != 1:
        return False
    try:
        num_machines = int(config.get("job.distributed.num_machines"))
    except (TypeError, ValueError):
        return False
    if num_machines != 1:
        return False
    if str(config.get("job.distributed.parameter_server")) != "shared":
        return False
    return True


def _run_single_process_distributed(
    config: Config, dataset: Optional[Dataset], checkpoint: Optional[Dict]
):
    if dataset is None:
        raise ValueError("Dataset must be provided for single-process mode.")
    job_device = str(config.get("job.device"))
    use_gpu = job_device.startswith("cuda")
    if config.get("job.type") == "train":
        monitor_process = mp.Process(
            target=monitor_hardware, args=(config.folder, 0.5), daemon=True
        )
        monitor_process.start()
        gpu_monitor_process = None
        if use_gpu:
            gpu_monitor_process = mp.Process(
                target=monitor_gpus, args=(config.folder, 1), daemon=True
            )
            gpu_monitor_process.start()
    num_keys = get_num_keys(config, dataset)
    embedding_dim = config.get("lookup_embedder.dim")
    optimizer_dim = get_optimizer_dim(config, embedding_dim)
    # Zero-init shared parameters to avoid adding into uninitialized memory
    parameters = torch.zeros(
        (num_keys, embedding_dim + optimizer_dim),
        dtype=torch.float32,
        requires_grad=False,
    )
    device_pool = list(config.get("job.device_pool") or [])
    if device_pool:
        config.set("job.device", device_pool[0])
    parameter_client = KgeParameterClient.create(
        config=config,
        server_id=0,
        client_id=get_min_rank(config),
        server=parameters,
        num_keys=num_keys,
    )
    scheduler_client = LocalSchedulerClient(config=config, dataset=dataset)
    init_for_load_only = checkpoint is not None
    job = Job.create(
        config=config,
        dataset=dataset,
        parameter_client=parameter_client,
        work_scheduler_client=scheduler_client,
        init_for_load_only=init_for_load_only,
    )
    if checkpoint is not None:
        job.load_distributed(checkpoint_name=checkpoint)
    job.run()
    if hasattr(job, "work_scheduler_client"):
        job.work_scheduler_client.shutdown()
    parameter_client.shutdown()
    valid_trace = getattr(job, "valid_trace", None)
    if config.get("job.type") == "train":
        monitor_process.terminate()
        monitor_process.join(timeout=1.0)
        if use_gpu and gpu_monitor_process is not None:
            gpu_monitor_process.terminate()
            gpu_monitor_process.join(timeout=1.0)
    return valid_trace
