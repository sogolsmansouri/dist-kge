"""Distributed helpers with lazy imports to avoid circular dependencies."""

import importlib

__all__ = [
    "WorkerProcess",
    "WorkerProcessPool",
    "KgeParameterClient",
    "TorchParameterClient",
    "LapseParameterClient",
    "KgeParameterServer",
    "TorchParameterServer",
    "LapseParameterServer",
    "WorkScheduler",
    "SchedulerClient",
]

_LAZY_IMPORTS = {
    "WorkerProcess": ("kge.distributed.worker_process", "WorkerProcess"),
    "WorkerProcessPool": ("kge.distributed.worker_process", "WorkerProcessPool"),
    "KgeParameterClient": ("kge.distributed.parameter_client", "KgeParameterClient"),
    "TorchParameterClient": ("kge.distributed.parameter_client", "TorchParameterClient"),
    "LapseParameterClient": ("kge.distributed.parameter_client", "LapseParameterClient"),
    "KgeParameterServer": ("kge.distributed.parameter_server", "KgeParameterServer"),
    "TorchParameterServer": ("kge.distributed.parameter_server", "TorchParameterServer"),
    "LapseParameterServer": ("kge.distributed.parameter_server", "LapseParameterServer"),
    "WorkScheduler": ("kge.distributed.work_scheduler", "WorkScheduler"),
    "SchedulerClient": ("kge.distributed.work_scheduler", "SchedulerClient"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(__all__) + list(globals().keys()))
