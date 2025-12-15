import os
import json
import argparse
import dgl
import torch
import numpy as np
import scipy as sp
from pathlib import Path
from util import read_data


def compute_entity_degrees(src, dst, num_entities):
    """Return total degree (in+out) per entity."""
    out_deg = np.bincount(src, minlength=num_entities)
    in_deg = np.bincount(dst, minlength=num_entities)
    return in_deg + out_deg


def compute_relation_counts(rel_ids, num_relations):
    return np.bincount(rel_ids, minlength=num_relations)


def select_hot_entities(entity_degrees, percent):
    if percent <= 0.0:
        return None
    target = max(1, int(len(entity_degrees) * percent))
    if target >= len(entity_degrees):
        hot = np.argsort(entity_degrees)[::-1]
    else:
        candidates = np.argpartition(entity_degrees, -target)[-target:]
        hot = candidates[np.argsort(entity_degrees[candidates])[::-1]]
    return hot


def compute_triple_costs(src, dst, rel, entity_degrees, relation_counts, alpha, beta, neg_factor):
    costs = np.ones(len(src), dtype=np.float32)
    if alpha != 0.0:
        costs += alpha * (entity_degrees[src] + entity_degrees[dst]).astype(np.float32)
    if beta != 0.0:
        costs += beta * relation_counts[rel].astype(np.float32)
    if neg_factor != 0.0:
        neg_load = (entity_degrees[src] * entity_degrees[dst]).astype(np.float32)
        costs += neg_factor * neg_load
    return costs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of dataset to partition")
    parser.add_argument("-n", "--num_partitions", type=int, help="number of partitions")
    parser.add_argument(
        "--cost-alpha",
        type=float,
        default=0.0,
        help="Coefficient for entity-degree contribution to per-triple cost.",
    )
    parser.add_argument(
        "--cost-beta",
        type=float,
        default=0.0,
        help="Coefficient for relation-frequency contribution to per-triple cost.",
    )
    parser.add_argument(
        "--cost-neg-factor",
        type=float,
        default=0.0,
        help="Coefficient for negative-sampling load (degree product) contribution to per-triple cost.",
    )
    parser.add_argument(
        "--hot-entity-percent",
        type=float,
        default=0.0,
        help="Percent of entities to mark as hot (0 disables).",
    )
    parser.add_argument(
        "--export-triple-costs",
        action="store_true",
        help="Write the per-triple cost array to disk (large).",
    )
    args = parser.parse_args()
    dataset_folder = args.dataset
    num_parts = args.num_partitions
    write=True
    data, entities, relations = read_data(dataset_folder, train=True, entity_ids=True, relation_ids=True)


    src_all = data[:, 0]
    dst_all = data[:, 2]
    rel_all = data[:, 1]
    num_entities = len(entities)
    num_relations = len(relations)
    print("computing graph statistics...")
    entity_degrees = compute_entity_degrees(src_all, dst_all, num_entities)
    relation_counts = compute_relation_counts(rel_all, num_relations)
    hot_entities = select_hot_entities(entity_degrees, args.hot_entity_percent)
    triple_costs = None
    if args.cost_alpha != 0.0 or args.cost_beta != 0.0 or args.cost_neg_factor != 0.0:
        print("computing per-triple costs...")
        triple_costs = compute_triple_costs(
            src_all,
            dst_all,
            rel_all,
            entity_degrees,
            relation_counts,
            args.cost_alpha,
            args.cost_beta,
            args.cost_neg_factor,
        )
    coo = sp.sparse.coo_matrix((np.ones(data.shape[0]), (src_all, dst_all)),
                               shape=[num_entities, num_entities])

    triple_partition_assignment = np.full((len(data)), -1, dtype=np.int64)
    entity_partition_assignment = np.full((num_entities), -1, dtype=np.int64)
    partition_triple_order = {i: [] for i in range(num_parts)}
    num_inner_edges_dict = {}
    inner_nodes_dict = {}
    print("construct graph...")
    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
    g.edata['tid'] = torch.from_numpy(data[:, 1])
    print('partition graph...')
    part_dict = dgl.transform.metis_partition(g, num_parts, 1)

    node_part_mapper = np.zeros(num_entities)-1

    tot_num_inner_edges = 0
    print("partition\t|\tentities needed\t|\tentities located in partition\t|\tnum triples\t|\tinner triples\t|\tnum outside partition acesses\t|\tpercent outside partition acesses")
    print("--------:\t|\t----:\t|\t--------:\t|\t------:\t|\t----------:\t|\t---------:\t|\t---------:")
    for part_id in part_dict:
        part = part_dict[part_id]
        #print(part.has_nodes(entities))
        part_src, part_dst = part.all_edges(form='uv', order='eid')
        #print(part_src)
        triple_partition_assignment[part.edata["_ID"].numpy()] = part_id
        partition_triple_order[part_id].extend(part.edata["_ID"].numpy().tolist())
        #entity_partition_assignment[part.ndata["_ID"].numpy()] = part_id


        num_inner_nodes = len(np.nonzero(part.ndata['inner_node'].numpy())[0])
        num_inner_edges = len(np.nonzero(part.edata['inner_edge'].numpy())[0])
        num_inner_edges_dict[part_id] = num_inner_edges
        outside_partition_accesses = part.number_of_edges() - num_inner_edges
        print("{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t".format(
            part_id, part.number_of_nodes(), num_inner_nodes,
            part.number_of_edges(), num_inner_edges,
            outside_partition_accesses,
            outside_partition_accesses/part.number_of_edges()
        ))
        tot_num_inner_edges += num_inner_edges

        part.copy_from_parent()
        parent_inner_nodes = part.parent_nid[part.ndata["inner_node"].numpy().astype(bool)]
        inner_nodes_dict[part_id] = parent_inner_nodes
        mapper = node_part_mapper[parent_inner_nodes]
        mapper[mapper==-1] = part_id
        #print(mapper)
        #node_part_mapper[part.parent_nid] = mapper
        entity_partition_assignment[parent_inner_nodes] = mapper
        #print(node_part_mapper)

    partition_counts = np.bincount(triple_partition_assignment, minlength=num_parts)
    partition_costs = None
    if triple_costs is not None:
        partition_costs = np.bincount(
            triple_partition_assignment, weights=triple_costs, minlength=num_parts
        )
        print("per-partition weighted cost:", partition_costs.tolist())
    else:
        partition_costs = partition_counts.astype(np.float64)

    # write to file

    if write:
        partition_folder = os.path.join(dataset_folder, "partitions","graph-cut")
        output_folder = os.path.join(partition_folder, f"num_{num_parts}")
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        print("write to file")
        np.savetxt(
            os.path.join(output_folder, "train_assign_partitions.del"),
            triple_partition_assignment,
            delimiter="\t",
            fmt="%s",
        )
        np.savetxt(
            os.path.join(output_folder, "entity_to_partitions.del"),
            entity_partition_assignment,
            delimiter="\t",
            fmt="%s",
        )
        np.save(
            os.path.join(output_folder, "partition_counts.npy"),
            partition_counts,
        )
        np.save(
            os.path.join(output_folder, "partition_costs.npy"),
            partition_costs,
        )
        reordered = {}
        for part_id in range(num_parts):
            idx = np.array(partition_triple_order[part_id], dtype=np.int64)
            if len(idx) == 0:
                reordered[f"part_{part_id}"] = idx
                continue
            order = np.lexsort((rel_all[idx], dst_all[idx], src_all[idx]))
            reordered[f"part_{part_id}"] = idx[order]
        np.savez(os.path.join(output_folder, "partition_triples.npz"), **reordered)

    analysis_prefix = Path(dataset_folder)
    np.save(analysis_prefix / "analysis_entity_degrees.npy", entity_degrees)
    np.save(analysis_prefix / "analysis_relation_counts.npy", relation_counts)
    if hot_entities is not None:
        np.save(analysis_prefix / "analysis_hot_entities.npy", hot_entities)
    if triple_costs is not None and args.export_triple_costs:
        np.save(
            analysis_prefix / f"analysis_triple_costs_num_{num_parts}.npy",
            triple_costs,
        )
    summary = {
        "num_triples": int(len(data)),
        "num_entities": int(num_entities),
        "num_relations": int(num_relations),
        "max_entity_degree": int(entity_degrees.max()) if len(entity_degrees) else 0,
        "median_entity_degree": float(np.median(entity_degrees))
        if len(entity_degrees)
        else 0.0,
        "max_relation_frequency": int(relation_counts.max())
        if len(relation_counts)
        else 0,
        "median_relation_frequency": float(np.median(relation_counts))
        if len(relation_counts)
        else 0.0,
        "hot_entity_percent": args.hot_entity_percent,
        "num_hot_entities": int(len(hot_entities)) if hot_entities is not None else 0,
        "cost_alpha": args.cost_alpha,
        "cost_beta": args.cost_beta,
        "cost_neg_factor": args.cost_neg_factor,
        "partition_counts": partition_counts.tolist(),
        "partition_costs": partition_costs.tolist()
        if partition_costs is not None
        else None,
    }
    summary_path = analysis_prefix / "analysis_graphcut_summary.json"
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    summary_num_path = (
        analysis_prefix / f"analysis_graphcut_summary_num_{num_parts}.json"
    )
    with open(summary_num_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print("some counting")
    unique_pairs, counts = np.unique(
        triple_partition_assignment, return_counts=True, axis=0
    )
    print(unique_pairs)
    print(counts)

    print(triple_partition_assignment)
