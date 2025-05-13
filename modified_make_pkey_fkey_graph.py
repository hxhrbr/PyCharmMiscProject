import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.modeling.utils import remove_pkey_fkey, to_unix_time
from typing import Any, Dict, NamedTuple, Optional, Tuple, Set

def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    edge_tables: Optional[Set[Any]] = None,
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    add_effective_time: bool = False,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]], Dict[Tuple[str, str, str], Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
    foreign key relationships, together with the column stats of each table.

    Args:
        db: A database object containing a set of tables.
        col_to_stype_dict: Column to stype for
            each table.
        edge_tables: A set of tables that we want to treat as tables of edges.
            These tables are assumed to have exactly two foreign fields.
            We require the two foreign fields to be, for each row, filled
            with valid indices of the tables they refer to.
        text_embedder_cfg: Text embedder config.
        cache_dir: A directory for storing materialized tensor
            frames. If specified, we will either cache the file or use the
            cached file. If not specified, we will not use cached file and
            re-process everything from scratch without saving the cache.
        add_effective_time: The introduction of this argument is motivated
            by the fact that currently NegborLoader doesn't support filtering
            by timestamp for heterogenous data if both edges and nodes have a
            time field. Setting add_effective_time to True we work around
            this limitation.
            If add_effective_time is set to True, we assign for each edge a field
            effective_time that corresponds to the maximum of the timestamps
            of the edge itself and of the incident nodes (if none of these types
            has an associated timestamp the field is not created).
    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
    """
    data = HeteroData()
    node_col_stats_dict = dict()
    edge_col_stats_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    if edge_tables is None:
        edge_tables = set()

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"meaningless_const": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"meaningless_const": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)


        if table_name not in edge_tables:

            data[table_name].tf = dataset.tensor_frame
            node_col_stats_dict[table_name] = dataset.col_stats

            # Add time attribute:
            if table.time_col is not None:
                data[table_name].time = torch.from_numpy(
                    to_unix_time(table.df[table.time_col])
                )

            # Add edges:
            for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
                pkey_index = df[fkey_name]
                # Filter out dangling foreign keys
                mask = ~pkey_index.isna()
                fkey_index = torch.arange(len(pkey_index))
                # Filter dangling foreign keys:
                pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
                fkey_index = fkey_index[torch.from_numpy(mask.values)]
                # Ensure no dangling fkeys
                assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

                # fkey -> pkey edges
                edge_index = torch.stack([fkey_index, pkey_index], dim=0)
                edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
                data[edge_type].edge_index = sort_edge_index(edge_index)

                # pkey -> fkey edges.
                # "rev_" is added so that PyG loader recognizes the reverse edges
                edge_index = torch.stack([pkey_index, fkey_index], dim=0)
                edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
                data[edge_type].edge_index = sort_edge_index(edge_index)
        else:
            # We want the only two foreign keys to represent the endpoints of the edge
            assert len(table.fkey_col_to_pkey_table) == 2

            fkey_name_0, pkey_table_name_0 = list(table.fkey_col_to_pkey_table.items())[0]
            fkey_name_1, pkey_table_name_1 = list(table.fkey_col_to_pkey_table.items())[1]

            index_0 = df[fkey_name_0]
            index_1 = df[fkey_name_1]
            # Check if data is valid by ensuring evey foreign reference is a number
            assert not (index_0.isna() | index_1.isna()).any()
            # and that foreign references are not dangling
            assert (index_0 < len(db.table_dict[pkey_table_name_0])).all()
            assert (index_1 < len(db.table_dict[pkey_table_name_1])).all()

            edge_type = (pkey_table_name_0, table_name, pkey_table_name_1)

            if table.time_col is not None:
                data[edge_type].time = torch.from_numpy(
                    to_unix_time(table.df[table.time_col])
                )

            edge_col_stats_dict[edge_type] = dataset.col_stats
            index_0 = torch.from_numpy(index_0.astype(int).values)
            index_1 = torch.from_numpy(index_1.astype(int).values)
            edge_index = torch.stack([index_0, index_1], dim=0)
            data[edge_type].edge_index,  perm = sort_edge_index(edge_index, torch.tensor([i for i in range(len(index_0))]))
            data[edge_type].tf = dataset.tensor_frame[perm]

            edge_type = (pkey_table_name_1, f"rev_{table_name}", pkey_table_name_0)

            if table.time_col is not None:
                data[edge_type].time = torch.from_numpy(
                    to_unix_time(table.df[table.time_col])
                )

            edge_col_stats_dict[edge_type] = dataset.col_stats
            edge_index = torch.stack([index_1, index_0], dim=0)
            data[edge_type].edge_index, perm = sort_edge_index(edge_index, torch.tensor([i for i in range(len(index_1))]))
            data[edge_type].tf = dataset.tensor_frame[perm]

    if add_effective_time:
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            src_idx = data[edge_type].edge_index[0]
            dst_idx = data[edge_type].edge_index[1]

            times = []

            if "time" in data[edge_type]:
                times.append(data[edge_type].time)
            if "time" in data[src_type]:
                times.append(data[src_type].time[src_idx])
            if "time" in data[dst_type]:
                times.append(data[dst_type].time[dst_idx])

            if times:
                data[edge_type].effective_time = torch.stack(times, dim=0).max(dim=0).values

    data.validate()

    return data, node_col_stats_dict, edge_col_stats_dict


