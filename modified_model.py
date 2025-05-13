import torch
import modified_nn
import utils
from modified_nn import *
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType, EdgeType
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.stats import StatType
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, BCELoss, L1Loss, Embedding, ModuleDict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Optional, Any, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import get_node_attribute_dict


class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        node_col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        edge_col_stats_dict: Dict[Tuple[str, str, str], Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        node_channels: int,
        edge_channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder_with_edge_features(
            node_channels=node_channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=node_col_stats_dict,
            edge_channels=edge_channels,
            edge_to_col_names_dict={
                edge_type: data[edge_type].tf.col_names_dict
                for edge_type in data.edge_types
            },
            edge_to_col_stats=edge_col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder_with_edge_features(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            node_channels=node_channels,
            edge_types=[
                edge_type for edge_type in data.edge_types if "time" in data[edge_type]
            ],
            edge_channels=edge_channels
        )
        self.gnn = HeteroGAT(
            node_types=data.node_types,
            edge_types=data.edge_types,
            node_channels=node_channels,
            edge_channels=edge_channels,
            aggr=aggr,
            num_layers=num_layers,
        )

        # For now, we don't incorporate edge features as input of the head
        self.head = torch.nn.Sequential(
            MLP(
                node_channels,
                out_channels=out_channels,
                norm=norm,
                num_layers=1,
            ),
            torch.nn.Sigmoid()
        )
        # For now, we don't incorporate the case where we have also shallow embeddings of edges
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], node_channels)
                for node in shallow_list
            }
        )

        # This should be modified to handle edges with id (those that are associated to a row in a table)
        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, node_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        for layer in self.head.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        node_tf_dict = dict()
        edge_tf_dict = dict()
        for key in batch.node_types:
            node_tf_dict[key] = batch[key].tf
        for key in batch.edge_types:
            edge_tf_dict[key] = batch[key].tf
        x_dict, edge_emb_dict = self.encoder(node_tf_dict=node_tf_dict, edge_tf_dict=edge_tf_dict)

        node_rel_time_dict, edge_rel_time_dict = self.temporal_encoder(
            seed_time=seed_time,
            node_time_dict = utils.get_node_attribute_dict(batch, "time"),
            node_batch_dict = utils.get_node_attribute_dict(batch, "batch"),
            edge_time_dict= utils.get_edge_attribute_dict(batch, "time"),
            edge_batch_dict= utils.get_edge_attribute_dict(batch, "batch")
        )

        for node_type, rel_time in node_rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for edge_type, rel_time in edge_rel_time_dict.items():
            edge_emb_dict[edge_type] = edge_emb_dict[edge_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict = x_dict,
            edge_index_dict = batch.edge_index_dict,
            edge_attr_dict = edge_emb_dict,
            num_sampled_nodes_dict = batch.num_sampled_nodes_dict,
            num_sampled_edges_dict = batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])