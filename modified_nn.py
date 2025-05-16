from typing import Any, Dict, List, Optional, Tuple

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv
from torch_geometric.typing import EdgeType, NodeType, EdgeTypeStr
from typing_extensions import Union

"""
def edge_type_to_valid_module_name(edge_type):
    return "__".join(edge_type)
"""




class HeteroEncoder_with_edge_features(torch.nn.Module):
    r"""HeteroEncoder based on PyTorch Frame.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """

    def __init__(
        self,
        node_channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        edge_channels: Optional[int] = None,
        edge_to_col_names_dict: Optional[Dict[EdgeType, Dict[str, List[str]]]] = None,
        edge_to_col_stats: Optional[Dict[EdgeType, Dict[str, Dict[StatType, Any]]]] = None,
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.node_encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=node_channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.node_encoders[node_type] = torch_frame_model

        self.edge_encoders = torch.nn.ModuleDict()
        if edge_to_col_names_dict is not None:
            for edge_type in edge_to_col_names_dict.keys():

                stype_encoder_dict = {
                    stype: default_stype_encoder_cls_kwargs[stype][0](
                        **default_stype_encoder_cls_kwargs[stype][1]
                    )
                    for stype in edge_to_col_names_dict[edge_type].keys()
                }
                torch_frame_model = torch_frame_model_cls(
                    **torch_frame_model_kwargs,
                    out_channels=edge_channels,
                    col_stats=edge_to_col_stats[edge_type],
                    col_names_dict=edge_to_col_names_dict[edge_type],
                    stype_encoder_dict=stype_encoder_dict,
                )
                edge_type_key = EdgeTypeStr(edge_type)
                self.edge_encoders[edge_type_key] = torch_frame_model

    def reset_parameters(self):
        for node_encoder in self.node_encoders.values():
            node_encoder.reset_parameters()
        for edge_encoder in self.edge_encoders.values():
            edge_encoder.reset_parameters()

    def forward(
        self,
        node_tf_dict: Dict[NodeType, torch_frame.TensorFrame],
        edge_tf_dict: Optional[Dict[EdgeType, torch_frame.TensorFrame]] = None,
    ) -> Union[Dict[NodeType, Tensor], Tuple[Dict[NodeType, Tensor],Dict[EdgeType, Tensor]]]:
        x_dict = {
            node_type: self.node_encoders[node_type](tf) for node_type, tf in node_tf_dict.items()
        }
        if edge_tf_dict is not None:
            edge_attr_dict = {
                edge_type: self.edge_encoders[EdgeTypeStr(edge_type)](tf) for edge_type, tf in edge_tf_dict.items()
            }
            return x_dict, edge_attr_dict
        return x_dict



class HeteroTemporalEncoder_with_edge_features(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            node_channels: int,
            edge_types: Optional[List[EdgeType]] = None,
            edge_channels: Optional[int] = None,
    ):
        super().__init__()

        self.node_encoder_dict = torch.nn.ModuleDict(
            {node_type: PositionalEncoding(node_channels) for node_type in node_types}
        )
        self.node_lin_dict = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(node_channels, node_channels) for node_type in node_types}
        )
        self.edge_encoder_dict = torch.nn.ModuleDict()
        self.edge_lin_dict = torch.nn.ModuleDict()
        if edge_types is not None:
            for edge_type in edge_types:
                self.edge_encoder_dict[EdgeTypeStr(edge_type)] = PositionalEncoding(edge_channels)
                self.edge_lin_dict[EdgeTypeStr(edge_type)] = torch.nn.Linear(edge_channels, edge_channels)

    def reset_parameters(self):
        for encoder in self.node_encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.node_lin_dict.values():
            lin.reset_parameters()
        for encoder in self.edge_encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.edge_lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        node_time_dict: Dict[NodeType, Tensor],
        node_batch_dict: Dict[NodeType, Tensor],
        edge_time_dict: Optional[Dict[EdgeType, Tensor]] = None,
        edge_index_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Union[Dict[NodeType, Tensor],Tuple[Dict[NodeType, Tensor], Dict[EdgeType, Tensor]]]:
        node_out_dict: Dict[NodeType, Tensor] = {}
        edge_out_dict: Dict[EdgeType, Tensor] = {}

        for node_type, time in node_time_dict.items():
            rel_time = seed_time[node_batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.
            x = self.node_encoder_dict[node_type](rel_time)
            x = self.node_lin_dict[node_type](x)
            node_out_dict[node_type] = x

        if edge_time_dict is not None:
            for edge_type, time in edge_time_dict.items():
                rel_time = seed_time[node_batch_dict[edge_type[0]][edge_index_dict[edge_type][0]]] - time
                rel_time = rel_time / (60 * 60 * 24)
                x = self.edge_encoder_dict[EdgeTypeStr(edge_type)](rel_time)
                x = self.edge_lin_dict[EdgeTypeStr(edge_type)](x)
                edge_out_dict[edge_type] = x
            return node_out_dict, edge_out_dict
        return node_out_dict

class HeteroGAT(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            node_channels: int,
            edge_channels: Optional[int] = None,
            num_layers: int = 2,
            aggr: str = "mean",
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        in_channels = (node_channels, node_channels),
                        out_channels = node_channels,
                        edge_dim = edge_channels,
                        aggr = aggr,
                        add_self_loops = False,
                    )
                    for edge_type in edge_types
                },
                aggr = "sum"
            )
            self.convs.append(conv)
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(node_channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()


    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            edge_attr_dict: Dict[EdgeType, Tensor] = None,
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            if edge_attr_dict is not None:
                x_dict = conv(
                    x_dict,
                    edge_index_dict,
                    edge_attr_dict = edge_attr_dict
                )
            else:
                x_dict = conv(
                    x_dict,
                    edge_index_dict
                )
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict