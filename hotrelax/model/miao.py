# TODO new save, something like
# torch.save({
#    'model_state_dict': model.state_dict(),
#        'some_param': model.some_param  # 保存字典参数
#        }, 'model_with_dict.pth')
#

from typing import Callable, List, Dict, Optional, Literal, Tuple
import torch
from torch import nn
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import NonLinearLayer, GraphConvLayer, SelfInteractionLayer
from ..utils import find_distances, _scatter_add, _scatter_mean, res_add, TensorAggregateOP


class UpdateNodeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 adddim_way     : Dict[int, int] = {0: 9, 1: 3, 2: 1, 3: 0},
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 ) -> None:
        super().__init__()
        self.graph_conv = GraphConvLayer(radial_fn=radial_fn,
                                         input_dim=input_dim,
                                         output_dim=output_dim,
                                         max_in_way=max_in_way,
                                         max_out_way=max_out_way,
                                         max_r_way=max_r_way,
                                         conv_mode=conv_mode,
                                         adddim_way=adddim_way,
                                         )
        self.self_interact = SelfInteractionLayer(input_dim=input_dim,
                                                  max_way=max_out_way,
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)
        self.register_buffer("norm_factor", torch.tensor(norm_factor))

    def forward(
        self,
        node_info    : Dict[int, torch.Tensor],
        edge_info    : Dict[int, torch.Tensor],
        batch_data   : Dict[str, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        message = self.graph_conv(
            node_info=node_info, edge_info=edge_info, batch_data=batch_data
        )
        res_info = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data["idx_i"]
        n_atoms = batch_data['atomic_number'].shape[0]
        for way in message.keys(): 
            res_info[way] = (
                _scatter_add(message[way], idx_i, dim_size=n_atoms) / self.norm_factor
                )
        res_info = self.non_linear(self.self_interact(res_info))
        return res_add(node_info, res_info)


class UpdateEdgeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 ) -> None:
        super().__init__()
        self.graph_conv = GraphConvLayer(radial_fn=radial_fn,
                                         input_dim=input_dim,
                                         output_dim=output_dim,
                                         max_in_way=max_in_way,
                                         max_out_way=max_out_way,
                                         max_r_way=max_r_way,
                                         conv_mode=conv_mode,)
        self.self_interact = SelfInteractionLayer(input_dim=input_dim,
                                                  max_way=max_out_way,
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = self.non_linear(self.self_interact(message))
        return res_add(edge_info, res_info)

class MiaoBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 adddim_way     : Dict[int, int] = {0: 9, 1: 3, 2: 0, 3: 0},
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 update_edge    : bool=True,
                 ) -> None:
        super().__init__()
        self.node_block = UpdateNodeBlock(radial_fn=radial_fn, 
                                          max_r_way=max_r_way, 
                                          max_in_way=max_in_way,
                                          max_out_way=max_out_way,
                                          input_dim=input_dim, 
                                          output_dim=output_dim,
                                          adddim_way=adddim_way,
                                          norm_factor=norm_factor, 
                                          activate_fn=activate_fn,
                                          conv_mode=conv_mode,
                                          )
        if update_edge:
            self.edge_block = UpdateEdgeBlock(radial_fn=radial_fn, 
                                              max_r_way=max_r_way, 
                                              max_in_way=max_in_way,
                                              max_out_way=max_out_way,
                                              input_dim=input_dim,
                                              output_dim=output_dim,
                                              activate_fn=activate_fn,
                                              conv_mode=conv_mode,
                                              )
        else:
            self.edge_block = None

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        node_info = self.node_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        if self.edge_block is not None:
            edge_info = self.edge_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        return node_info, edge_info

class MiaoNet(nn.Module):
    """
    Miao nei ga
    duo xi da miao nei
    """
    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 n_layers        : int,
                 max_r_way       : List[int],
                 max_out_way     : List[int],
                 output_dim      : List[int],
                 activate_fn     : str="silu",
                 adddim_way      : Dict[int, int] = {0: 9, 1: 3, 2: 0, 3: 0},
                 target_way      : Dict[str, int]={"site_energy": 0},
                 mean_pos        : float=0.,
                 mean_cell       : float=1.,
                 std_pos         : float=1.,
                 std_cell        : float=1.,
                 norm_factor     : float=1.,
                 bilinear        : bool=False,
                 conv_mode       : Literal['node_j', 'node_edge']='node_j',
                 update_edge     : bool=False,
                 use_graph       : bool=False,
                 use_cycle       : bool=False,
                 graph_feat_mean : float=0.,
                 graph_feat_std  : float=1.,
                 ):
        super().__init__()
        self.register_buffer("mean_pos", torch.tensor(mean_pos).float())
        self.register_buffer("mean_cell", torch.tensor(mean_cell).float())
        self.register_buffer("std_pos", torch.tensor(std_pos).float())
        self.register_buffer("std_cell", torch.tensor(std_cell).float())
        self.register_buffer("graph_feat_mean", torch.tensor(graph_feat_mean).float())
        self.register_buffer("graph_feat_std", torch.tensor(graph_feat_std).float())
        self.embedding_layer = embedding_layer
        self.radial_fn = radial_fn
        self.target_way = target_way
        self.use_graph = use_graph
        self.use_cycle = use_cycle

        max_in_way = [0] + max_out_way[:-1]
        hidden_nodes = [embedding_layer.n_channel] + output_dim
        self.en_equivalent_blocks = self.get_eq_blocks(
            activate_fn,
            max_r_way,
            max_in_way,
            max_out_way,
            hidden_nodes,
            norm_factor,
            conv_mode,
            update_edge,
            n_layers,
            adddim_way,
        )

        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                        target_way=target_way,
                                        activate_fn=activate_fn,
                                        bilinear=bilinear,
                                        e_dim=embedding_layer.n_channel,
                                        )
        self.graph_feat_encoder = nn.Sequential(
            nn.LazyLinear(hidden_nodes[-1]),
            nn.SiLU(),
            nn.Linear(hidden_nodes[-1], hidden_nodes[-1]),
            nn.SiLU(),
        )
        self.graph_context_layer = nn.Sequential(
            nn.Linear(hidden_nodes[-1] * 2, hidden_nodes[-1]),
            nn.SiLU(),
        )
        self.graph_pos_layer = nn.Sequential(
            nn.Linear(hidden_nodes[-1] * 2, hidden_nodes[-1]),
            nn.SiLU(),
            nn.Linear(hidden_nodes[-1], 3),
        )
        self.graph_cell_layer = nn.Sequential(
            nn.Linear(hidden_nodes[-1], hidden_nodes[-1]),
            nn.SiLU(),
            nn.Linear(hidden_nodes[-1], 9),
        )
        # TensorAggregateOP.set_max(max(max_in_way), max(max_out_way), max(max_r_way))

    def forward(self,
                batch_data   : Dict[str, torch.Tensor],
                properties   : List[str] | None = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Run the Miao network forward pass.

        Args:
            batch_data: Batched structure tensors.
            properties: Requested property names.
            create_graph: Whether to build higher-order autograd graph.

        Returns:
            Batch dictionary with prediction tensors appended.
        """
        batch_data["direct_pos_p"] = []
        batch_data["direct_cell_p"] = []

        idx_i, idx_j = batch_data["idx_i"], batch_data["idx_j"]
        abc_unsqueeze = batch_data["cell_u"].repeat_interleave(batch_data["n_edges"], dim=0)
        batch_data["rij"] = batch_data["pos_u"][idx_j] + torch.einsum("a p, a p v -> a v", \
                            batch_data["offset"], abc_unsqueeze) - batch_data["pos_u"][idx_i]

        node_info, edge_info = self.get_init_info(batch_data)
        for en_equivalent in self.en_equivalent_blocks:
            node_info, edge_info = en_equivalent(node_info, edge_info, batch_data)
        output_tensors = self.readout_layer(node_info, batch_data)

        direct_pos = output_tensors["direct_pos"] * self.std_pos + self.mean_pos
        direct_cell = _scatter_add(output_tensors['direct_cell'], batch_data['batch']) * self.std_cell + self.mean_cell

        if self.use_graph:
            graph_context = self.get_graph_context(node_info, batch_data)
            direct_pos = direct_pos + self.get_graph_pos_shift(node_info, graph_context, batch_data)
            direct_cell = direct_cell + self.graph_cell_layer(graph_context).view(-1, 3, 3)

        batch_data["direct_pos_p"].append(direct_pos)
        batch_data["direct_cell_p"].append(direct_cell)

        if self.use_cycle:
            batch_data["cycle_residual_p"] = [
                self.get_cycle_residual(direct_pos, direct_cell, batch_data)
            ]
            batch_data["cycle_residual_t"] = torch.zeros_like(batch_data["cycle_residual_p"][-1])

        return batch_data

    def get_init_info(self,
                      batch_data : Dict[str, torch.Tensor],
                      )->Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Build initial node and edge representations.

        Args:
            batch_data: Batched structure tensors.

        Returns:
            Initial node tensor dict and edge tensor dict.
        """
        emb = self.embedding_layer(batch_data=batch_data)
        node_info = {0: emb}
        _, dij, _ = find_distances(batch_data)
        rbf = self.radial_fn(dij)
        edge_info = {0: rbf}
        return node_info, edge_info

    def get_graph_context(self,
                          node_info : Dict[int, torch.Tensor],
                          batch_data: Dict[str, torch.Tensor],
                          ) -> torch.Tensor:
        """
        Build graph-level conditioning vectors from learned node states and handcrafted features.

        Args:
            node_info: Final node tensor dictionary.
            batch_data: Batched structure tensors.

        Returns:
            Graph-level conditioning tensor.
        """
        node_scalar = node_info[0]
        graph_embed = _scatter_mean(node_scalar, batch_data["batch"])
        graph_feat_input = (batch_data["graph_feat"] - self.graph_feat_mean) / torch.clamp(self.graph_feat_std, min=1e-8)
        graph_feat = self.graph_feat_encoder(graph_feat_input)
        return self.graph_context_layer(torch.cat([graph_embed, graph_feat], dim=1))

    def get_graph_pos_shift(self,
                            node_info : Dict[int, torch.Tensor],
                            graph_context: torch.Tensor,
                            batch_data: Dict[str, torch.Tensor],
                            ) -> torch.Tensor:
        """
        Predict graph-conditioned residual shifts for atomic positions.

        Args:
            node_info: Final node tensor dictionary.
            graph_context: Graph-level conditioning tensor.
            batch_data: Batched structure tensors.

        Returns:
            Graph-conditioned atomic position residuals.
        """
        node_scalar = node_info[0]
        node_context = graph_context[batch_data["batch"]]
        return self.graph_pos_layer(torch.cat([node_scalar, node_context], dim=1))

    def get_cycle_residual(self,
                           direct_pos : torch.Tensor,
                           direct_cell: torch.Tensor,
                           batch_data : Dict[str, torch.Tensor],
                           ) -> torch.Tensor:
        """
        Compute predicted cycle residuals from position and cell updates.

        Args:
            direct_pos: Predicted atomic displacement tensor.
            direct_cell: Predicted cell displacement tensor.
            batch_data: Batched structure tensors.

        Returns:
            Padded cycle residual tensor with shape [batch, max_cycle, 3].
        """
        idx_i, idx_j = batch_data["idx_i"], batch_data["idx_j"]
        cell_pred = batch_data["cell_u"] + direct_cell
        cell_uinv = torch.linalg.inv(batch_data["cell_u"])
        cell_uinv_atom = cell_uinv[batch_data["batch"]]
        cell_pred_atom = cell_pred[batch_data["batch"]]
        cell_u_repeat = batch_data["cell_u"].repeat_interleave(batch_data["n_edges"], dim=0)
        cell_pred_repeat = cell_pred.repeat_interleave(batch_data["n_edges"], dim=0)
        edge_unrelax = batch_data["pos_u"][idx_j] - batch_data["pos_u"][idx_i] + torch.einsum(
            "a p, a p v -> a v", batch_data["offset"], cell_u_repeat
        )
        pos_pred_cart = batch_data["pos_u"] + direct_pos
        pos_relax_pred = torch.matmul(
            torch.matmul(pos_pred_cart.unsqueeze(1), cell_uinv_atom),
            cell_pred_atom,
        ).squeeze(1)
        edge_relax = pos_relax_pred[idx_j] - pos_relax_pred[idx_i] + torch.einsum(
            "a p, a p v -> a v", batch_data["offset"], cell_pred_repeat
        )
        edge_delta = edge_relax - edge_unrelax

        batch_size = int(batch_data["n_atoms"].shape[0])
        max_edge = int(batch_data["edge_mask"].shape[1])
        edge_delta_padded = edge_delta.new_zeros((batch_size, max_edge, edge_delta.shape[-1]))

        n_edges = batch_data["n_edges"].to(torch.long)
        batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=edge_delta.device),
            n_edges,
        )
        edge_offsets = torch.cumsum(n_edges, dim=0) - n_edges
        global_edge_ids = torch.arange(edge_delta.shape[0], device=edge_delta.device)
        edge_ids = global_edge_ids - torch.repeat_interleave(edge_offsets, n_edges)
        edge_delta_padded[batch_ids, edge_ids] = edge_delta

        cycle_residual = torch.matmul(batch_data["cycle_basis"], edge_delta_padded)
        return cycle_residual - batch_data["cycle_offset"] @ direct_cell

    def get_eq_blocks(self, activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers, adddim_way):
        """
        Build stacked equivariant message-passing blocks.

        Args:
            activate_fn: Activation function name.
            max_r_way: Maximum radial tensor order for each layer.
            max_in_way: Maximum input tensor order for each layer.
            max_out_way: Maximum output tensor order for each layer.
            hidden_nodes: Hidden channel sizes.
            norm_factor: Neighbor normalization factor.
            conv_mode: Graph convolution mode.
            update_edge: Whether to update edge features.
            n_layers: Number of message-passing layers.
            adddim_way: Extra input dimensions by tensor order.

        Returns:
            ModuleList of Miao blocks.
        """
        return nn.ModuleList([
            MiaoBlock(activate_fn=activate_fn,
                      radial_fn=self.radial_fn.replicate(),
                      # Use factory method, so the radial_fn in each layer are different
                      max_r_way=max_r_way[i],
                      max_in_way=max_in_way[i],
                      max_out_way=max_out_way[i],
                      input_dim=hidden_nodes[i],
                      output_dim=hidden_nodes[i + 1],
                      norm_factor=norm_factor,
                      conv_mode=conv_mode,
                      update_edge=update_edge,
                      adddim_way=adddim_way,
                      ) for i in range(n_layers)])
