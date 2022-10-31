import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter

class GenericGNN(MessagePassing):
    
    def __init__(self, node_dim, edge_dim, out_dim, msg_dim=100, hidden=300, aggr='add'):
        
        super(GenericGNN, self).__init__(aggr=aggr)  # "Add" aggregation.
 
        self.msg_fnc = Seq(
            Lin(2*node_dim+edge_dim, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim)
        )
        
        self.node_fnc = Seq(
            Lin(msg_dim+node_dim, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, out_dim)
        )
            
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        else:
            return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:            
            return self.msg_fnc(torch.cat([x_i, x_j, edge_attr], dim=1))
        else:
            return self.msg_fnc(torch.cat([x_i, x_j], dim=1))
        
    def update(self, aggr_out, x=None):
        return self.node_fnc(torch.cat([x, aggr_out], dim=1))
    
    def loss(self, actual, pred):
        return torch.sum(torch.abs(actual - pred))
    