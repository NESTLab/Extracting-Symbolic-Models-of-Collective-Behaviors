import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU,Tanh
from torch_scatter import scatter

class Generator(MessagePassing):
    
    def __init__(self, node_dim, edge_dim, out_dim, msg_dim=100, hidden=300, aggr='add'):
        
        super(Generator, self).__init__(aggr=aggr)  # "Add" aggregation.
 
        self.msg_fnc = Seq(
            Lin(2*node_dim+edge_dim, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            Tanh(),
            Lin(hidden, msg_dim)
        )
        
        self.node_fnc = Seq(
            Lin(msg_dim+node_dim, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            Tanh(),
            Lin(hidden, out_dim)
        )
            
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr:
            result = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        else:
            result = self.propagate(edge_index, x=x)
        return result
    
    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr:
            tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        else:
            tmp = torch.cat([x_i, x_j], dim=1)
        return self.msg_fnc(tmp)
    
    def update(self, aggr_out, x=None):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp)
    
    def loss(self, actual, pred):
        return torch.sum(torch.abs(actual - pred))
    