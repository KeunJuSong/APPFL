from torch_geometric.nn import MessagePassing
from torch_geometric.nn import aggr
from torch_geometric.utils import degree
import torch.nn as nn
import torch


## GNN for edge features
class EdgeAggregation(MessagePassing):
    """
    MessagePassing for aggregating edge features
    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr=aggr.MultiAggregation(aggrs=[aggr.PowerMeanAggregation(p=1), 'std', aggr.VariancePreservingAggregation()], 
                                                    mode='cat',
                                                    mode_kwargs=dict(in_channels = hidden_dim, out_channels = hidden_dim)))

        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim

        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim*2 + efeature_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False)            
        )
    
    # def reset_parameters(self):
    #   self.edge_aggr.reset_parameters()

    def message(self, x_i, x_j, edge_attr, norm):
        """
        x_j:        shape (N, nfeature_dim,)
        edge_attr:  shape (N, efeature_dim,)
        """
        # return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1)) # PNAConv style... (or this can be seen as CANOS??)
        return self.edge_aggr(torch.cat([norm.view(-1, 1)*x_i, norm.view(-1, 1)*x_j, norm.view(-1, 1)*edge_attr], dim=-1)) #
    
    def forward(self, x, edge_index, edge_attr):
        '''
        input:
            x:          shape (N, num_nodes, nfeature_dim,)
            edge_attr:  shape (N, num_edges, efeature_dim,)
            
        output:
            out:        shape (N, num_nodes, output_dim,)
        '''
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # no self loop because NO EDGE ATTR FOR SELF LOOP

        # Step 2: Calculate the degree of each node.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 

        # Step 3: Feature transformation. 
        # x = self.linear(x) # no feature transformation

        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        return out
