import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from edge_aggr import EdgeAggregation

# import os
# from resources.dataset.acopf_dataset import ACOPFProblem

## Main model part
class EAGNN(nn.Module):
    def __init__(self, n_gnn_layers, hidden_dim, K, dropout, concat, beta):
    
        super().__init__()
        # # Load acopf dataset
        # dir = os.getcwd() + "/datasets/acopf/case"+str(client_id)+"/"
        # data_dir = dir + "FeasiblePairs_case"+str(client_id)+"_perturb_5000_samples.mat"
        # grid_dir = dir + "pglib_opf_case"+str(client_id)+".mat"
        # acopf_dataset = ACOPFProblem(data_filename=data_dir, grid_filename=grid_dir)
        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()

        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(2, 4, hidden_dim, hidden_dim))
            self.layers.append(TransformerConv(hidden_dim*3, hidden_dim, heads=K, dropout=dropout, concat=concat, beta=beta))
        else:
            self.layers.append(EdgeAggregation(2, 4, hidden_dim, hidden_dim))
            self.layers.append(TransformerConv(hidden_dim*3, hidden_dim, heads=K, dropout=dropout, concat=concat, beta=beta))

        for l in range(n_gnn_layers-1):
            self.layers.append(EdgeAggregation(hidden_dim, 4, hidden_dim, hidden_dim))
            self.layers.append(TransformerConv(hidden_dim*3, hidden_dim, heads=K, dropout=dropout, concat=concat, beta=beta))

        self.dropout = nn.Dropout(dropout, inplace=False)


    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        if edge_index.shape[1] == 0:
            # no edge at all, only single nodes. automatically undirected
            return False
        # next line: if there is the reverse of the first edge does not exist, then directed. 
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr

    def forward(self, data, n_means, n_stds, e_means, e_stds):
    # def forward(self, data, e_means, e_stds):
        # x, edge_index, edge_features = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_features, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # x, edge_index, edge_features, cluster, batch = data.x, data.edge_index, data.edge_attr, data.cluster, data.batch

        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        x_stand = (x - n_means)/n_stds # z-score normalization
        x_stand[:,n_stds.squeeze(0)==0] = 0
        # x_stand = x
        edge_stand = (edge_features - e_means)/e_stds # z-score normalization
        edge_stand[:,e_stds.squeeze(0)==0] = 0

        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x_stand = self.layers[i](x=x_stand, edge_index=edge_index, edge_attr=edge_stand)
                x_stand = nn.ReLU()(x_stand)
                # x_stand = nn.ELU()(x_stand)
            else:
                x_stand = self.layers[i](x=x_stand, edge_index=edge_index)
                # x_stand = self.graphnorm(x_stand)
                x_stand = nn.ReLU()(x_stand)
                # x_stand = nn.ELU()(x_stand)

            x_stand = self.dropout(x_stand)
        
        return x_stand