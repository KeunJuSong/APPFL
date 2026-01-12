import torch
import torch.nn as nn
from torch.types import Device
from torch_geometric.nn import MessagePassing, TAGConv, GCNConv, GATConv, ChebConv, SGConv, SSGConv, TransformerConv
from torch_geometric.nn import aggr, norm
from torch_geometric.utils import degree
import math

class EdgeAggregation(MessagePassing):
    """
    MessagePassing for aggregating edge features
    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        # super().__init__(aggr='add') # var, add, mean, std, median
        # aggrs=[aggr.PowerMeanAggregation(p=1), aggr.SoftmaxAggregation(t=1), 'var']
        super().__init__(aggr=aggr.MultiAggregation(aggrs=[aggr.PowerMeanAggregation(p=1), 'std', aggr.VariancePreservingAggregation()], # 'median', 'mean', 'std', 'var', 'min', 'max', aggrs=[aggr.SoftmaxAggregation(learn=True), aggr.PowerMeanAggregation(learn=True)],
                                                    # aggrs=[aggr.SoftmaxAggregation(t=3), aggr.PowerMeanAggregation(p=3), 'std', 'var'],
                                                    mode='cat', # ) # sum, logsumexp, proj, attn, cat
                                                    # mode_kwargs=dict(in_channels = hidden_dim, out_channels = hidden_dim, num_heads=2)))
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

class Edge_GNNSolver(nn.Module):
    def __init__(self, data, args):
    
        super().__init__()
        self._data = data # NOTE: "ACOPFProblem" class type
        self._args = args

        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if self._args['n_gnn_layers'] == 1:
            self.layers.append(EdgeAggregation(self._args['nfeature_dim'], self._args['efeature_dim'], self._args['hidden_dim'], self._args['hidden_dim']))
            # self.layers.append(TAGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K']))
            # self.layers.append(ChebConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K'], bias=True))
            # self.layers.append(GCNConv(self._args['hidden_dim']*3, self._args['hidden_dim'], bias=True))
            # self.layers.append(GATConv(self._args['hidden_dim']*3, self._args['hidden_dim'], heads=self._args['K'], concat=False))
            # self.layers.append(SGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K'], bias=True))
            # self.layers.append(SSGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], alpha=0.5, K=self._args['K'], bias=True))
            self.layers.append(TransformerConv(self._args['hidden_dim']*3, self._args['hidden_dim'], heads=self._args['K'], dropout=0.1, concat=False, beta=True, bias=True))
        else:
            self.layers.append(EdgeAggregation(self._args['nfeature_dim'], self._args['efeature_dim'], self._args['hidden_dim'], self._args['hidden_dim']))
            # self.layers.append(TAGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K']))
            # self.layers.append(ChebConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K'], bias=True))
            # self.layers.append(GCNConv(self._args['hidden_dim']*3, self._args['hidden_dim'], bias=True))
            # self.layers.append(GATConv(self._args['hidden_dim']*3, self._args['hidden_dim'], heads=self._args['K'], concat=False))
            # self.layers.append(SGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K'], bias=True))
            # self.layers.append(SSGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], alpha=0.5, K=self._args['K'], bias=True))
            self.layers.append(TransformerConv(self._args['hidden_dim']*3, self._args['hidden_dim'], heads=self._args['K'], dropout=0.1, concat=False, beta=True, bias=True))

        for l in range(self._args['n_gnn_layers']-1):
            self.layers.append(EdgeAggregation(self._args['hidden_dim'], self._args['efeature_dim'], self._args['hidden_dim'], self._args['hidden_dim']))
            # self.layers.append(TAGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K']))
            # self.layers.append(ChebConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K'], bias=True))
            # self.layers.append(GCNConv(self._args['hidden_dim']*3, self._args['hidden_dim'], bias=True))
            # self.layers.append(GATConv(self._args['hidden_dim']*3, self._args['hidden_dim'], heads=self._args['K'], concat=False))
            # self.layers.append(SGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], K=self._args['K'], bias=True))
            # self.layers.append(SSGConv(self._args['hidden_dim']*3, self._args['hidden_dim'], alpha=0.5, K=self._args['K'], bias=True))
            self.layers.append(TransformerConv(self._args['hidden_dim']*3, self._args['hidden_dim'], heads=self._args['K'], dropout=0.1, concat=False, beta=True, bias=True))

        output_dim = data.ydim - data.nknowns # slack bus의 "Va"은 이미 알기 때문에 제외.. 
        if self._args['useCompl']:        
            # self.flatten = nn.Linear((data.npv+data.nslack)*self._args['hidden_dim'], output_dim - data.neq, bias=True)
            self.flatten = nn.Linear((data.npv+data.nslack)*self._args['hidden_dim']*3, output_dim - data.neq, bias=True)
            # self.flatten = nn.Linear((data.npv+data.nslack)*self._args['hidden_dim']+self._args['hidden_dim'], output_dim - data.neq, bias=True)
        else:
            self.flatten = nn.Linear((data.npv+data.nslack)*self._args['hidden_dim'], output_dim, bias=True)
            # self.flatten = nn.Linear((data.npv+data.nslack)*self._args['hidden_dim']+self._args['hidden_dim'], output_dim, bias=True)

        # self.graphnorm = norm.GraphNorm(self._args['hidden_dim'])        
        self.dropout = nn.Dropout(self._args['dropout_rate'], inplace=False)

        # ## NOTE: For running Fed-GraphOPF, do not set the weight initialization as zeros!!
        # for layer in self.layers:
        #     if type(layer) == EdgeAggregation:
        #         # nn.init.sparse_(layer.edge_aggr[0].weight, sparsity=0.5)
        #         # nn.init.sparse_(layer.edge_aggr[2].weight, sparsity=0.5)
        #         # nn.init.uniform_(layer.edge_aggr[0].weight, a=-(11/math.sqrt(layer.edge_aggr[0].weight.shape[0]*layer.edge_aggr[0].weight.shape[1])), b=(11/math.sqrt(layer.edge_aggr[0].weight.shape[0]*layer.edge_aggr[0].weight.shape[1])))
        #         # nn.init.uniform_(layer.edge_aggr[2].weight, a=-(11/math.sqrt(layer.edge_aggr[2].weight.shape[0]*layer.edge_aggr[2].weight.shape[1])), b=(11/math.sqrt(layer.edge_aggr[2].weight.shape[0]*layer.edge_aggr[2].weight.shape[1])))
        #         nn.init.uniform_(layer.edge_aggr[0].weight, a=-0.25, b=0.25)
        #         nn.init.uniform_(layer.edge_aggr[2].weight, a=-0.25, b=0.25)
        #     if type(layer) == ChebConv: # TAGConv, ChebConv, GCNConv, GATConv
        #         for k in range(self._args['K']):
        #             # nn.init.sparse_(layer.lins[k].weight, sparsity=0.5)
        #             # nn.init.uniform_(layer.lins[k].weight, a=-(11/math.sqrt(layer.lins[k].weight.shape[0]*layer.lins[k].weight.shape[1])), b=(11/math.sqrt(layer.lins[k].weight.shape[0]*layer.lins[k].weight.shape[1])))
        #             nn.init.uniform_(layer.lins[k].weight, a=-0.25, b=0.25)
        #         # nn.init.uniform_(layer.lin.weight, a=-(1/(layer.lin.weight.shape[1])), b=(1/(layer.lin.weight.shape[1])))
        #         # nn.init.xavier_uniform_(layer.lin.weight)
        #         # nn.init.sparse_(layer.lin.weight, sparsity=0.3)
        #     # if type(layer) == nn.Linear:
        #     #     nn.init.sparse_(layer.weight, sparsity=0.9)

        # # nn.init.kaiming_normal_(self.flatten.weight, nonlinearity='relu') # leaky_relu
        # # nn.init.kaiming_uniform_(self.flatten.weight, nonlinearity='relu')
        # # nn.init.xavier_normal_(self.flatten.weight)
        # # nn.init.xavier_uniform_(self.flatten.weight)
        # nn.init.zeros_(self.flatten.weight)

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
        ## We need to reshape the input data x for implicit layer.
        X = torch.zeros((int(data.x.shape[0]/self._data.nbus), self._data.nbus*2)).to(self._data.device)
        for idx in range(X.shape[0]):
            X[idx,:self._data.nbus] = data.x[self._data.nbus*idx:self._data.nbus*(idx+1),0] # Pd
            X[idx,self._data.nbus:] = data.x[self._data.nbus*idx:self._data.nbus*(idx+1),1] # Qd
        
        # # data.x의 크기를 맞추어 reshape한 뒤, X에 바로 할당합니다.
        # reshaped_data = data.x.view(-1, self._data.nbus, 2)  # [num_samples, nbus, 2]
        # # 필요한 부분만 슬라이싱하여 X에 할당
        # X[:, :self._data.nbus] = reshaped_data[:, :, 0]  # Pd
        # X[:, self._data.nbus:] = reshaped_data[:, :, 1]  # Qd
        
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

        # x_stand_pool = max_pool_x(cluster=cluster,x=x_stand,batch=batch)[0]
        # x_stand_pool = global_mean_pool(x=x_stand,batch=batch)

        # take the spv bus only! 
        x_stand = x_stand[data.node_mask,:]

        # change the dim from graph data to normal data! (batch_size, nbus*hiddenSize)
        # x_reshape = x_stand.reshape(-1, (self._data.npv+self._data.nslack)*self._args['hidden_dim'])
        x_reshape = x_stand.reshape(-1, (self._data.npv+self._data.nslack)*self._args['hidden_dim']*3)

        # x_reshape = torch.cat((x_reshape,x_stand_pool),dim=1)

        out = self.flatten(x_reshape) # (batch_size, nspv*hiddenSize) --> (batch_size, output_dim-data.neq)

        if self._args['useCompl']:
            out = nn.Sigmoid()(out)   # used to interpolate between max and min values
            return self._data.complete_partial(X, out) ## NOTE: utils 파일의 PFFunction 함수를 call하는 부분!! i.e., Solving PF using implicit layer??
        else:
            return self._data.process_output(X, out) ## TODO: Check if this is for LDF...