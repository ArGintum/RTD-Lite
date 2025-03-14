import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from sklearn.metrics import pairwise_distances


### Disjoint set union structure to maintain cluster structure of a graph
class DSU:
    def __init__(self, n_vertices):
        self.parent = np.arange(n_vertices)
        self.rank = np.zeros(n_vertices)

    def find(self, v):
        if self.parent[v] == v:
            return v
        self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def unite(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)
        if self.rank[u_root] < self.rank[v_root]:
            u_root, v_root = v_root, u_root
        if self.rank[u_root] == self.rank[v_root]:
            self.rank[u_root] += 1
        self.parent[v_root] = u_root
        
### Prim's minimal spanning tree algorithm

def prim_algo(adjacency_matrix):
    n = len(adjacency_matrix)
    
    infty = torch.max(adjacency_matrix).item() + 10
    dst = torch.ones(n, device=adjacency_matrix.device) * infty
    ancestors = -torch.ones(n, dtype=int, device=adjacency_matrix.device)
    visited = torch.zeros(n, dtype=bool, device=adjacency_matrix.device)
    
    mst_edges = np.zeros((n - 1, 2), dtype=np.int32)
    s, v = torch.tensor(0.0, device=adjacency_matrix.device), 0
    for i in range(n - 1):
        visited[v] = 1
        
        ancestors[dst > adjacency_matrix[v]] = v
        dst = torch.minimum(dst, adjacency_matrix[v])
        dst[visited] = infty
        v = torch.argmin(dst)

        s += adjacency_matrix[v][ancestors[v]]
        
        mst_edges[i][0] = v
        mst_edges[i][1] = ancestors[v]
                
    edge_weights = adjacency_matrix[mst_edges[:, 0], mst_edges[:, 1] ].cpu()
    return s, mst_edges, edge_weights

### As above, so below.
## Prim's algorithm only for total weight (without returning actual edges)

def prim_algo_simplified(adjacency_matrix):
    n = len(adjacency_matrix)
    
    infty = torch.max(adjacency_matrix).item() + 10
    dst = torch.ones(n, device=adjacency_matrix.device) * infty
    ancestors = -torch.ones(n, dtype=int, device=adjacency_matrix.device)
    visited = torch.zeros(n, dtype=bool, device=adjacency_matrix.device)
    
    s, v = torch.tensor(0.0, device=adjacency_matrix.device), 0
    for i in range(n - 1):
        visited[v] = 1
        
        ancestors[dst > adjacency_matrix[v]] = v
        dst = torch.minimum(dst, adjacency_matrix[v])
        dst[visited] = infty
        v = torch.argmin(dst)
       
        s += adjacency_matrix[v, ancestors[v]]

    return s


### Main part
class RTD_Lite:
    def __init__(self, r1, r2, quant_outer=None, quant_inner=None, distance='euclidean'):
        dists_1 = torch.cdist(r1, r1)
        if quant_outer is None:
            quant_outer = torch.quantile(dists_1, 0.9)
        self.r1 = dists_1 / quant_outer
        
        dists_2 = torch.cdist(r2, r2)
        if quant_inner is None:
            quant_inner = torch.quantile(dists_2, 0.9)
        self.r2 = dists_2 / quant_inner
        self.device = r1.device

        
    def __call__(self):
        rmin = torch.minimum(self.r1, self.r2)              
        
        rmin_sum, rmin_edge_idx, rmin_edge_w = prim_algo(rmin.cpu())
        r1_sum, r1_edge_idx, r1_edge_w = prim_algo(self.r1.cpu())
        r2_sum, r2_edge_idx, r2_edge_w = prim_algo(self.r2.cpu())

        rmin_edge_idx = rmin_edge_idx[rmin_edge_w.argsort()]
        rmin_edge_w = rmin_edge_w[rmin_edge_w.argsort()]
               
        r1_edge_idx = r1_edge_idx[r1_edge_w.argsort()]
        r1_edge_w = r1_edge_w[r1_edge_w.argsort()]
        
        r2_edge_idx = r2_edge_idx[r2_edge_w.argsort()]
        r2_edge_w = r2_edge_w[r2_edge_w.argsort()]
        
        min_graph_dsu = DSU(self.r1.shape[0])       
        barcodes = {'1->2' : [], '2->1' : []}
        for i in range(len(rmin_edge_idx)):
            u_clique = min_graph_dsu.find(rmin_edge_idx[i][0])
            v_clique = min_graph_dsu.find(rmin_edge_idx[i][1])
            birth = rmin_edge_w[i]
            
            r1_graph_dsu = deepcopy(min_graph_dsu)
            for j in range(len(r1_edge_idx)):
                r1_graph_dsu.unite(r1_edge_idx[j][0], r1_edge_idx[j][1])    
                if r1_graph_dsu.find(u_clique) == r1_graph_dsu.find(v_clique):
                    death_1 = r1_edge_w[j]
                    break
            
            r2_graph_dsu = deepcopy(min_graph_dsu)
            for j in range(len(r2_edge_idx)):
                r2_graph_dsu.unite(r2_edge_idx[j][0], r2_edge_idx[j][1])
                
                if r2_graph_dsu.find(u_clique) == r2_graph_dsu.find(v_clique):
                    death_2 = r2_edge_w[j]
                    break

            if death_1 > birth:
                barcodes['1->2'].append(torch.stack((birth, death_1)))
            if death_2 > birth:
                barcodes['2->1'].append(torch.stack((birth, death_2)))
            min_graph_dsu.unite(rmin_edge_idx[i][0], rmin_edge_idx[i][1])
       
        if len(barcodes['1->2']) > 0:
            barcodes['1->2'] = torch.stack(barcodes['1->2']).to(self.device)
        if len(barcodes['2->1']) > 0:
            barcodes['2->1'] = torch.stack(barcodes['2->1']).to(self.device)
        return barcodes
    
class RTD_Lite_summ_only:
    def __init__(self, r1, r2, quant_outer=None, quant_inner=None, distance='euclidean'):
        dists_1 = torch.cdist(r1, r1)
        if quant_outer is None:
            quant_outer = torch.quantile(dists_1, 0.9)
        self.r1 = dists_1 / quant_outer
        
        dists_2 = torch.cdist(r2, r2)
        if quant_inner is None:
            quant_inner = torch.quantile(dists_2, 0.9)
        self.r2 = dists_2 / quant_inner
        
        self.device = r1.device
        
    def __call__(self):
        rmin = torch.minimum(self.r1, self.r2)
        
        rmin_sum = prim_algo_simplified(rmin.cpu())
        r1_sum = prim_algo_simplified(self.r1.cpu())
        r2_sum = prim_algo_simplified(self.r2.cpu())

        return 0.5 * (r1_sum - rmin_sum + r2_sum - rmin_sum) 