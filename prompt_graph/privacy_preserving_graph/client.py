import random

import torch
import numpy as np
import math
from typing import Tuple

from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from torch_geometric.data import data, Data

def set_seed(seed=42):
    """ 固定所有随机种子，确保可复现 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class GraphLDP:
    def __init__(self, eps, delta, data) -> None:
        set_seed(42)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.data = data.to(device)  # 图数据（如邻接矩阵、节点特征）传到设备
        self.n = data.num_nodes  # 图的节点数
        self.eps_a = eps * delta  # 邻接矩阵的隐私预算
        self.eps_b = eps * (1-delta)

    def AddLDP(self) -> data:
        """
        向图数据添加局部差分隐私噪声
        :return: 添加噪声后的邻接矩阵、度数和节点特征
        """
        set_seed(42)
        device = self.device
        n = self.n

        # 获取邻接矩阵
        adj = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1], sparse_sizes=(n, n)).to(
            device).to_dense()

        # 节点特征（假设节点特征为 x）
        x = self.data.x.to(device)

        def flip_prob_from_epsilon(epsilon: float) -> float:
            p = math.exp(epsilon) / (1.0 + math.exp(epsilon))
            flip_prob = 1.0 - p
            return flip_prob

        def rr_adj_sparse(
                          flip_prob: float,
                          sample_non_neighbors: int = 20,
                          seed: int = 42) -> torch.Tensor:
            set_seed(seed)

            # 取原始无向边并放到 CPU 上构建邻接列表
            edge_index_undirected = to_undirected(self.data.edge_index, num_nodes=n).cpu()

            # adjacency[i] = set_of_neighbors
            adjacency = [set() for _ in range(n)]
            for i, j in edge_index_undirected.t().tolist():
                adjacency[i].add(j)
                adjacency[j].add(i)

            flipped_edges = set()  # 存储(i,j)形式, i<j避免重复

            for i in range(n):
                neighbors_i = list(adjacency[i])

                # 1) 邻居(1->0)翻转
                for j in neighbors_i:
                    if j > i:
                        # 以 flip_prob 概率删除
                        if random.random() >= flip_prob:
                            # 没翻转 => 保留
                            flipped_edges.add((i, j))
                    elif j < i:
                        # 已经在(j,i)处理过了, 无需重复
                        pass

                # 2) 非邻居(0->1)翻转：随机抽样
                #   为防止遍历所有非邻居(可能很大), 只抽 sample_non_neighbors 个
                possible_non_neighbors = []
                tries = 0
                while len(possible_non_neighbors) < sample_non_neighbors and tries < sample_non_neighbors * 10:
                    cand = random.randint(0, n - 1)
                    tries += 1
                    if cand != i and (cand not in adjacency[i]):
                        possible_non_neighbors.append(cand)

                for k in possible_non_neighbors:
                    if random.random() < flip_prob:
                        # 翻转0->1 => 新增边
                        if i < k:
                            flipped_edges.add((i, k))
                        else:
                            flipped_edges.add((k, i))

            # 合并结果并 to_undirected, 保证无向
            new_edge_index = torch.tensor(list(flipped_edges), dtype=torch.long).t()
            new_edge_index = to_undirected(new_edge_index, num_nodes=n)
            return new_edge_index

        def rr_adj() -> torch.Tensor:
            set_seed(42)
            p = 1 / (1.0 + math.exp(self.eps_a))
            noisy_adj = ((adj + torch.bernoulli(torch.full((n, n), p)).to(device)) % 2).float()
            noisy_adj.fill_diagonal_(0)
            return noisy_adj

        def add_noise_to_features() -> torch.Tensor:
            set_seed(42)
            scale = 1.0 / self.eps_b
            noise = torch.distributions.laplace.Laplace(loc=0, scale=scale).sample(x.shape).to(device)
            noisy_x = noise + x
            #noisy_x = (noisy_x - noisy_x.mean()) / (noisy_x.std() + 1e-8)
            return noisy_x

        # 添加噪声
        noisy_adj = rr_adj()
        # 添加噪声到节点特征
        noisy_x = add_noise_to_features()

        edge_index = noisy_adj.nonzero(as_tuple=False).t()


        private_graph = Data(
            x=noisy_x,
            edge_index=edge_index.to(torch.long),
            y=self.data.y
        )

        private_graph.num_nodes = self.data.num_nodes
        private_graph.num_classes = len(torch.unique(self.data.y))

        return private_graph

