import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
import torch.nn.functional as F
from prompt_graph.model import GAT, GCN, GraphSAGE, GIN, GCov, GraphTransformer


def jaccard_similarity(embeddings):
    intersection = torch.mm(embeddings, embeddings.T)  # 交集
    union = embeddings.sum(dim=1, keepdim=True) + embeddings.sum(dim=1, keepdim=True).T - intersection  # 并集
    return intersection / (union + 1e-10)  # 避免除零

def cosine_similarity(embeddings):
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # L2 归一化
    return torch.mm(norm_emb, norm_emb.T)  # 计算余弦相似度

class Server:
    def __init__(self, priv_data, eps=None, delta=None):
        """
        Args:
            priv_data: 输入图数据，包含 x（特征），edge_index（边列表），y（标签）等。
            eps: 隐私参数 epsilon。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.priv_data = priv_data.to(self.device)
        self.n = priv_data.num_nodes
        self.eps = eps
        self.delta = delta
        self.priv_adj = torch.zeros((self.n, self.n), dtype=torch.float32, device=self.device)
        self.priv_adj[priv_data.edge_index[0], priv_data.edge_index[1]] = 1



    def estimate_prior(self, gnn_type, pre_train_model_path, hid_dim, num_layer, epochs, lr, reg_weight, gnn_weight):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_dim = self.priv_data.x.shape[1]
        hid_dim = hid_dim

        if gnn_type == 'GAT':
            gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
        elif gnn_type == 'GCN':
            gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
        elif gnn_type == 'GraphSAGE':
            gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
        elif gnn_type == 'GIN':
            gnn = GIN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
        elif gnn_type == 'GCov':
            gnn = GCov(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
        elif gnn_type == 'GraphTransformer':
            gnn = GraphTransformer(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        gnn.load_state_dict(torch.load(pre_train_model_path, map_location='cpu'))
        gnn.to(device)

        # 初始化可训练的 beta 参数
        beta = torch.nn.Parameter(torch.randn(self.n, 1, device=device) * 0.01)


        # 定义优化器
        optimizer = torch.optim.Adam([beta], lr=lr)

        # 预计算一些常量
        ones_1xn = torch.ones(1, self.n).to(device)
        ones_nx1 = torch.ones(self.n, 1).to(device)

        with torch.no_grad():
            node_embeddings = gnn(self.priv_data.x, self.priv_data.edge_index)
            gnn_similarity = cosine_similarity(node_embeddings)  # **计算节点相似度**
            #gnn_similarity = jaccard_similarity(node_embeddings)
            gnn_prior = torch.sigmoid(gnn_similarity)  # **转换成概率**
            gnn_prior.fill_diagonal_(0)  # **去掉自环**


        best_loss = float("inf")
        best_beta = None
        patience_counter = 0
        patience = 20

        for i in range(epochs):
            optimizer.zero_grad()

            # 计算边概率矩阵
            temperature = 0.05  # 可以调整
            s = (ones_nx1.matmul(beta.T) + beta.matmul(ones_1xn)) / temperature # β_i + β_j
            prob_matrix = torch.exp(s) / (1 + torch.exp(s))  # p_ij = σ(β_i + β_j)
            prob_matrix.fill_diagonal_(0)  # 对角线设置为0（无自环）
            prob_matrix[prob_matrix <= 1e-6] = 1e-6

            # MLE对数似然损失：我们计算当前的概率矩阵与GNN先验之间的差距
            # 使用 GNN 先验概率矩阵作为目标
            epsilon = 1e-10

            log_likelihood = torch.sum(
                gnn_prior * torch.log(prob_matrix + epsilon) + (1 - gnn_prior) * torch.log(
                    1 - prob_matrix + epsilon))

            # 正则化项
            reg = reg_weight * (beta ** 2).sum()

            # GNN输出的正则化项
            gnn_loss = gnn_weight * torch.norm(prob_matrix - gnn_prior, p=2) ** 2

            # 总损失：负对数似然 + 正则化 + GNN正则化
            loss = -log_likelihood + reg + gnn_loss

            # 梯度更新
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_beta = beta.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1

            # 打印损失值来监控优化过程
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

            if patience_counter >= patience:
                beta = best_beta  # 使用最佳 beta
                break

        # 计算最终的边概率矩阵
        s = ones_nx1.matmul(beta.T) + beta.matmul(ones_1xn)
        prior = torch.exp(s) / (1 + torch.exp(s))
        prior.fill_diagonal_(0)
        return prior

    def estimate_posterior(self, prior):
        """根据先验和私有化的邻接矩阵估计后验概率。
        Args:
            prior: 先验概率矩阵。
        Returns:
            posterior: 后验概率矩阵。
        """
        p =  1 / (1.0 + np.exp(self.eps * self.delta))
        priv_adj_t = self.priv_adj.T
        x = self.priv_adj + priv_adj_t
        pr_y_edge = 0.5 * (x - 1) * (x - 2) * p * p + 0.5 * x * (x - 1) * (1 - p) * (1 - p) - x * (x - 2) * p * (1 - p)
        pr_y_no_edge = 0.5 * (x - 1) * (x - 2) * (1 - p) * (1 - p) + 0.5 * x * (x - 1) * p * p - x * (x - 2) * p * (
                    1 - p)
        posterior = pr_y_edge * prior / (pr_y_edge * prior + pr_y_no_edge * (1 - prior))
        posterior = (posterior - posterior.min()) / (posterior.max() - posterior.min())
        return posterior

    def reconstruct_graph(self, posterior, threshold=0.9, top_k=None):

        device = posterior.device
        original_edges = self.priv_data.edge_index.shape[1]

        if top_k is None:
            # 直接使用阈值选择边
            posterior.fill_diagonal_(0)
            edge_index = (posterior > threshold).nonzero(as_tuple=False).T
        else:
            # 选取 top_k 个最可能的边

            max_edges = int(original_edges * 1.5)
            top_k = min(top_k, max_edges)
            posterior_flat = posterior.flatten()
            num_elements = posterior_flat.numel()  # 获取元素个数
            if num_elements == 0:
                raise ValueError("posterior_flat is empty, check input data.")

            # 限制 top_k 不超过 num_elements
            top_k = min(top_k, num_elements)

            # 检查是否有 NaN
            if torch.isnan(posterior_flat).any():
                raise ValueError("posterior_flat contains NaN values, check the computation.")

            top_k_indices = torch.topk(posterior_flat, top_k, sorted=False).indices  # 获取 top-k 索引

            # 计算对应的 (i, j) 位置
            row_indices = top_k_indices // posterior.shape[1]
            col_indices = top_k_indices % posterior.shape[1]
            edge_index = torch.stack([row_indices, col_indices], dim=0)


        degree = torch.bincount(edge_index.flatten(), minlength=self.priv_data.num_nodes)
        isolated_nodes = torch.where(degree == 0)[0]  # 找到孤立的节点

        if len(isolated_nodes) > 0:
            print(f"🚨 Found {len(isolated_nodes)} isolated nodes. Adding edges to prevent isolation.")

            # **给每个孤立节点随机连接一个非孤立节点**
            non_isolated_nodes = torch.where(degree > 0)[0]
            for node in isolated_nodes:
                closest_node = non_isolated_nodes[torch.randint(0, len(non_isolated_nodes), (1,))]
                edge_index = torch.cat(
                    [edge_index, torch.tensor([[node, closest_node], [closest_node, node]], device=device)], dim=1)


        # 移除自环
        edge_index, _ = remove_self_loops(edge_index)

        # 如果是无向图，去重
        edge_index = to_undirected(edge_index)



        # 创建新的图数据
        reconstruct_graph = Data(
            x=self.priv_data.x,  # 继承原始特征
            edge_index=edge_index.to(device),
            #edge_attr=edge_weights.to(device),
            y=self.priv_data.y
        )

        reconstruct_graph.num_nodes = self.priv_data.num_nodes
        reconstruct_graph.num_classes = len(torch.unique(self.priv_data.y))

        return reconstruct_graph