import torch
from torch_geometric.datasets import Planetoid
from .client import GraphLDP, set_seed  # 确保 GraphLDP 已经正确导入
import os

def apply_graph_ldp(dataset_name: str, data, epsilon: float, delta: float, save: bool = False):
    set_seed(42)
    ldp = GraphLDP(eps=epsilon, delta=delta, data=data)
    private_graph = ldp.AddLDP()
    print(f"✅ 生成完成: Nodes={private_graph.num_nodes}, Edges={private_graph.num_edges}")

    # **Step 3: 选择是否保存**
    if save:
        save_path = f"./privacy_preserving_graph/data/{dataset_name}_private_eps{epsilon}_delta{delta}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(private_graph, save_path)
        print(f"📁 已保存至: {save_path}")

    return private_graph  # 返回新的 PyG 图
