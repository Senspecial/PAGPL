import torch


def drop_feature(x, drop_prob=0.3):
    """
    Drop feature by randomly zeroing out feature dimensions across all nodes.
    Args:
        x: [N, F] feature matrix
        drop_prob: proportion of features to drop
    Returns:
        x_dropped: feature matrix after drop
    """
    if drop_prob <= 0.0:
        return x

    drop_mask = torch.rand(x.size(1), device=x.device) > drop_prob
    x = x * drop_mask  # broadcast to each node
    return x
