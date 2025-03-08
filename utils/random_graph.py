import numpy as np

def er_graph(num_nodes, num_edges):
    sources = np.random.randint(0, num_nodes, size=num_edges)
    targets = np.random.randint(0, num_nodes, size=num_edges)

    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]

    while len(sources) < num_edges:
        extra_sources = np.random.randint(0, num_nodes, size=(num_edges - len(sources)))
        extra_targets = np.random.randint(0, num_nodes, size=(num_edges - len(targets)))
        extra_mask = extra_sources != extra_targets
        sources = np.concatenate([sources, extra_sources[extra_mask]])
        targets = np.concatenate([targets, extra_targets[extra_mask]])

    return torch.from_numpy(np.array([sources[:num_edges], targets[:num_edges]]))
