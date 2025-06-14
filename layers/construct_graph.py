import dgl
import torch.nn.functional as F
import torch as th
import numpy as np


th.backends.cudnn.benchmark = True


class GraphConstructionWithMasking:
    def __init__(self, args):
        self.mask_ratio = args.mask_ratio
        self.drop_ratio = args.drop_ratio
        self.device = args.device

    def construct_graph(self, id, m_embed, d_embed):

        src = id[:, 0]
        dst = id[:, 1]

        num_nodes = m_embed.size(0) + d_embed.size(0)

        graph = dgl.graph((src, dst), num_nodes=num_nodes)

        graph = graph.to(self.device)

        node_features = th.cat((m_embed, d_embed), dim=0)

        node_features = node_features.to(self.device)

        graph.ndata['feature'] = node_features

        return graph, node_features

    def mask_features(self, graph, node_features):

        num_nodes, feature_dim = node_features.size()

        mask = th.rand(num_nodes, feature_dim, device=self.device) < self.mask_ratio
        masked_features = node_features.clone()
        masked_features[mask] = 0
        graph.ndata['mask'] = mask

        return graph, masked_features, mask

    def drop_graph(self, graph, node_features):

        num_nodes = graph.number_of_nodes()
        num_drop_nodes = int(num_nodes * self.drop_ratio)

        drop_nodes = th.randperm(num_nodes, device=self.device)[:num_drop_nodes]

        drop_edges = graph.edges(etype=None)
        drop_src, drop_dst = drop_edges

        drop_edge_mask = th.isin(drop_src, drop_nodes) | th.isin(drop_dst, drop_nodes)
        drop_edge_ids = th.nonzero(drop_edge_mask).squeeze()

        graph = dgl.remove_edges(graph, drop_edge_ids)

        graph = dgl.remove_nodes(graph, drop_nodes)


        remaining_mask = th.ones(num_nodes, dtype=bool, device=self.device)
        remaining_mask[drop_nodes] = False
        features_with_dropped = node_features[remaining_mask]

        remaining_nodes = []
        original_edges = th.stack(graph.edges(), dim=1).cpu().numpy()
        for edge in original_edges:
            src, dst = edge
            if src not in drop_nodes and dst not in drop_nodes:
                remaining_nodes.append(edge)
        remaining_nodess = th.stack([th.tensor(x, device=self.device) for x in remaining_nodes])


        return graph, remaining_nodess, features_with_dropped

    def construct_masked_graph(self, id, m_embed, d_embed):


        original_graph, original_features = self.construct_graph(id, m_embed, d_embed)

        masked_graph, masked_features, mask = self.mask_features(original_graph, original_features)
        masked_graph.ndata['feature'] = masked_features
        dropped_graph, remaining_nodes, features_with_dropped = self.drop_graph(original_graph, original_features)

        return masked_graph, dropped_graph, masked_features, remaining_nodes, original_features, original_graph, features_with_dropped
