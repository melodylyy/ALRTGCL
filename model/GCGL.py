import torch.nn as nn
import dgl
import torch.nn.functional as F
import torch as th
import numpy as np

from loss.total_loss import Transformer
from layers.construct_graph import GraphConstructionWithMasking
from layers.transformer_encoder import TokenEmbeddingExtractor
from layers.predict import MLP

class modell(nn.Module):
    def __init__(self, args):
        super(modell, self).__init__()

        self.transformer = Transformer(args)
        self.graph_constructor = GraphConstructionWithMasking(args)
        self.get_token_embedding = TokenEmbeddingExtractor(args)
        self.mlp = MLP(args)

    def extract_edge_indices(graph):
        src, dst = graph.edges()
        edge_indices = th.stack((src, dst), dim=1)
        return edge_indices

    def forward(self, id, m_embed, d_embed):
        masked_graph, dropped_graph, masked_features, remaining_nodes, original_features, original_graph, features_with_dropped \
            = self.graph_constructor.construct_masked_graph(id, m_embed, d_embed)

        final_features, total_loss = self.transformer(id, masked_graph, dropped_graph, masked_features, remaining_nodes,
                                                      original_features, original_graph, features_with_dropped)


        prediction = self.mlp(final_features)
        return prediction, total_loss