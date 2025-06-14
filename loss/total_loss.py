import torch.nn as nn
import dgl
import torch.nn.functional as F
import torch as th
import numpy as np
from layers.transformer_encoder import TransformerEncoder,TokenEmbeddingExtractor
from layers.LSTM_decoder import LSTMDecoder



class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(args)

        self.projector = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.output_dim)
        )


        self.generative_mapper = nn.Linear(args.hidden_dim, args.final_dim)
        self.contrastive_mapper = nn.Linear(args.output_dim, args.final_dim)
        self.masked_mapper = nn.Linear(args.input_dim, args.final_dim)
        self.dim_adjuster = nn.Linear(1024, 512)
        self.get_token_embedding = TokenEmbeddingExtractor(args)
        self.decoder = LSTMDecoder(args)


    def forward(self, id, masked_graph, dropped_graph, masked_features, remaining_nodes,
                original_features, original_graph, features_with_dropped):
        z_masked = self.encoder(id, masked_features)
        z_original = self.get_token_embedding(id, original_features)

        z_rmask = self.decoder(id, masked_features)

        z_dropped = self.encoder(remaining_nodes, features_with_dropped)

        z_rdrop = self.decoder(id, features_with_dropped)

        contrastive_features_masked = self.projector(z_masked)
        contrastive_features_dropped = self.projector(z_dropped)


        contrastive_loss_masked = self.contrastive_loss(contrastive_features_masked, contrastive_features_dropped)


        reconstruction_loss_masked = self.reconstruction_loss(z_rmask, z_original)
        reconstruction_loss_dropped = self.reconstruction_loss(z_rdrop,
                                                               z_original)

        concatenated_features = th.cat((z_rmask, z_rdrop), dim=-1)

        concatenated_features = self.dim_adjuster(concatenated_features)

        concatenated_loss = self.reconstruction_loss(concatenated_features, z_original)


        variance_discriminative_loss_masked = self.variance_discriminative_loss(z_masked, z_dropped)

        total_loss = contrastive_loss_masked +reconstruction_loss_masked + reconstruction_loss_dropped + concatenated_loss + variance_discriminative_loss_masked


        final_features = th.cat((z_masked, z_dropped,), dim=-1)

        return final_features, total_loss

    def contrastive_loss(self, z_proj1, z_proj2):


        z_proj1 = F.normalize(z_proj1, p=2, dim=1)
        z_proj2 = F.normalize(z_proj2, p=2, dim=1)


        cosine_sim = th.matmul(z_proj1, z_proj2.T)


        contrastive_loss = 1 - th.diagonal(cosine_sim)
        contrastive_loss = th.mean(contrastive_loss)

        return contrastive_loss

    def reconstruction_loss(self, reconstructed_feat, original_feat):


        return F.mse_loss(reconstructed_feat, original_feat)

    def variance_discriminative_loss(self, z, graph):

        variance_z = th.var(z, dim=0)


        variance_loss = th.sum(variance_z)
        return variance_loss