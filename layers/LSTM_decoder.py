import torch.nn as nn
import dgl
import torch.nn.functional as F
import torch as th
import numpy as np

from layers.transformer_encoder import TokenEmbeddingExtractor
class LSTMDecoder(nn.Module):
    def __init__(self, args):
        super(LSTMDecoder, self).__init__()
        self.gru = nn.GRU(args.hidden_dim, args.output_dim, args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.output_dim, args.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.get_token_embedding = TokenEmbeddingExtractor(args)

    def forward(self, enc_inputs, embedding):
        enc_outputs = self.get_token_embedding(enc_inputs, embedding)
        gru_out, _ = self.gru(enc_outputs)
        outputs = self.fc(gru_out)
        outputs = self.softmax(outputs)
        return outputs


