import torch.nn as nn
import torch as th
import numpy as np
# from timm.layers import LayerNorm2d

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args

        # Token Embedding Extractor
        self.token_embedding_extractor = TokenEmbeddingExtractor(args)

        # Encoder Layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(args) for _ in range(int(args.enc_layers))])

    def forward(self, enc_inputs, embedding):
        # Extract token embeddings
        enc_outputs = self.token_embedding_extractor(enc_inputs, embedding)

        # Generate attention mask
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)

        # Reshape outputs
        enc_outputs = enc_outputs.view(enc_outputs.size()[0], -1)
        return enc_outputs

class TokenEmbeddingExtractor(nn.Module):
    def __init__(self, args):
        super(TokenEmbeddingExtractor, self).__init__()
        self.args = args

    def forward(self, datax: th.Tensor, embedding: th.Tensor):
        token_embedding = []
        for pair in datax:
            with th.no_grad():  # 禁用梯度计算
                embed = th.cat([embedding[pair[0]], embedding[pair[1]]], dim=0)
            token_embedding.append(embed)

        token_embed = th.stack(token_embedding, dim=0)  # Convert list to tensor
        return token_embed


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k)

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(th.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(th.ones(normalized_shape))
        self.bias = nn.Parameter(th.zeros(normalized_shape))

    def forward(self, x):
        x = th.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args

        # Low-rank matrix factorization
        self.W_Q1 = nn.Linear(args.hidden_dim, args.d_h // 2, bias=False)
        self.W_Q2 = nn.Linear(args.d_h // 2, args.d_h * args.n_heads, bias=False)

        self.W_K1 = nn.Linear(args.hidden_dim, args.d_h // 2, bias=False)
        self.W_K2 = nn.Linear(args.d_h // 2, args.d_h * args.n_heads, bias=False)

        self.W_V = nn.Linear(args.hidden_dim, args.d_h * args.n_heads, bias=False)
        self.W_Q = nn.Linear(args.hidden_dim, args.d_h * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.hidden_dim, args.d_h * args.n_heads, bias=False)

        # Final linear projection
        self.fc = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)

        # Learnable parameters for Agent Attention
        self.A = nn.Parameter(th.randn(args.d_h, args.d_h))  # Shared projection matrix


        # Learnable scaling factor
        self.layer_norm = nn.LayerNorm(args.hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        input_Q = input_Q.to(self.args.device)
        input_K = input_K.to(self.args.device)
        input_V = input_V.to(self.args.device)

        residual, batch_size = input_Q, input_Q.size(0)

        # Compute Q and K using low-rank matrix factorization
        Q = self.W_Q2(self.W_Q1(input_Q)).view(batch_size, -1, self.args.n_heads, self.args.d_h).transpose(1, 2)
        K = self.W_K2(self.W_K1(input_K)).view(batch_size, -1, self.args.n_heads, self.args.d_h).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.args.n_heads, self.args.d_h).transpose(1, 2)

        # Step 1: Compute VV using Softmax Attention (A^T K)
        Q_A = self.A.transpose(0, 1)  # Transpose of A
        scores_VV = th.matmul(Q_A.unsqueeze(0).unsqueeze(0), K.transpose(-1, -2)) / (np.sqrt(self.args.d_h))
        attn_VV = nn.Softmax(dim=-1)(scores_VV)
        VV = th.matmul(attn_VV, V)  # Compute intermediate representation VV

        # Step 2: Compute final attention context (Q A VV)
        K_A = self.A
        scores_VV2 = th.matmul(Q.unsqueeze(0).unsqueeze(0), K_A.transpose(-1, -2)) / (np.sqrt(self.args.d_h))
        attn_VV2 = nn.Softmax(dim=-1)(scores_VV2)
        VVV = th.matmul(attn_VV2, VV)  # Compute intermediate representation VV

        # Reshape and project
        VVV = VVV.transpose(1, 2).reshape(batch_size, -1, self.args.n_heads * self.args.d_h)
        outputs1 = self.layer_norm(VVV + residual)
        outputs2 = self.fc(outputs1)
        # outputs2 = outputs2 + self.dropout(outputs2)
        outputs = self.layer_norm(outputs2 + outputs1)

        return outputs, attn_VV

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.enc_self_attn = MultiHeadAttention(args)

    def forward(self, enc_inputs, enc_self_attn_mask):
        device = enc_inputs.device
        enc_self_attn_mask = enc_self_attn_mask.to(device)

        if enc_inputs.dim() == 2:  # (batch_size, 512) → (batch_size, 1, 512)
            enc_inputs = enc_inputs.unsqueeze(1)

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        return enc_outputs, attn