import torch
import torch.nn as nn
from transformers import AutoModel
from model.utils import (
    BatchIndices,
    FeatureDropout,
    MultiHeadAttention,
    PartitionedPositionwiseFeedForward,
)


class OneHotEmbedding(nn.Module):
    def __init__(self, num_words, embedding_dim):
        super(OneHotEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_words, embedding_dim)

    def forward(self, token_idx, valid_token_mask):
        return self.embedding(token_idx[valid_token_mask])


class TransformerEmbedding(nn.Module):
    def __init__(self, encoder, embedding_dim):
        super(TransformerEmbedding, self).__init__()
        self.contextual_embedding = AutoModel.from_pretrained(encoder)
        self.linear = nn.Linear(
            1024 if "-large" in encoder else 768, embedding_dim, bias=False
        )

    def forward(self, token_idx, valid_token_mask, wordpiece_mask):
        bert_output = self.contextual_embedding(
            token_idx, attention_mask=valid_token_mask.to(dtype=torch.float32)
        )[0]
        return self.linear(bert_output[wordpiece_mask])


class Encoder(nn.Module):
    def __init__(self, position_table, vocab, config):
        super(Encoder, self).__init__()
        self.use_tags = config.use_tags  # POS tags
        self.use_words = config.use_words  # one-hot encoding
        d_model = config.d_model
        d_content = d_model // 2
        d_position = d_model - d_content

        self.word_embedding = TransformerEmbedding(config.encoder, d_content)
        self.word_dropout = FeatureDropout(config.word_emb_dropout)

        if self.use_tags:
            self.tag_embedding = OneHotEmbedding(len(vocab["tag"]), d_content)
            self.tag_dropout = FeatureDropout(config.tag_emb_dropout)
        if self.use_words:
            self.word_onehot_embedding = OneHotEmbedding(len(vocab["token"]), d_content)

        self.position_table = position_table
        self.layer_norm = nn.LayerNorm([d_model])
        self.attention_layers = []
        for i in range(config.num_attn_layers):
            attn = MultiHeadAttention(
                config.num_attn_heads,
                d_model,
                config.d_kqv,
                config.d_kqv,
                config.residual_dropout,
                config.attention_dropout,
                d_position,
            )
            ffn = PartitionedPositionwiseFeedForward(
                d_model,
                config.d_ff,
                d_position,
                config.relu_dropout,
                config.residual_dropout,
            )
            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ffn_{i}", ffn)
            self.attention_layers.append((attn, ffn))

    def forward(self, token_idx, tag_idx, valid_token_mask, wordpiece_mask):
        word_embedding = self.word_embedding(
            token_idx, valid_token_mask, wordpiece_mask
        )
        if self.use_words:
            word_embedding += self.word_onehot_embedding(token_idx, valid_token_mask)
        lens = wordpiece_mask.detach().sum(dim=-1).tolist()
        max_len = max(lens)
        batch_idxs = BatchIndices.from_lens(lens)
        word_embedding = self.word_dropout(word_embedding, batch_idxs)
        if self.use_tags:
            valid_tags_mask = wordpiece_mask.new_zeros(
                (len(lens), max_len), dtype=torch.bool
            )
            for i, l in enumerate(lens):
                valid_tags_mask[i, :l] = True
            tag_idx = tag_idx[:, :max_len]
            tag_embedding = self.tag_dropout(
                self.tag_embedding(tag_idx, valid_tags_mask), batch_idxs
            )
            word_embedding += tag_embedding

        timing_signal = torch.cat([self.position_table[:l] for l in lens], dim=0)
        token_embedding = torch.cat([word_embedding, timing_signal], dim=1)
        token_embedding = self.layer_norm(token_embedding)

        for attn, ffn in self.attention_layers:
            token_embedding, _ = attn(token_embedding, batch_idxs)
            token_embedding = ffn(token_embedding, batch_idxs)

        token_embedding_padded = batch_idxs.inflate(token_embedding)
        return token_embedding_padded
