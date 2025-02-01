import torch
import torch.nn as nn
from model.decoder import GraphDecoder, SequenceDecoder
from model.encoder import Encoder


class Parser(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, vocabs, config):
        super().__init__()
        d_positional = config.d_model - config.d_model // 2
        self.position_table = nn.Parameter(
            torch.empty(config.max_sentence_len, d_positional)
        )

        nn.init.normal(self.position_table)
        self.encoder = Encoder(self.position_table, vocabs, config)
        if config.decoder == "graph":
            self.decoder = GraphDecoder(self.position_table, vocabs, config)
        elif config.decoder == "sequence":
            self.decoder = SequenceDecoder(vocabs, config)
        else:
            raise ValueError(f"Unknown decoder type: {config.decoder}")

    def forward(self, state, top_k=None):
        return (
            self.decoder(state, top_k)
            if isinstance(self.decoder, GraphDecoder)
            else self.decoder(state)
        )
