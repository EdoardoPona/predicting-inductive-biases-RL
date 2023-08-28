# import torch
# import torch.nn as nn
# from torch.optim import Adam

# from . import head
# from .base import BaseClassifier


# class TransformerToy(BaseClassifier):

#     def __init__(
#             self, 
#             model,
#             vocab_size=50_000,
#             num_classes=2,
#             num_heads=4,
#             num_layers=4,
#             hidden_size: int = 1024
#         ):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_size)
#         self.transformer = nn.TransformerEncoder(
# 			nn.TransformerEncoderLayer(
# 			    hidden_size, 
# 		        num_heads, 
#                 4*hidden_size
# 		    ),
# 		    num_layers,
#         )
#         self.classifier = head.ClassificationHead(hidden_size, num_classes)

#     def configure_optimizers(self):
#         optimizer = Adam(self.parameters(), lr=1e-4)
#         return [optimizer]

#     def step(self, batch):
#         device = self.device
#         texts, labels = batch
#         tokens = self.tokenize(texts).to(device)
#         embeddings = self.embedding(tokens)
#         residual = self.transformer(embeddings)
#         logits = self.classifier(residual[:,-1])
#         loss = nn.functional.cross_entropy(logits, labels.to(device))
#         return logits, loss
    
#     def tokenize(self, texts):
#         return torch.nn.utils.rnn.pad_sequence(
#             [torch.tensor([int(w) for w in t.split()]) for t in texts],
#             batch_first=True,
#         )
    


# NOT USED IN RL

import torch
import torch.nn as nn
from torch.optim import Adam

from . import head
from .base import BaseClassifier

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x


class TransformerToy(BaseClassifier):
    def __init__(
        self,
        model,
        vocab_size=50_000,
        num_classes=2,
        num_heads=4,
        num_layers=2,
        hidden_size: int = 256,
    ):
        super(TransformerToy, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.classifier = head.ClassificationHead(hidden_size, num_classes)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        return [optimizer]

    # def step(self, batch):
    #     device = self.device
    #     texts, labels = batch
    #     tokens = self.tokenize(texts).to(device)
    #     embeddings = self.embedding(tokens)
    #     positional_embeddings = self.positional_encoding(embeddings)
    #     transformer_output = self.transformer(positional_embeddings)
    #     logits = self.classifier(transformer_output[:, -1, :])  # consider only the output of the last token
    #     loss = nn.functional.cross_entropy(logits, labels.to(device))
    #     return logits, loss
    
    def step(self, batch):
        device = self.device
        texts, labels = batch
        tokens = self.tokenize(texts).to(device)
        embeddings = self.embedding(tokens)
        positional_embeddings = self.positional_encoding(embeddings).permute(1, 0, 2)  # permute to (S, N, E)
        transformer_output = self.transformer(positional_embeddings)
        transformer_output = transformer_output.permute(1, 0, 2)  # permute back to (N, S, E)
        logits = self.classifier(transformer_output[:, -1, :])  # consider only the output of the last token
        loss = nn.functional.cross_entropy(logits, labels.to(device))
        return logits, loss



    def tokenize(self, texts):
        return torch.stack([torch.tensor([int(w) for w in t.split()]) for t in texts])


