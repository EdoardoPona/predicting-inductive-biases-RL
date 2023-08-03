import torch
import torch.nn as nn
from torch.optim import Adam

from . import head
from .base import BaseClassifier


class TransformerToy(BaseClassifier):

    def __init__(
            self, 
            model,
            vocab_size=50_000,
            num_classes=2,
            num_heads=4,
            num_layers=4,
            hidden_size: int = 256
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
			nn.TransformerEncoderLayer(
			    hidden_size, 
		        num_heads, 
                4*hidden_size
		    ),
		    num_layers,
        )
        self.classifier = head.ClassificationHead(hidden_size, num_classes)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return [optimizer]

    def step(self, batch):
        device = self.device
        texts, labels = batch
        tokens = self.tokenize(texts).to(device)
        embeddings = self.embedding(tokens)
        residual = self.transformer(embeddings)
        logits = self.classifier(residual[:,-1])
        loss = nn.functional.cross_entropy(logits, labels.to(device))
        return logits, loss
    
    def tokenize(self, texts):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([int(w) for w in t.split()]) for t in texts],
            batch_first=True,
        )
