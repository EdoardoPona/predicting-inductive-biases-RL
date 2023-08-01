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
			hidden_size: int = 300
		):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.transformer = nn.TransformerDecoder(
			hidden_size, 
			num_heads, 
			hidden_size,
			'relu'
		)
		self.classifier = head.ClassificationHead(hidden_size, num_classes)

	def configure_optimizers(self):
		optimizer = Adam(self.parameters())
		return [optimizer]

	def step(self, batch):
			



