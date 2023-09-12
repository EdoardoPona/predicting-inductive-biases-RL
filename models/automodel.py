from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
)
from .base import BaseClassifier


class AutoModelClassifier(BaseClassifier):
	''' generic wrapper for any model that 
	can be loaded with AutoModelForSequenceClassification '''

	def __init__(self, model, num_steps, num_classes=2):
		super(AutoModelClassifier, self).__init__(num_classes)
		self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)   # lightning cries otherwise
		print('tokenizer', type(self.tokenizer))
		self.tokenizer.pad_token = self.tokenizer.eos_token
		self.encoder = AutoModelForSequenceClassification.from_pretrained(model)
		self.encoder.config.pad_token_id = self.tokenizer.eos_token_id
		self.num_steps = num_steps
