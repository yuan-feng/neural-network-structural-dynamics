
import torch
from nn_base import BaseNN

class ECNN(torch.nn.Module):
	def __init__(self, input_dim, base_model, baseline=False):
		super(ECNN, self).__init__()
		self.baseline = baseline
		self.base_model = base_model

	def forward(self, x):
		if self.baseline:
			return self.base_model(x)
		y = self.base_model(x)
		return y.split(1,1)

	def time_derivative(self, x, t=None):
		if self.baseline:
			return self.base_model(x)

		return self.base_model(x)