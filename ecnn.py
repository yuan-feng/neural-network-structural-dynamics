
import torch
from nn_base import BaseNN

class ECNN(torch.nn.Module):
	def __init__(self, input_dim, base_model, baseline=False):
		super(ECNN, self).__init__()
		self.baseline = baseline
		self.base_model = base_model
		self.M = self.permutation_tensor(input_dim)

	def forward(self, x):
		if self.baseline:
			return self.base_model(x)
		y = self.base_model(x)
		return y.split(1,1)

	def time_derivative(self, x, t=None):
		if self.baseline:
			return self.base_model(x)

		F1, F2 = self.forward(x)

		differential = torch.zeros_like(x)
		diff1 = torch.zeros_like(x)
		dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]
		differential = dF2 @ self.M.t()
		return diff1 + differential

	def permutation_tensor(self, n):
		M = torch.eye(n)
		M = torch.cat([M[n//2:], -M[:n//2]])
		return M

