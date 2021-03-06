import torch

class BaseNN(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, baseline=True):
		super(BaseNN, self).__init__()
		if not baseline:
			input_dim = input_dim + 2
		self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
		for layer in [self.linear1, self.linear2, self.linear3]:
			torch.nn.init.orthogonal_(layer.weight)
		self.activation = torch.relu

	def forward(self, x):
		h = self.activation(self.linear1(x))
		h = self.activation(self.linear2(h))
		return self.linear3(h)

