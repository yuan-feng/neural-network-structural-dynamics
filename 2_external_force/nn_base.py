import torch

class BaseNN(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(BaseNN, self).__init__()
		self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.linear4 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
		for layer in [self.linear1, self.linear2, self.linear3, self.linear4]:
			torch.nn.init.orthogonal_(layer.weight)
		self.activation = torch.tanh

	def forward(self, x):
		h1 = self.activation(self.linear1(x))
		h2 = self.activation(self.linear2(h1)) + h1
		h3 = self.activation(self.linear3(h2)) + h2 + h1
		return self.linear4(h3)

