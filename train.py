import torch
import numpy as np

from nn_base import BaseNN
from data import get_dataset
from util import L2_loss

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='parameters setting')
	parser.add_argument('--seed', default=0, type=int, help='seed for random number')
	parser.add_argument('--input_dim', default=2, type=int, help='input tensor dimension')
	parser.add_argument('--hidden_dim', default=200, type=int, help='hidden tensor dimension')
	parser.add_argument('--learn_rate', default=1e-3, type=float, help='hidden tensor dimension')
	parser.add_argument('--num_steps', default=2000, type=int, help='number of steps')
	parser.add_argument('--print_every', default=200, type=int, help='print every n steps')
	return parser.parse_args()

def train(args):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	output_dim = 2 
	nn_model = BaseNN(args.input_dim, args.hidden_dim, output_dim)
	optim = torch.optim.Adam(nn_model.parameters(), args.learn_rate, weight_decay=1e-4)

	data = get_dataset(seed=args.seed)

	uv = torch.tensor(data['uv'], requires_grad=True, dtype=torch.float32 )
	test_uv = torch.tensor(data['test_uv'], requires_grad=True, dtype=torch.float32 )
	duv = torch.Tensor(data['duv'])
	test_duv = torch.Tensor(data['test_duv'])

	stats = {'train_loss': [], 'test_loss': []}
	for step in range(args.num_steps + 1):
		# train step
		duv_hat = nn_model.forward(uv)
		loss = L2_loss(duv, duv_hat)
		loss.backward(); optim.step(); optim.zero_grad();

		# run test data
		test_duv_hat = nn_model.forward(test_uv)
		test_loss = L2_loss(test_duv, test_duv_hat)

		# logging
		stats['train_loss'].append(loss.item())
		stats['test_loss'].append(test_loss.item())
		if step % args.print_every == 0:
			print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
	
	train_duv_hat = nn_model.forward(uv)
	train_dist = (duv - train_duv_hat)**2
	test_duv_hat = nn_model.forward(test_uv)
	test_dist = (test_duv - test_duv_hat)**2
	print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
		.format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
			test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

	return nn_model, stats

if __name__ == "__main__":
	args = parse_args()
	model, stats = train(args)

