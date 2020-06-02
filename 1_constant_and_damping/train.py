import torch
import numpy as np

from nn_base import BaseNN
from ecnn import ECNN
from data import get_dataset
from util import L2_loss
import argparse
import os 

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
	parser = argparse.ArgumentParser(description='parameters setting')
	parser.add_argument('--seed', default=0, type=int, help='seed for random number')
	parser.add_argument('--input_dim', default=2, type=int, help='input tensor dimension')
	parser.add_argument('--hidden_dim', default=200, type=int, help='hidden tensor dimension')
	parser.add_argument('--learn_rate', default=1e-3, type=float, help='hidden tensor dimension')
	parser.add_argument('--num_steps', default=1600, type=int, help='number of steps')
	parser.add_argument('--print_every', default=200, type=int, help='print every n steps')
	parser.add_argument('--name', default='dynamics', type=str, help='output name')
	parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or energy conserving')
	parser.add_argument('--save_dir', default=FILE_DIR, type=str, help='dir to save the trained model')
	return parser.parse_args()


def train(args):
	first_call = True
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	output_dim = 2 
	nn_model = BaseNN(args.input_dim, args.hidden_dim, output_dim)
	model = ECNN(args.input_dim, nn_model, args.baseline )
	optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
	data = get_dataset(seed=args.seed)

	uv = torch.tensor(data['uv'], requires_grad=True, dtype=torch.float32 )
	test_uv = torch.tensor(data['test_uv'], requires_grad=True, dtype=torch.float32 )
	duv = torch.Tensor(data['duv'])
	test_duv = torch.Tensor(data['test_duv'])

	stats = {'train_loss': [], 'test_loss': []}
	for step in range(args.num_steps + 1):
		# train step
		duv_hat = model.time_derivative(uv)
		loss = L2_loss(duv, duv_hat)
		if not args.baseline:
			u, v = uv.split(1,1)
			du, dv = duv_hat.split(1,1)

			energy1 = (u**2 + v**2) 
			energy2 = (du**2 + dv**2)
			add_loss = L2_loss(energy1, energy2)
			loss += add_loss
		loss.backward(); optim.step(); optim.zero_grad();

		# run test data
		test_duv_hat = model.time_derivative(test_uv)
		test_loss = L2_loss(test_duv, test_duv_hat)

		# logging
		stats['train_loss'].append(loss.item())
		stats['test_loss'].append(test_loss.item())
		if step % args.print_every == 0:
			print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
	
	train_duv_hat = model.time_derivative(uv)
	train_dist = (duv - train_duv_hat)**2
	test_duv_hat = model.time_derivative(test_uv)
	test_dist = (test_duv - test_duv_hat)**2
	print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
		.format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
			test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

	return model, stats

if __name__ == "__main__":
	args = parse_args()
	model, stats = train(args)

	# save
	os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
	naming = '-baseline' if args.baseline else ''
	path = '{}/{}{}.tar'.format(args.save_dir, args.name, naming)
	torch.save(model.state_dict(), path)
