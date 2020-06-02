import torch
import os
from data import get_dataset, get_rawdata, get_velo
import matplotlib.pyplot as plt
from nn_base import BaseNN
from ecnn import ECNN
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import numpy as np

# Constants
args_dict = {
    'seed': 0,
    'input_dim': 2, 
    'hidden_dim': 200,
    'learn_rate': 1e-3,
    'num_steps': 2000,
    'print_every': 200,
    'name': 'dynamics'
}
class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d
args = ObjectView(args_dict)

data = get_dataset()

# Construct Models
def get_model(args, baseline):
    output_dim = 2
    nn_model = BaseNN(args.input_dim, args.hidden_dim, output_dim)
    model = ECNN( args.input_dim, nn_model, baseline )
    naming = '-baseline' if baseline else ''
    path = 'dynamics{}.tar'.format(naming)
    model.load_state_dict(torch.load(path))
    return nn_model

# Integrate Model
def integrate_model(model, t_span, uv0, **kwargs):
    damping = kwargs.pop('damping', 0)
    def fun(t, uv_in):
        scale = np.exp( - t * damping)
        uv = torch.tensor( uv_in, requires_grad=True, dtype=torch.float32 ).view(1,2)
        duv = model.forward(uv).data.numpy().reshape(-1)
        duv[0] = duv[0] * scale
        return duv
    return solve_ivp(fun=fun, t_span=t_span, y0=uv0, **kwargs)

# Analysis
base_model = get_model(args, True)
ecnn_model = get_model(args, False)
t_span = [0,300]
uv0 = np.array([1.0, 0.])
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 20000), 'rtol': 1e-12, 'damping':0.02}
base_ivp = integrate_model( base_model, t_span, uv0, **kwargs )
ecnn_ivp = integrate_model( ecnn_model, t_span, uv0, **kwargs )

LINE_SEGMENTS = 10
base_v = []
for i, l in enumerate(np.split(base_ivp['y'].T, LINE_SEGMENTS)):
    base_v = np.concatenate((base_v, l[:,0]))

fig = plt.figure()
LINE_SEGMENTS = 10
ecnn_v = []
for i, l in enumerate(np.split(ecnn_ivp['y'].T, LINE_SEGMENTS)):
    tt = np.linspace(0, t_span[1], len(l[:,0])) + t_span[1] * i
    ecnn_v = np.concatenate((ecnn_v, l[:,0]))


# groundtruth
num_of_split = 1000
base_t = np.linspace(0, t_span[1], len(base_v))
u0 = 0
v0 = base_v[0]
omage = 0.1
true_v = get_velo(u0, v0, base_t, omage)

plt.title('comparison')
plt.plot(base_t, base_v, label='base')
plt.plot(base_t, ecnn_v, label='ecnn')
plt.plot(base_t, true_v, label='groundtruth')
plt.legend()
plt.show()

