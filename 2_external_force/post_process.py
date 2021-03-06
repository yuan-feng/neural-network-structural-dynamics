import torch
import os
from data import get_dataset, get_rawdata, get_velo, get_disp, get_force, get_energy_input
import matplotlib.pyplot as plt
from nn_base import BaseNN
from ecnn import ECNN
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import numpy as np

# Constants
args_dict = {
    'seed': 0,
    'input_dim': 3, 
    'hidden_dim': 100,
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
    nn_model = BaseNN(args.input_dim, args.hidden_dim, output_dim, baseline)
    model = ECNN( nn_model, baseline )
    naming = '-baseline' if baseline else ''
    path = 'dynamics{}.tar'.format(naming)
    model.load_state_dict(torch.load(path))
    return nn_model

# Integrate Model
def integrate_model(model, t_span, p0, k, omega, omega_n, uv0, baseline, **kwargs):
    damping = kwargs.pop('damping', 0)
    def fun(t, uv_in):
        scale = np.exp( - t * damping)
        f = get_force(p0, t, omega)
        uvf_in = np.append(uv_in, f)
        input_sz = 3
        if not baseline:
          e = get_energy_input(p0, k, t, omega, omega_n)
          uvf_in = np.append(uvf_in, e)
          input_sz = input_sz + 1
        uvf = torch.tensor( uvf_in, requires_grad=True, dtype=torch.float32 ).view(1,input_sz)
        duv = model.forward(uvf).data.numpy().reshape(-1)
        duv[0] = duv[0] * scale
        return duv
    return solve_ivp(fun=fun, t_span=t_span, y0=uv0, **kwargs)

# Analysis
base_model = get_model(args, True)
ecnn_model = get_model(args, False)
t_span = [0,30]
uv0 = np.array([0., 0.])
p0 = 1
omega = 2 
k=1
omega_n = 1
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 2000), 'rtol': 1e-12, 'damping':0.0}
base_ivp = integrate_model( base_model, t_span, p0, k, omega, omega_n, uv0, baseline=True,  **kwargs )
ecnn_ivp = integrate_model( ecnn_model, t_span, p0, k, omega, omega_n, uv0, baseline=False, **kwargs )

LINE_SEGMENTS = 10
base_dis = []
base_vel = []
for i, l in enumerate(np.split(base_ivp['y'].T, LINE_SEGMENTS)):
    base_dis = np.concatenate((base_dis, l[:,0]))
    base_vel = np.concatenate((base_vel, l[:,1]))

ecnn_dis = []
ecnn_vel = []
for i, l in enumerate(np.split(ecnn_ivp['y'].T, LINE_SEGMENTS)):
    ecnn_dis = np.concatenate((ecnn_dis, l[:,0]))
    ecnn_vel = np.concatenate((ecnn_vel, l[:,1]))


# groundtruth
base_t = np.linspace(0, t_span[1], len(base_dis))
omage = 1
p0 = 1 
k = 1
omega = 2
omega_n = 1

damp = kwargs['damping']
damp = 0.1
true_dis = get_disp(p0, k, base_t, omega, omega_n)

# #############################
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (10, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
# #############################

# #############################
# #############################
# plt.title('comparison')
plt.plot(base_t, base_dis, color='b', label='Baseline Neural Network')
plt.plot(base_t, ecnn_dis, color='r', label='Energy Constant Neural Network')
plt.plot(base_t, true_dis, color='k', label='Groundtruth')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True
          # , ncol=3
          )
plt.xlabel(" Time [second] ")
plt.ylabel(" Displacement [meters] ")
plt.yticks(np.arange(true_dis.min(), true_dis.max()*1.01, 0.5))
plt.grid()
plt.savefig( "sdof_external_force_comparison_disp.jpg",bbox_inches="tight")
plt.show()

# # # #############################
base_error = abs(base_dis - true_dis)
ecnn_error = abs(ecnn_dis - true_dis)

plt.plot(base_t, base_error, color='b', label='Baseline Neural Network')
plt.plot(base_t, ecnn_error, color='r', label='Energy Constant Neural Network')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True
          # , ncol=3
          )
plt.xlabel(" Time [second] ")
plt.ylabel(" Displacement Error [meter] ")
# plt.yticks(np.arange(energy_base_nn.min(), energy_base_nn.max()*1.01, 0.1))
plt.grid()
plt.savefig( "sdof_external_force_comparison_error.jpg",bbox_inches="tight")
plt.show()





