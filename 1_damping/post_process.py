import torch
import os
from data import get_dataset, get_rawdata, get_velo, get_disp
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
t_span = [0,30]
uv0 = np.array([1.0, 0.])
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 2000), 'rtol': 1e-12, 'damping':0.01}
base_ivp = integrate_model( base_model, t_span, uv0, **kwargs )
ecnn_ivp = integrate_model( ecnn_model, t_span, uv0, **kwargs )

LINE_SEGMENTS = 10
base_dis = []
base_vel = []
for i, l in enumerate(np.split(base_ivp['y'].T, LINE_SEGMENTS)):
    base_dis = np.concatenate((base_dis, l[:,0]))
    base_vel = np.concatenate((base_vel, l[:,1]))

fig = plt.figure()
LINE_SEGMENTS = 10
ecnn_dis = []
ecnn_vel = []
for i, l in enumerate(np.split(ecnn_ivp['y'].T, LINE_SEGMENTS)):
    ecnn_dis = np.concatenate((ecnn_dis, l[:,0]))
    ecnn_vel = np.concatenate((ecnn_vel, l[:,1]))


# groundtruth
base_t = np.linspace(0, t_span[1], len(base_dis))
u0 = uv0[0]
v0 = uv0[1]
omage = 1
# true_v = get_velo(u0, v0, base_t, omage)
damp = kwargs['damping']
damp = 0.1
true_dis = get_disp(u0, damp, base_t, omage)

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
plt.savefig( "sdof_damping_comparison_disp.jpg",bbox_inches="tight")
plt.show()


# # #############################
# # #############################
# energy_base_nn = base_dis**2 + base_vel**2
# energy_ecnn = ecnn_dis**2 + ecnn_vel**2

# plt.plot(base_t, energy_base_nn, color='b', label='Baseline Neural Network')
# plt.plot(base_t, energy_ecnn, color='r', label='Energy Constant Neural Network')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, shadow=True
#           # , ncol=3
#           )
# plt.xlabel(" Time [second] ")
# plt.ylabel(" System Energy")
# # plt.yticks(np.arange(energy_base_nn.min(), energy_base_nn.max()*1.01, 0.1))
# plt.grid()
# plt.savefig( "sdof_damping_comparison_energy.jpg",bbox_inches="tight")
# plt.show()

def getPeaks(dis):
  peaks = []
  n_steps = len(dis)
  for i in range(n_steps-2):
    if (dis[i+1] - dis[i]) * (dis[i+1]-dis[i+2]) > 0:
      peaks.append(i+1)
  return peaks



def getLag(true_dis, calc_dis):
  true_peaks = getPeaks(true_dis)
  calc_peaks = getPeaks(calc_dis)
  l = min(len(true_peaks), len(calc_peaks))
  lags = []
  steps = []
  for i in range(l):
    lags.append(calc_peaks[i] - true_peaks[i])
    steps.append(calc_peaks[i])
  steps = np.array(steps) / len(base_dis) * t_span[1]
  lags = np.array(lags) / len(base_dis) * t_span[1]
  return lags, steps

base_lags, base_steps = getLag(true_dis, base_dis)
ecnn_lags, ecnn_steps = getLag(true_dis, ecnn_dis)

print('base_steps = {}'.format(base_steps))
print('base_lags = {}'.format(base_lags))
print('\n')
print('ecnn_steps = {}'.format(ecnn_steps))
print('ecnn_lags = {}'.format(ecnn_lags))

plt.xlabel(" Time [second] ")
plt.ylabel(" Lag [second] ")
plt.plot(base_steps, base_lags, color='b', label='Baseline Neural Network')
plt.plot(ecnn_steps, ecnn_lags, color='r', label='Energy Constant Neural Network')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True
          # , ncol=3
          )
plt.xticks(np.arange(0, 31, 5))
plt.grid()
plt.savefig( "sdof_damping_comparison_lag.jpg",bbox_inches="tight")
plt.show()


# def getLag(base_t, true_dis, calc_dis):
#   n_steps = len(true_dis)
#   lag = np.zeros(n_steps)
#   for i in range(1,n_steps):
#     lag[i] = i
#     val = calc_dis[i]
#     for j in range(i):
#       if (val - true_dis[j]) * (val - true_dis[j+1]) < 0:
#         lag[i] = i-j
#         break
#   return lag

# base_lag = getLag(base_t, true_dis, base_dis)
# ecnn_lag = getLag(base_t, true_dis, ecnn_dis)
# plt.plot(base_t, base_lag)
# plt.plot(base_t, ecnn_lag)
# plt.show()

