import torch
import os
from rawdata3 import getData, getInputAccVelDis, getOutputAccDis
import matplotlib.pyplot as plt
from nn_base import BaseNN
from ecnn import ECNN
import numpy as np

# Constants
args_dict = {
    'seed': 0,
    'input_dim': 4, 
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

# Construct Models
def get_model(args, baseline):
    output_dim = 2
    nn_model = BaseNN(args.input_dim, args.hidden_dim, output_dim, baseline)
    model = ECNN( nn_model, baseline )
    naming = '-baseline' if baseline else ''
    path = 'dynamics{}.tar'.format(naming)
    model.load_state_dict(torch.load(path))
    return nn_model

def getPredict(model, frq, acc, vel, dis, j, baseline):
  omega_n = 3.473873
  # print('frq = {}'.format(frq))
  # print('acc = {}'.format(acc))
  # print('vel = {}'.format(vel))
  # print('dis = {}'.format(dis))
  sz = min(len(frq), len(acc), len(vel), len(dis))
  # print(' sz = {}'.format(sz))
  o_acc, o_dis = [], []
  for i in range(sz):
    f, a, v, d = frq[i], acc[i], vel[i], dis[i]
    favd = torch.tensor([f,a,v,d], requires_grad=True, dtype=torch.float32).view(1, args.input_dim)
    if not baseline:
      omega = f/omega_n
      damping = 0.5 + 0.06 * j 
      Ra = omega**2 / np.sqrt((1-omega**2)**2 + (2*damping*omega)**2) 
      Rd = 1 / np.sqrt((1-omega**2)**2 + (2*damping*omega)**2)
      favd = torch.tensor([f,a,v,d, Ra, Rd], requires_grad=True, dtype=torch.float32).view(1, args.input_dim+2)
    [a, d] = model.forward(favd).data.numpy().reshape(-1)
    o_acc.append(a)
    o_dis.append(d)
  return o_acc, o_dis

# Analysis
base_model = get_model(args, True)
ecnn_model = get_model(args, False)

input_data = getInputAccVelDis()
output_data = getOutputAccDis()

in_frq_all = input_data['frq']
in_acc_all = input_data['acc']
in_vel_all = input_data['vel']
in_dis_all = input_data['dis']

out_acc_all = output_data['acc']
out_dis_all = output_data['dis']

i = 10
j = 5
damping_len = 6
o_i = i * damping_len + j

in_frq = in_frq_all[i]
in_acc = in_acc_all[i]
in_vel = in_vel_all[i]
in_dis = in_dis_all[i]
out_acc = out_acc_all[o_i]
out_dis = out_dis_all[o_i]
sz = min(len(in_frq), len(out_acc), len(out_dis))
true_frq = in_frq[:sz]
true_acc = out_acc[:sz]
true_dis = out_dis[:sz]

nn_acc, nn_dis = getPredict(base_model, in_frq, in_acc, in_vel, in_dis, j, True)
ecnn_acc, ecnn_dis = getPredict(ecnn_model, in_frq, in_acc, in_vel, in_dis, j, False)


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
f, (ax1) = plt.subplots(1, 1)
plt.semilogx(true_frq, true_acc, label='acc')
plt.semilogx(true_frq, nn_acc, label='nn')
plt.semilogx(true_frq, ecnn_acc, label='ecnn')
plotFreq_min = 0.1
plotFreq_max = 20
ax1.set( xlabel = "Frequency [Hz] ",
         ylabel = "Acceleration [m^2/s] ",
         xlim = [plotFreq_min,plotFreq_max]
         # ,title = "FFT of Acceleration"
         )
a = np.linspace(0.1,1,10)
b = np.linspace(2,10,9) 
c = np.concatenate((a,b))
plt.xticks(c)
plt.grid()
plt.legend()
plt.savefig( "mdof_compare_acc.jpg",bbox_inches="tight")
plt.show()




f, (ax1) = plt.subplots(1, 1)
plt.semilogx(true_frq, nn_acc - true_acc, label='nn')
plt.semilogx(true_frq, ecnn_acc - true_acc, label='ecnn')
plotFreq_min = 0.1
plotFreq_max = 20
ax1.set( xlabel = "Frequency [Hz] ",
         ylabel = "Acceleration Error[m^2/s] ",
         xlim = [plotFreq_min, plotFreq_max] 
         # , title = "Error of Acceleration"
         )
a = np.linspace(0.1,1,10)
b = np.linspace(2,10,9) 
c = np.concatenate((a,b))
plt.xticks(c)
plt.grid()
plt.legend()
plt.savefig( "mdof_compare_acc_error.jpg",bbox_inches="tight")
plt.show()



f, (ax1) = plt.subplots(1, 1)
plt.semilogx(true_frq, nn_dis, label='nn')
plt.semilogx(true_frq, ecnn_dis, label='ecnn')
plotFreq_min = 0.1
plotFreq_max = 20
ax1.set( xlabel = "Frequency [Hz] ",
         ylabel = "Displacement [m] ",
         xlim = [plotFreq_min, plotFreq_max] 
         # , title = "Error of Displacement"
         )
a = np.linspace(0.1,1,10)
b = np.linspace(2,10,9) 
c = np.concatenate((a,b))
plt.xticks(c)
plt.grid()
plt.legend()
plt.savefig( "mdof_compare_dis.jpg",bbox_inches="tight")
plt.show()




f, (ax1) = plt.subplots(1, 1)
plt.semilogx(true_frq, nn_dis - true_dis, label='nn')
plt.semilogx(true_frq, ecnn_dis - true_dis, label='ecnn')
plotFreq_min = 0.1
plotFreq_max = 20
ax1.set( xlabel = "Frequency [Hz] ",
         ylabel = "Displacement Error[m] ",
         xlim = [plotFreq_min, plotFreq_max] 
         # , title = "Error of Displacement"
         )
a = np.linspace(0.1,1,10)
b = np.linspace(2,10,9) 
c = np.concatenate((a,b))
plt.xticks(c)
plt.grid()
plt.legend()
plt.savefig( "mdof_compare_dis_error.jpg",bbox_inches="tight")
plt.show()


