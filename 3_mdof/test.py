import numpy as np
import matplotlib.pyplot as plt

data1_a = np.loadtxt('data1_a.txt')
data1_u = np.loadtxt('data1_u.txt')

data2_a = np.loadtxt('data2_a.txt')
data2_u = np.loadtxt('data2_u.txt')

plt.plot(data1_a[:,0], data1_a[:,1]*3, label='input1')
plt.plot(data1_u[:,0], data1_u[:,1]*100, label='input2')

plt.plot(data2_a[:,0], data2_a[:,1], label='output')

plt.legend()
plt.show()




