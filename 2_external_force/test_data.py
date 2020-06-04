import matplotlib.pyplot as plt
import numpy as np 
from data import get_disp, get_velo, get_acce
p0 = 1 
k = 1
t = np.linspace(0, 30, 1000)
omega = 2
omega_n = 1

disp = get_disp(p0, k, t, omega, omega_n)
velo = get_velo(p0, k, t, omega, omega_n)
acce = get_acce(p0, k, t, omega, omega_n)

plt.plot(t,disp,label='disp')
plt.plot(t,velo,label='velo')
plt.plot(t,acce,label='acce')
plt.legend()
plt.show()


