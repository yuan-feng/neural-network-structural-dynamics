import matplotlib.pyplot as plt
import numpy as np 
from data import get_disp, get_velo, get_acce, get_energy_input, get_force
p0 = 1 
k = 1
t = np.linspace(0, 90, 1000)
omega = 2
omega_n = 1

disp = get_disp(p0, k, t, omega, omega_n)
velo = get_velo(p0, k, t, omega, omega_n)
acce = get_acce(p0, k, t, omega, omega_n)
ener = [get_energy_input(p0, k, t_, omega, omega_n) for t_ in t]
forc = get_force(p0, t, omega)

# plt.plot(t,disp,label='disp')
# plt.plot(t,velo,label='velo')
# plt.plot(t,acce,label='acce')
plt.plot(t,ener,label='ener')
# plt.plot(t,forc,label='forc')
plt.legend()
plt.show()


