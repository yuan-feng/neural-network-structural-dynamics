import matplotlib.pyplot as plt
import numpy as np 
from data import get_disp, get_velo, get_acce
u0 = 1 
damping = 0.1
t = np.linspace(0, 30, 1000)
omega = 1

disp = get_disp(u0, damping, t, omega)
velo = get_velo(u0, damping, t, omega)
acce = get_acce(u0, damping, t, omega)

plt.plot(t,disp,label='disp')
plt.plot(t,velo,label='velo')
plt.plot(t,acce,label='acce')
plt.legend()
plt.show()


