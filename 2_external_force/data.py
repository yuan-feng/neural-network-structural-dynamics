
import numpy as np 
import scipy.integrate as integrate

def get_disp(p0, k, t, omega, omega_n):
	"""
	p0     : force amplitude
	k      : stiffness of the system
	t      : time 
	omega  : frequency of the force p0
	omage_n: natural frequency of the system
	"""
	disp = p0 / k / ( 1 - (omega/omega_n)**2) * ( 
		np.sin(omega*t) - omega/omega_n * np.sin(omega_n * t)
		) 
	return disp

def get_velo(p0, k, t, omega, omega_n):
	velo = p0 / k * omega / ( 1 - (omega/omega_n)**2) * ( 
		np.cos(omega*t) - np.cos(omega_n * t)
		) 
	return velo

def get_acce(p0, k, t, omega, omega_n):
	acce = p0 / k * omega / ( 1 - (omega/omega_n)**2) * ( 
		- omega * np.sin(omega*t) + omega_n * np.sin(omega_n * t)
		) 
	return acce

def get_force(p0, t, omega):
	force = p0 * np.sin(omega*t)
	return force

def get_energy_input(p0, k, t, omega, omega_n, tol = 1e-6):
	energy_input, error = integrate.quad(lambda x: 
		get_force(p0, t, omega) * get_velo(p0, k, t, omega, omega_n), 
		t - np.pi*2/omega , t
		)
	if abs(error) > tol:
		print("integrate error [{}] greater than tol!".format(error))
	return energy_input

def get_rawdata(t_span=[0,12.56], num_of_split=10, p0=None, k=1, omega=2, omega_n=1, noise_std=0.1):
	t_eval = np.linspace(t_span[0], t_span[1], int(num_of_split*t_span[1]-t_span[0]))

	if p0 is None:
		p0 = np.random.rand() * 1.8

	u = np.array([ get_disp(p0, k, t, omega, omega_n) for t in t_eval ])
	v = np.array([ get_velo(p0, k, t, omega, omega_n) for t in t_eval ])
	dudt = np.array([ get_velo(p0, k, t, omega, omega_n) for t in t_eval ])
	dvdt = np.array([ get_acce(p0, k, t, omega, omega_n) for t in t_eval ])
	f = np.array([ get_force(p0, t, omega) for t in t_eval ])
	e = np.array([ get_energy_input(p0, k, t, omega, omega_n) for t in t_eval ])

	u += np.random.randn(*u.shape) * noise_std
	v += np.random.randn(*v.shape) * noise_std
	return u, v, f, e, dudt, dvdt, t_eval

def get_dataset(baseline=True, seed=0, samples=50, test_split=0.5, **kwargs):
	data = {'meta': locals()}

	np.random.seed(seed)
	uvf = []
	duv = []
	for idx in range(samples):
		u, v, f, e, dudt, dvdt, t = get_rawdata(**kwargs)
		if baseline:
			uvf.append( np.stack( [u,v,f] ).T )
		else:
			uvf.append( np.stack( [u,v,f,e] ).T )
		duv.append( np.stack( [dudt,dvdt] ).T )

	data['uvf'] = np.concatenate(uvf)
	data['duv'] = np.concatenate(duv)

	split_idx = int(len(data['uvf']) * test_split ) 
	split_data = {}
	for d in ['uvf', 'duv']:
		split_data[d], split_data['test_' + d] = data[d][:split_idx], data[d][split_idx:]
	data = split_data
	return data


def plot_dataset(**kwargs):
	dataset = get_dataset(**kwargs)
	import matplotlib.pyplot as plt
	uvf = dataset['uvf']
	duv = dataset['duv']
	print(' plot u, v ... ')
	plt.scatter(uvf[:,0],uvf[:,1])
	plt.show()

	print(' plot dudt, dvdt ... ')
	plt.scatter(duv[:,0],duv[:,1])
	plt.show()