
import numpy as np 


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

def get_rawdata(t_span=[0,12.56], num_of_split=10, p0=None, k=1, omega=2, omega_n=1, noise_std=0.1):
	t_eval = np.linspace(t_span[0], t_span[1], int(num_of_split*t_span[1]-t_span[0]))

	if p0 is None:
		p0 = np.random.rand() * 1.8

	u = np.array([ get_disp(p0, k, t, omega, omega_n) for t in t_eval ])
	v = np.array([ get_velo(p0, k, t, omega, omega_n) for t in t_eval ])
	dudt = np.array([ get_velo(p0, k, t, omega, omega_n) for t in t_eval ])
	dvdt = np.array([ get_acce(p0, k, t, omega, omega_n) for t in t_eval ])


	u += np.random.randn(*u.shape) * noise_std
	v += np.random.randn(*v.shape) * noise_std
	return u, v, dudt, dvdt, t_eval

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
	data = {'meta': locals()}

	np.random.seed(seed)
	uv = []
	duv = []
	for idx in range(samples):
		u, v, dudt, dvdt, t = get_rawdata(**kwargs)
		uv.append( np.stack( [u,v] ).T )
		duv.append( np.stack( [dudt,dvdt] ).T )

	data['uv'] = np.concatenate(uv)
	data['duv'] = np.concatenate(duv)

	split_idx = int(len(data['uv']) * test_split ) 
	split_data = {}
	for d in ['uv', 'duv']:
		split_data[d], split_data['test_' + d] = data[d][:split_idx], data[d][split_idx:]
	data = split_data
	return data


def plot_dataset(**kwargs):
	dataset = get_dataset(**kwargs)
	import matplotlib.pyplot as plt
	uv = dataset['uv']
	duv = dataset['duv']
	print(' plot u, v ... ')
	plt.scatter(uv[:,0],uv[:,1])
	plt.show()

	print(' plot dudt, dvdt ... ')
	plt.scatter(duv[:,0],duv[:,1])
	plt.show()