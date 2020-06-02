import numpy as np 


def get_disp(u0, v0, t, omega):
	disp = u0 * np.cos(omega * t) + v0 / omega * np.sin( omega * t )
	return disp

def get_velo(u0, v0, t, omega):
	velo = - u0 * omega * np.sin(omega * t) + v0 * np.cos( omega * t )
	return velo

def get_acce(u0, v0, t, omega):
	acce = - u0 * omega**2 * np.cos(omega * t) - v0 * omega * np.sin( omega * t )
	return acce

def get_rawdata(t_span=[0,12.56], num_of_split=10, u0=None, v0=None, omega=1, noise_std=0.1):
	t_eval = np.linspace(t_span[0], t_span[1], int(num_of_split*t_span[1]-t_span[0]))

	if u0 is None:
		u0 = np.random.rand() * 1.8
	if v0 is None:
		v0 = np.random.rand() * 1.8
		# v0 = 0

	u = np.array([ get_disp(u0, v0, t, omega) for t in t_eval ])
	v = np.array([ get_velo(u0, v0, t, omega) for t in t_eval ])
	dudt = np.array([ get_velo(u0, v0, t, omega) for t in t_eval ])
	dvdt = np.array([ get_acce(u0, v0, t, omega) for t in t_eval ])



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