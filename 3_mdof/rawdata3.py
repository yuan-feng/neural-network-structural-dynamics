import numpy as np
import matplotlib.pyplot as plt
damping_len = 6
def getInputAccVelDis():
	# ene -> energy
	acc, vel, dis, frq = [], [], [], []
	for i in [0,2,5,7,8,12,26,34,51,53,64,66,69,70]:
		acc_filename = 'processed_data/acc_{}_freq_ampl.txt'.format(i)
		vel_filename = 'processed_data/vel_{}_freq_ampl.txt'.format(i)
		dis_filename = 'processed_data/dis_{}_freq_ampl.txt'.format(i)
		acc_d = np.loadtxt(acc_filename)
		vel_d = np.loadtxt(vel_filename)
		dis_d = np.loadtxt(dis_filename)
		frq.append(acc_d[:,0])
		acc.append(acc_d[:,1])
		vel.append(vel_d[:,1])
		dis.append(dis_d[:,1])
	print(' len(frq), len(frq[0]), len(frq[-1]) = {}, {}, {} '.format( len(frq), len(frq[0]), len(frq[-1]) ))
	print(' len(acc), len(acc[0]), len(acc[-1]) = {}, {}, {} '.format( len(acc), len(acc[0]), len(acc[-1]) ))
	print(' len(vel), len(vel[0]), len(vel[-1]) = {}, {}, {} '.format( len(vel), len(vel[0]), len(vel[-1]) ))
	print(' len(dis), len(dis[0]), len(dis[-1]) = {}, {}, {} '.format( len(dis), len(dis[0]), len(dis[-1]) ))
	data = {}
	data['frq'] = frq
	data['acc'] = acc
	data['vel'] = vel
	data['dis'] = dis
	return data

# inputdata = getInputAccVelDis()

def getOutputAccDis():
	acc, dis = [], []
	for i in [0,2,5,7,8,12,26,34,51,53,64,66,69,70]:
		for j in range(damping_len):
			acc_filename = 'processed_data3/shell_structure_{}_{}_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i,j)
			dis_filename = 'processed_data3/shell_structure_{}_{}_imposed_motion_node_38_x_disp_freq_ampl.txt'.format(i,j)
			acc.append(np.loadtxt(acc_filename)[:,1])
			dis.append(np.loadtxt(dis_filename)[:,1])
	print(' len(acc), len(acc[0]), len(acc[-1]) = {}, {}, {} '.format( len(acc), len(acc[0]), len(acc[-1]) ))
	print(' len(dis), len(dis[0]), len(dis[-1]) = {}, {}, {} '.format( len(dis), len(dis[0]), len(dis[-1]) ))
	data = {}
	data['acc'] = acc
	data['dis'] = dis
	return data

# outputdata = getOutputAccDis()
def getData(baseline=False, min_frq = 0.01, max_frq = 30):
	omega_n = 3.473873
	inputdata = getInputAccVelDis()
	outputdata = getOutputAccDis()
	data = {}
	# input
	# f,a,v,d = frequency, acc, vel, dis
	favd = []
	# output
	# a, d = acc, dis
	ad = []
	# first = True
	for i in range( len(inputdata['acc'])  ):
		# sz = min(len(inputdata['frq'][i]), len(outputdata['acc'][i]))
		frq = f = inputdata['frq'][i]
		l = np.searchsorted(frq, min_frq)
		r = np.searchsorted(frq, max_frq)
		f = inputdata['frq'][i][l:r]
		a = inputdata['acc'][i][l:r]
		v = inputdata['vel'][i][l:r]
		d = inputdata['dis'][i][l:r]
		omega = f/omega_n
		for j in range(damping_len):
			a_out = outputdata['acc'][ i * damping_len + j ][l:r]
			d_out = outputdata['dis'][ i * damping_len + j ][l:r]
			damping = 0.5 + 0.06 * j 
			Ra = omega**2 / np.sqrt((1-omega**2)**2 + (2*damping*omega)**2) 
			Rd = 1 / np.sqrt((1-omega**2)**2 + (2*damping*omega)**2) 
			# if first:
			# 	first=False
			# 	plt.plot(f, Rd)
			# 	plt.show()
			if not baseline:
				favd.append( np.stack([f,a,v,d,Ra,Rd]).T )
			else:
				favd.append( np.stack([f,a,v,d]).T )
			ad.append( np.stack([a_out,d_out]).T )
	data['favd'] = np.concatenate(favd)
	data['ad'] = np.concatenate(ad)
	# print("data['favd'].shape = {}".format(data['favd'].shape) )
	# print("data['ad'].shape = {}".format(data['ad'].shape) )
	split_idx = int(len(data['favd']) * 0.5)
	split_data = {}
	for d in data.keys():
		split_data[d], split_data['test_' + d] = \
		data[d][:split_idx], data[d][split_idx:]
	data = split_data
	return data

# data = getData()



