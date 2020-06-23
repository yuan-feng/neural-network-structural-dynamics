import numpy as np

damping_len = 6
def getInputAccVelDis():
	# ene -> energy
	acc, vel, dis, frq = [], [], [], []
	for i in [0,2,5,7,8,12,26,34,51,53,64,66,69,70]:
		acc_filename = 'processed_data/acc_{}.txt'.format(i)
		vel_filename = 'processed_data/vel_{}.txt'.format(i)
		dis_filename = 'processed_data/dis_{}.txt'.format(i)
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
			acc_filename = 'processed_data3/shell_structure_{}_{}_imposed_motion_node_38_x_acce.txt'.format(i,j)
			dis_filename = 'processed_data3/shell_structure_{}_{}_imposed_motion_node_38_x_disp.txt'.format(i,j)
			acc.append(np.loadtxt(acc_filename)[:,1])
			dis.append(np.loadtxt(dis_filename)[:,1])
	print(' len(acc), len(acc[0]), len(acc[-1]) = {}, {}, {} '.format( len(acc), len(acc[0]), len(acc[-1]) ))
	print(' len(dis), len(dis[0]), len(dis[-1]) = {}, {}, {} '.format( len(dis), len(dis[0]), len(dis[-1]) ))
	data = {}
	data['acc'] = acc
	data['dis'] = dis
	return data

# outputdata = getOutputAccDis()

def getData(baseline=False, min_frq = 3, max_frq = 5):
	inputdata = getInputAccVelDis()
	outputdata = getOutputAccDis()
	data = {}
	# input
	# f,a,v,d = frequency, acc, vel, dis
	favd = []
	# output
	# a, d = acc, dis
	ad = []
	for i in range( len(inputdata['acc'])  ):
		sz = min(len(inputdata['frq'][i]), len(outputdata['acc'][i]))
		f = inputdata['frq'][i][:sz]
		a = inputdata['acc'][i][:sz]
		v = inputdata['vel'][i][:sz]
		d = inputdata['dis'][i][:sz]
		ones = np.ones(sz)
		for j in range(damping_len):
			a_out = outputdata['acc'][i * damping_len + j ][:sz]
			d_out = outputdata['dis'][i * damping_len + j ][:sz]
			damping = 0.5 + 0.06 * j 
			damp_arr = damping * ones
			if not baseline:
				favd.append( np.stack([f,a,v,d,damp_arr]).T )
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



