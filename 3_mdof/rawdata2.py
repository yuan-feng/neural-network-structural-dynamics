import numpy as np

max_len = 20 
data_len = 20
damp_len = 10
def getInputAccVelDis():
	# ene -> energy
	acc, vel, dis,frq, ene = [], [], [], [], []
	for i in range(1,data_len+1):
		w = 2 * i + 1
		acc_filename = 'processed_data/acc_{}_acce_freq_ampl.txt'.format(w)
		vel_filename = 'processed_data/vel_{}_freq_ampl.txt'.format(w)
		dis_filename = 'processed_data/dis_{}_freq_ampl.txt'.format(w)
		acc_d = np.loadtxt(acc_filename)
		vel_d = np.loadtxt(vel_filename)
		dis_d = np.loadtxt(dis_filename)
		frq.append(acc_d[:,0])
		acc.append(acc_d[:,1])
		vel.append(vel_d[:,1])
		dis.append(dis_d[:,1])
		ene.append(vel_d[:,1]*acc_d[:,1])
	print(' len(frq), len(frq[0]), len(frq[data_len-2]) = {}, {}, {} '.format( len(frq), len(frq[0]), len(frq[data_len-2]) ))
	print(' len(acc), len(acc[0]), len(acc[data_len-2]) = {}, {}, {} '.format( len(acc), len(acc[0]), len(acc[data_len-2]) ))
	print(' len(vel), len(vel[0]), len(vel[data_len-2]) = {}, {}, {} '.format( len(vel), len(vel[0]), len(vel[data_len-2]) ))
	print(' len(dis), len(dis[0]), len(dis[data_len-2]) = {}, {}, {} '.format( len(dis), len(dis[0]), len(dis[data_len-2]) ))
	print(' len(ene), len(ene[0]), len(ene[data_len-2]) = {}, {}, {} '.format( len(ene), len(ene[0]), len(ene[data_len-2]) ))
	data = {}
	data['frq'] = frq
	data['acc'] = acc
	data['vel'] = vel
	data['dis'] = dis
	data['ene'] = ene
	return data

# inputdata = getInputAccVelDis()

def getOutputAccDis():
	acc, dis = [], []
	for i in range(1,data_len+1):
		for j in range(1, damp_len+1):
			acc_filename = 'processed_data2/shell_structure_{}_{}_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i,j)
			dis_filename = 'processed_data2/shell_structure_{}_{}_imposed_motion_node_38_x_disp_freq_ampl.txt'.format(i,j)
			acc.append(np.loadtxt(acc_filename)[:,1])
			dis.append(np.loadtxt(dis_filename)[:,1])
	print(' len(acc), len(acc[0]), len(acc[data_len-2]) = {}, {}, {} '.format( len(acc), len(acc[0]), len(acc[data_len-2]) ))
	print(' len(dis), len(dis[0]), len(dis[data_len-2]) = {}, {}, {} '.format( len(dis), len(dis[0]), len(dis[data_len-2]) ))
	data = {}
	data['acc'] = acc
	data['dis'] = dis
	return data

# outputdata = getOutputAccDis()

def getData(baseline=False):
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
		a_out = outputdata['acc'][i][:sz]
		d_out = outputdata['dis'][i][:sz]
		ones = np.ones(sz)
		# damping 1 - 10
		for j in range(1,11):
			damping = 0.5 + 0.01 * j 
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



