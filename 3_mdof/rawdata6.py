import numpy as np

damping_len = 6
def getInputAccVelDis():
	# ene -> energy
	acc_r, vel_r, dis_r = [], [], []
	acc_i, vel_i, dis_i = [], [], []
	frq = []
	for i in [0,2,5,7,8,12,26,34,51,53,64,66,69,70]:
		acc_filename = 'processed_data6/acc_{}_fft_complex.txt'.format(i)
		vel_filename = 'processed_data6/vel_{}_fft_complex.txt'.format(i)
		dis_filename = 'processed_data6/dis_{}_fft_complex.txt'.format(i)
		acc_d = np.loadtxt(acc_filename, dtype=np.complex_)
		vel_d = np.loadtxt(vel_filename, dtype=np.complex_)
		dis_d = np.loadtxt(dis_filename, dtype=np.complex_)
		frq.append(acc_d[:,0].real)
		acc_r.append(acc_d[:,1].real)
		acc_i.append(acc_d[:,1].imag)
		vel_r.append(vel_d[:,1].real)
		vel_i.append(vel_d[:,1].imag)
		dis_r.append(dis_d[:,1].real)
		dis_i.append(dis_d[:,1].imag)
	print(' len(frq), len(frq[0]), len(frq[-1]) = {}, {}, {} '.format( len(frq), len(frq[0]), len(frq[-1]) ))
	print(' len(acc), len(acc[0]), len(acc[-1]) = {}, {}, {} '.format( len(acc_i), len(acc_i[0]), len(acc_i[-1]) ))
	print(' len(acc), len(acc[0]), len(acc[-1]) = {}, {}, {} '.format( len(acc_r), len(acc_r[0]), len(acc_r[-1]) ))
	print(' len(vel), len(vel[0]), len(vel[-1]) = {}, {}, {} '.format( len(vel_i), len(vel_i[0]), len(vel_i[-1]) ))
	print(' len(vel), len(vel[0]), len(vel[-1]) = {}, {}, {} '.format( len(vel_r), len(vel_r[0]), len(vel_r[-1]) ))
	print(' len(dis), len(dis[0]), len(dis[-1]) = {}, {}, {} '.format( len(dis_i), len(dis_i[0]), len(dis_i[-1]) ))
	print(' len(dis), len(dis[0]), len(dis[-1]) = {}, {}, {} '.format( len(dis_r), len(dis_r[0]), len(dis_r[-1]) ))
	data = {}
	data['frq'] = frq
	data['acc_i'] = acc_i
	data['acc_r'] = acc_r
	data['vel_i'] = vel_i
	data['vel_r'] = vel_r
	data['dis_i'] = dis_i
	data['dis_r'] = dis_r
	return data

# inputdata = getInputAccVelDis()

def getOutputAccDis():
	acc_i, dis_i = [], []
	acc_r, dis_r = [], []
	for i in [0,2,5,7,8,12,26,34,51,53,64,66,69,70]:
		for j in range(damping_len):
			acc_filename = 'processed_data6/shell_structure_{}_{}_imposed_motion_node_38_x_acce_fft_complex.txt'.format(i,j)
			dis_filename = 'processed_data6/shell_structure_{}_{}_imposed_motion_node_38_x_disp_fft_complex.txt'.format(i,j)
			acc_d = np.loadtxt(acc_filename, dtype=np.complex_)[:,1]
			dis_d = np.loadtxt(dis_filename, dtype=np.complex_)[:,1]
			acc_r.append(acc_d.real)
			acc_i.append(acc_d.imag)
			dis_r.append(dis_d.real)
			dis_i.append(dis_d.imag)
	print(' len(acc), len(acc[0]), len(acc[-1]) = {}, {}, {} '.format( len(acc_r), len(acc_r[0]), len(acc_r[-1]) ))
	print(' len(acc), len(acc[0]), len(acc[-1]) = {}, {}, {} '.format( len(acc_i), len(acc_i[0]), len(acc_i[-1]) ))
	print(' len(dis), len(dis[0]), len(dis[-1]) = {}, {}, {} '.format( len(dis_r), len(dis_r[0]), len(dis_r[-1]) ))
	print(' len(dis), len(dis[0]), len(dis[-1]) = {}, {}, {} '.format( len(dis_i), len(dis_i[0]), len(dis_i[-1]) ))
	data = {}
	data['acc_i'] = acc_i
	data['acc_r'] = acc_r
	data['dis_i'] = dis_i
	data['dis_r'] = dis_r
	return data

# outputdata = getOutputAccDis()

def getData(baseline=False, min_frq = 1, max_frq = 10):
	inputdata = getInputAccVelDis()
	outputdata = getOutputAccDis()
	data = {}
	# input
	# f,a,v,d = frequency, acc, vel, dis
	favd = []
	# output
	# a, d = acc, dis
	ad = []
	for i in range( len(inputdata['acc_i'])  ):
		# sz = min(len(inputdata['frq'][i]), len(outputdata['acc'][i]))
		frq = f = inputdata['frq'][i]
		l = np.searchsorted(frq, min_frq)
		r = np.searchsorted(frq, max_frq)
		f = inputdata['frq'][i][l:r]
		a_i = inputdata['acc_i'][i][l:r]
		a_r = inputdata['acc_r'][i][l:r]
		v_i = inputdata['vel_i'][i][l:r]
		v_r = inputdata['vel_r'][i][l:r]
		d_i = inputdata['dis_i'][i][l:r]
		d_r = inputdata['dis_r'][i][l:r]
		a_out_i = outputdata['acc_i'][i][l:r]
		a_out_r = outputdata['acc_r'][i][l:r]
		d_out_i = outputdata['dis_i'][i][l:r]
		d_out_r = outputdata['dis_r'][i][l:r]
		ones = np.ones(r-l)
		for j in range(damping_len):
			damping = 0.5 + 0.06 * j 
			damp_arr = damping * ones
			if not baseline:
				favd.append( np.stack([f,a_r,v_r,d_r,damp_arr]).T )
			else:
				favd.append( np.stack([f,a_r,v_r,d_r]).T )
			# ad.append( np.stack([a_out_i,d_out_i,a_out_r,d_out_r]).T )
			ad.append( np.stack([a_out_r,d_out_r]).T )
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



