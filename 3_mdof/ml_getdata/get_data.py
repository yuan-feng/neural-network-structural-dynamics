import numpy as np

def getInputAccVelDis():
	acc, vel, dis = [], [], []
	for i in range(1,76):
		w = 2 * i + 1
		acc_filename = '../processed_data/acc_{}_acce_freq_ampl.txt'.format(w)
		vel_filename = '../processed_data/vel_{}_freq_ampl.txt'.format(w)
		dis_filename = '../processed_data/dis_{}_freq_ampl.txt'.format(w)
		acc.append(np.loadtxt(acc_filename))
		vel.append(np.loadtxt(vel_filename))
		dis.append(np.loadtxt(dis_filename))
	print(' len(acc), len(acc[0]), len(acc[74]) = {}, {}, {} '.format( len(acc), len(acc[0]), len(acc[74]) ))
	print(' len(vel), len(vel[0]), len(vel[74]) = {}, {}, {} '.format( len(vel), len(vel[0]), len(vel[74]) ))
	print(' len(dis), len(dis[0]), len(dis[74]) = {}, {}, {} '.format( len(dis), len(dis[0]), len(dis[74]) ))
	data = {}
	data['acc'] = acc
	data['vel'] = vel
	data['dis'] = dis
	return data

# inputdata = getInputAccVelDis()

def getOutputAccDis():
	acc, dis = [], []
	for i in range(1,76):
		acc_filename = '../processed_data/shell_structure_{}_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i)
		dis_filename = '../processed_data/shell_structure_{}_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i)
		acc.append(np.loadtxt(acc_filename))
		dis.append(np.loadtxt(dis_filename))
	print(' len(acc), len(acc[0]), len(acc[74]) = {}, {}, {} '.format( len(acc), len(acc[0]), len(acc[74]) ))
	print(' len(dis), len(dis[0]), len(dis[74]) = {}, {}, {} '.format( len(dis), len(dis[0]), len(dis[74]) ))
	data = {}
	data['acc'] = acc
	data['dis'] = dis
	return data

# outputdata = getOutputAccDis()

def getData():
	inputdata = getInputAccVelDis()
	outputdata = getOutputAccDis()
	
	


