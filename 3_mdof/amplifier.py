
import numpy as np

damping_len = 6 - 1
for i in [0,2,5,7,8,12,26,34,51,53,64,66,69,70]:
	min_acc_file = 'processed_data3/shell_structure_{}_5_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i)
	min_dis_file = 'processed_data3/shell_structure_{}_5_imposed_motion_node_38_x_disp_freq_ampl.txt'.format(i)
	min_acc = np.loadtxt(min_acc_file)[:,1]
	min_dis = np.loadtxt(min_dis_file)[:,1]
	sz = len(min_dis)
	for j in range(damping_len):
		acc_filename = 'processed_data3/shell_structure_{}_{}_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i,j)
		dis_filename = 'processed_data3/shell_structure_{}_{}_imposed_motion_node_38_x_disp_freq_ampl.txt'.format(i,j)
		acc = np.loadtxt(acc_filename)
		dis = np.loadtxt(dis_filename)
		acc_out = 'processed_data4/shell_structure_{}_{}_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i,j)
		dis_out = 'processed_data4/shell_structure_{}_{}_imposed_motion_node_38_x_disp_freq_ampl.txt'.format(i,j)
		for k in range(sz):
			acc[k,1] = acc[k,1] - min_acc[k] * 0.8
			dis[k,1] = dis[k,1] - min_dis[k] * 0.8
		np.savetxt(acc_out, acc)
		np.savetxt(dis_out, dis)
