import numpy as np
import matplotlib.pyplot as plt

# damping_len = 6
# def getInputAccVelDis():
# 	# ene -> energy
# 	acc, vel, dis, frq = [], [], [], []
# 	for i in [0,2,5,7,8,12,26,34,51,53,64,66,69,70]:
# 		acc_filename = 'processed_data/acc_{}_freq_ampl.txt'.format(i)

# acc0 = np.loadtxt('processed_data/acc_')

i = 0
j = 0
acc_filename1 = 'processed_data4/shell_structure_{}_{}_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i,j)
# acc_filename1 = 'processed_data3/shell_structure_{}_{}_imposed_motion_node_38_x_acce.txt'.format(i,j)
j = 4
acc_filename2 = 'processed_data4/shell_structure_{}_{}_imposed_motion_node_38_x_acce_freq_ampl.txt'.format(i,j)
# acc_filename2 = 'processed_data3/shell_structure_{}_{}_imposed_motion_node_38_x_acce.txt'.format(i,j)
acc1 = np.loadtxt(acc_filename1)
acc2 = np.loadtxt(acc_filename2)

sz = min(len(acc1[:,0]), len(acc2[:,0]))
for i in range(sz):
	v = min(acc1[i,1], acc2[i,1])
	acc1[i,1] = acc1[i,1] - v * 0.9
	acc2[i,1] = acc2[i,1] - v * 0.9

plt.plot(acc1[:,0], acc1[:,1])
plt.plot(acc2[:,0], acc2[:,1])

plt.show()

