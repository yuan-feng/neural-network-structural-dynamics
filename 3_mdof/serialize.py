


import numpy as np 
import scipy.fftpack

a = [1,2,3,4,5]
fft  = scipy.fftpack.fft(a);

sz = len(fft)
x = np.linspace(0,1,sz)
print('x = {}'.format(x))
ss = np.stack([x,fft]).T
np.savetxt('complex.txt', ss)


vals = np.loadtxt('complex.txt',dtype=np.complex_)

print( 'vals col0 = {}'.format( vals[:,0] ) )
print( 'vals col1 = {}'.format( vals[:,1] ) )




