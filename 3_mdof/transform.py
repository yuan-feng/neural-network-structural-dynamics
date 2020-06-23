# *************************************************************************************
# FFT function
# *************************************************************************************
import scipy.fftpack
import math
def FFT(x,dt):
	N = x.size;
	T = N*dt;
	Fs = 1./dt;
	xfft  = scipy.fftpack.fft(x);
	xfreq = np.linspace(0.0, Fs/2, N/2);
	xfftHalf = 2.0/N * np.abs(xfft[:N//2]);
	xfftHalf[0] = xfftHalf[0]/2;
	return xfreq, xfftHalf, xfft * 2.0/N

def IFFT(f, N):
	frq = f * N / 2.0
	y = scipy.fftpack.ifft(frq)
	return np.abs(y)

import numpy as np
N = 20
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

freq, ampl, full_fft = FFT(y, T)

y_hat = scipy.fftpack.ifft(scipy.fftpack.fft(y))

frqs = np.fft.fftfreq(N, d=T)
full_fft = np.abs(full_fft)
# y_hat = IFFT(full_fft, N)

print(' y = {}'.format( y ) )
print(' y_hat = {}'.format( y_hat.real ) )
print(' y_hat = {}'.format( y_hat.imag ) )
print(' frqs = {}'.format( frqs ) )
print(' len(frqs) = {}'.format( len(frqs) ) )
print(' len(y_hat) = {}'.format( len(y_hat) ) )
# print(' len(ampl) = {} '.format(len(ampl)))
# print(' len(full_fft) = {} '.format(len(full_fft)))
print(' ampl = {}'.format(ampl))
print(' full_fft abs = {}'.format(full_fft) )

import matplotlib.pyplot as plt 
plt.semilogx(freq, ampl, '-k', linewidth=3)

plt.show()


# def IFFT

