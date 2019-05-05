import cmath
import numpy as np


# Part 1a: Slow DFT
def slow_dft(input_x):
    N = len(input_x)
    out = []
    for k in range(N):
        z = complex(0)
        for n in range(N):
            exponent = 2j * cmath.pi * k * n / N
            z += input_x[n] * cmath.exp(-exponent)
        out.append(z)
    return out


# Part 1b: Slow IDFT
def slow_idft(input_X):
    N = len(input_X)
    out = []
    for k in range(N):
        z = complex(0)
        for n in range(N):
            exponent = 2j * cmath.pi * k * n / N
            z += input_X[n] * cmath.exp(exponent)
        out.append(1 / N * z)
    return out


# Part 2a: FFT
def recursive_fft(input_x):
    x = np.asarray(input_x, dtype=complex)
    N = x.shape[0]

    if N is 1:
        return x
    else:
        even = recursive_fft(x[::2])
        odd = recursive_fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        # combine two halves
        return np.concatenate([even + factor[:N // 2] * odd, even + factor[N // 2:] * odd])


# Part 2b: IFFT
def recursive_sub_ifft(input_x):
    x = np.asarray(input_x, dtype=complex)
    N = x.shape[0]

    if N is 1:
        return x
    else:
        even = recursive_sub_ifft(x[::2])
        odd = recursive_sub_ifft(x[1::2])
        # numpy arange(N) gets the list of N
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        # combine two halves using numpy to combine them as it iterates through the halves
        return np.concatenate([even + factor[:N // 2] * odd, even + factor[N // 2:] * odd])


def ifft(input_x):
    return recursive_sub_ifft(input_x) / len(input_x)


vect = [0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071]
# vect = [0, -4j, 0, 0, 0, 0, 0, 4j]
print(recursive_fft(vect))
print(np.allclose(recursive_fft(vect), np.fft.fft(vect)))
print(np.fft.fft(vect))
#print(np.fft.ifft(vect))

x = np.random.random(1024)
# print(np.allclose(slow_dft(x), np.fft.fft(x)))
# print(np.allclose(slow_idft(x), np.fft.ifft(x)))
print(np.allclose(recursive_fft(x), np.fft.fft(x)))
