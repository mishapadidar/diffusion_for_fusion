import numpy as np
from scipy.fft import fft, fftfreq, ifft

def fourier_interp1d_regular_grid(fx, m):
    """
    Interpolate a periodic function onto a uniform grid. This function uses
    the fft and ifft to accelerate the interpolant to O(nlogn + mlogm) for 
    n interpolation knots and m evaluation points. f can be a multi-output
    function.
    
    Args:
        fx (array): (n,) or (n,d) array of evaluations of the function for a function
        with d outputs. Evaluations should be taken on a uniform grid over the period,
        exclusive of the right endpoint, i.e.
            x = linspace(a, b, n, endpoint=False)
            fx = f(x)
        m (int): Number of points to evaluate the interpolant at. Must be greater than
            or equal to n.
        
    Returns:
        interpolated_signal: (m,d) array of interpolated values.
    """
    ndim = fx.ndim
    if ndim == 1:
        fx = fx.reshape((-1,1))
    
    n, d = fx.shape
    pad = m - n
    if pad < 0:
        raise ValueError("m must be >= the number of interpolation points.")

    fx_interp = np.zeros((m, d))
    for ii in range(d):
        # compute FFT
        coeffs = fft(fx[:,ii])
        
        # split and pad the spectrum
        half_len = n // 2
        padded_coeffs = np.concatenate([coeffs[:half_len], np.zeros(pad), coeffs[half_len:]])
        
        # scale factor
        scale_factor = m / n
        
        # ifft with scaling
        interpolated_signal = ifft(padded_coeffs) * scale_factor
        fx_interp[:,ii] = np.real(interpolated_signal)

    if ndim == 1:
        fx_interp = fx_interp.flatten()

    return fx_interp