"""Scripts that contains the functions to apply filters."""

from scipy.fft import fft, fftfreq
from scipy.fftpack import fftshift
from scipy.signal import filtfilt, firwin, iirfilter, kaiserord, lfilter


def fourier_transform(signal, sample_rate=44100, duration=5):
    """Creation of the Fourier Transform signal."""
    # Number of samples in normalized_tone
    N = sample_rate * duration
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)
    # Ordered FFT
    yf = fftshift(yf)
    xf = fftshift(xf)
    return xf, yf


def iir_filter(signal, f_cutoff, f_sampling, fbf=False):
    """IIR filter applied to a signal."""
    b, a = iirfilter(4, Wn=f_cutoff, fs=f_sampling, btype="low", ftype="butter")
    if not fbf:
        filtered = lfilter(b, a, signal)
    else:
        filtered = filtfilt(b, a, signal)
    return filtered


def fir_filter(signal, nyq_rate, cutoff_hz):
    """FIR filter applied to a signal."""
    # ------------------------------------------------
    # Create a FIR filter and apply it to x.
    # ------------------------------------------------
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate. We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0 / nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 20.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=("kaiser", beta))
    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, signal)

    return filtered_x, taps, N
