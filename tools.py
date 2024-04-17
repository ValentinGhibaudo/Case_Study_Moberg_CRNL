import pandas as pd
import pycns
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import xarray as xr
from configuration import *
from scipy import signal
import physio
import matplotlib.dates as mdates
from pycns import CnsReader
import xmltodict
import scipy

def get_metadata(sub = None):
    """
    Inputs
        sub : str id of patient to get its metadata or None if all metadata. Default is None
    Ouputs 
        pd.DataFrame or pd.Series
    """
    path = base_folder / 'tab_base_neuromonito.xlsx'
    if sub is None:
        return pd.read_excel(path)
    else:
        return pd.read_excel(path).set_index('ID_pseudo').loc[sub,:]
    
def read_events(sub):
    file = base_folder / 'raw_data' / sub / 'Events.xml'
    with open(file, 'r', encoding='utf-8') as file: 
        my_xml = file.read() 
    my_dict = xmltodict.parse(my_xml)
    return pd.DataFrame(my_dict['Events']['Event'])

def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter', verbose = False, show = False, axis = -1):

    """
    IIR-Filter of signal
    -------------------
    Inputs : 
    - sig : 1D numpy vector
    - srate : sampling rate of the signal
    - lowcut : lowcut of the filter. Lowpass filter if lowcut is None and highcut is not None
    - highcut : highcut of the filter. Highpass filter if highcut is None and low is not None
    - order : N-th order of the filter (the more the order the more the slope of the filter)
    - ftype : Type of the IIR filter, could be butter or bessel
    - verbose : if True, will print information of type of filter and order (default is False)
    - show : if True, will show plot of frequency response of the filter (default is False)
    """

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=axis)

    if verbose:
        print(f'{ftype} iirfilter of {order}th-order')
        print(f'btype : {btype}')


    if show:
        w, h = signal.sosfreqz(sos,fs=srate)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig

def get_amp(sig, axis = -1):
    analytic_signal = signal.hilbert(sig, axis = axis)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def sliding_mean(sig, nwin, mode = 'same'):
    """
    Sliding mean
    ------
    Inputs =
    - sig : 1D np vector
    - nwin : N samples in the sliding window
    - mode : default = 'same' = size of the output (could be 'valid' or 'full', see doc scipy.signal.fftconvolve)
    Output =
    - smoothed_sig : signal smoothed
    """

    kernel = np.ones(nwin)/nwin
    smoothed_sig = signal.fftconvolve(sig, kernel , mode = mode)
    return smoothed_sig

def get_wsize(srate, lowest_freq , n_cycles=5):
    nperseg = ( n_cycles / lowest_freq) * srate
    return int(nperseg)

def spectre(sig, srate, lowest_freq, n_cycles = 5, nfft_factor = 1, axis = -1, scaling = 'spectrum', verbose = False):

    """
    Compute Power Spectral Density of the signal with Welch method

    -----------------
    Inputs =
    - sig : Nd array with time in last dim
    - srate : samping rate
    - lowest_freq : Lowest frequency of interest, window sizes will be automatically computed based on this freq and set min number of cycle in window
    - n_cycles : Minimum cycles of the lowest frequency in the window size (default = 5)
    - nfft_factor : Factor of zero-padding (default = 1)
    - verbose : if True, print informations about windows length (default = False)
    - scaling : 'spectrum' or 'density' (cf scipy.signal.welch) (default = 'scaling')

    Outputs = 
    - f : frequency vector
    - Pxx : Power Spectral Density vector (scaling = spectrum so unit = V**2)

    """

    nperseg = get_wsize(srate, lowest_freq, n_cycles)
    nfft = int(nperseg * nfft_factor)
    f, Pxx = signal.welch(sig, fs=srate, nperseg = nperseg , nfft = nfft, scaling=scaling, axis=axis)

    if verbose:
        n_windows = 2 * sig.size // nperseg
        print(f'nperseg : {nperseg}')
        print(f'sig size : {sig.size}')
        print(f'total cycles lowest freq : {int(sig.size / ((1 / lowest_freq)*srate))}')
        print(f'nwindows : {n_windows}')

    return f, Pxx

def get_rate_variablity(cycles, rate_bins, bin_size_min, colname_date, colname_time, units):
    times = cycles[colname_date].values

    start = times[0]
    stop = times[-1]
    delta = np.timedelta64(int(bin_size_min*60), 's')
    time_bins = np.arange(start, stop, delta)

    rate_dist = np.zeros((time_bins.size - 1, rate_bins.size - 1)) * np.nan
    rate = np.zeros(time_bins.size - 1) * np.nan
    rate_variability = np.zeros(time_bins.size - 1) * np.nan

    for i in range(time_bins.size - 1):

        t0, t1 = time_bins[i], time_bins[i+1]

        keep = (cycles[colname_date] > t0) & (cycles[colname_date] < t1)
        cycles_keep = cycles[keep]

        if cycles_keep.shape[0] < 2:
            continue

        d = np.diff(cycles_keep[colname_time].values)
        if units == 'Hz':
            r = 1 / d
        elif  units == 'bpm':
            r = 60 / d
        else:
            raise ValueError(f'bad units {units}')

        count, bins = np.histogram(r, bins=rate_bins, density=True)
        rate_dist[i, :] = count
        rate[i], rate_variability[i] = physio.compute_median_mad(r)

    results = dict(
    time_bins=time_bins,
    rate_bins=rate_bins,
    rate_dist=rate_dist,
    rate=rate,
    rate_variability=rate_variability,
    units=units,
    )
    return results


def plot_variability(results, ratio_saturation=4, ax=None, plot_type = '2d', color='red'):
    globals().update(results)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    if plot_type == '2d':
        
        im = ax.imshow(rate_dist.T, origin='lower', aspect='auto', interpolation='None',
                 extent=[mdates.date2num(time_bins[0]), mdates.date2num(time_bins[-1]),
                         rate_bins[0], rate_bins[-1]])
        ax.plot(mdates.date2num(time_bins[:-1]), rate, color=color)
        ax.set_ylabel(f'rate [{units}]')

        im.set_clim(0, np.nanmax(rate_dist) / ratio_saturation)
    
    elif plot_type == '1d':
        ax.plot(mdates.date2num(time_bins[:-1]), rate_variability, color=color)
        ax.set_ylabel(f'rate variability [{units}]')  
    return ax

def interpolate_samples(data, data_times, time_vector, kind = 'linear'):
    f = scipy.interpolate.interp1d(data_times, data, fill_value="extrapolate", kind = kind)
    xnew = time_vector
    ynew = f(xnew)
    return ynew

def complex_mw(time, n_cycles , freq, a= 1, m = 0): 
    """
    Create a complex morlet wavelet by multiplying a gaussian window to a complex sinewave of a given frequency
    
    ------------------------------
    a = amplitude of the wavelet
    time = time vector of the wavelet
    n_cycles = number of cycles in the wavelet
    freq = frequency of the wavelet
    m = 
    """
    s = n_cycles / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2)) # real gaussian window
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time) # complex sinusoidal signal
    cmw = GaussWin * complex_sinewave
    return cmw

def morlet_family(srate, f_start, f_stop, n_steps, n_cycles):
    """
    Create a family of morlet wavelets
    
    ------------------------------
    srate : sampling rate
    f_start : lowest frequency of the wavelet family
    f_stop : highest frequency of the wavelet family
    n_steps : number of frequencies from f_start to f_stop
    n_cycles : number of waves in the wavelet
    """
    tmw = np.arange(-5,5,1/srate)
    freqs = np.linspace(f_start,f_stop,n_steps) 
    mw_family = np.zeros((freqs.size, tmw.size), dtype = 'complex')
    for i, fi in enumerate(freqs):
        mw_family[i,:] = complex_mw(tmw, n_cycles = n_cycles, freq = fi)
    return freqs, mw_family

def morlet_power(sig, srate, f_start, f_stop, n_steps, n_cycles, amplitude_exponent=2):
    """
    Compute time-frequency matrix by convoluting wavelets on a signal
    
    ------------------------------
    Inputs =
    - sig : the signal (1D np vector)
    - srate : sampling rate
    - f_start : lowest frequency of the wavelet family
    - f_stop : highest frequency of the wavelet family
    - n_steps : number of frequencies from f_start to f_stop
    - n_cycles : number of waves in the wavelet
    - amplitude_exponent : amplitude values extracted from the length of the complex vector will be raised to this exponent factor (default = 2 = V**2 as unit)

    Outputs = 
    - freqs : frequency 1D np vector
    - power : 2D np array , axis 0 = freq, axis 1 = time

    """
    freqs, family = morlet_family(srate, f_start = f_start, f_stop = f_stop, n_steps = n_steps, n_cycles = n_cycles)
    sigs = np.tile(sig, (n_steps,1))
    tf = signal.fftconvolve(sigs, family, mode = 'same', axes = 1)
    power = np.abs(tf) ** amplitude_exponent
    return freqs , power