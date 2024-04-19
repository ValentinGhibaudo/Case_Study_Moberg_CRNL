import numpy as np
from tools import *
import scipy
import physio

class ECG_Detections:
    name = 'ECG_Detections'
    
    def __init__(self, stream, ecg_features):
        self.stream = stream
        self.ecg_features = ecg_features
        
    def plot(self, ax, t0, t1):
        sig, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True)
        
        if not sig is None:
            ecg_features = self.ecg_features
            local_peaks = ecg_features[(ecg_features['peak_date'] > t0) & (ecg_features['peak_date'] < t1)]

            local_peak_dates = local_peaks['peak_date'].values
            local_peak_inds = np.searchsorted(dates, local_peak_dates)

            ax.plot(dates, sig, color='k')
            ax.scatter(local_peak_dates, sig[local_peak_inds], color='m')
        else:
            ax.plot()
        
        
class Resp_Detections:
    name = 'Resp_Detections'
    
    def __init__(self, stream, resp_features):
        self.stream = stream
        self.resp_features = resp_features
        
    def plot(self, ax, t0, t1):
        sig, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True)
        
        if not sig is None:
            resp_features = self.resp_features
            local_resp_features = resp_features[(resp_features['inspi_date'] > t0) & (resp_features['expi_date'] < t1)]

            local_inspi_dates = local_resp_features['inspi_date'].values
            local_expi_dates = local_resp_features['expi_date'].values
            local_inspi_inds= np.searchsorted(dates, local_inspi_dates)
            local_expi_inds = np.searchsorted(dates, local_expi_dates)

            ax.plot(dates, sig, color='k')
            ax.scatter(local_inspi_dates, sig[local_inspi_inds], color='g')
            ax.scatter(local_expi_dates, sig[local_expi_inds], color='r')
        else:
            ax.plot()

class Monopolar:
    name = 'Monopolar'

    def __init__(self, stream, chan, down_sampling_factor = 1, lowcut = None, highcut = None, centering = True):
        self.stream = stream
        self.chan_name = chan
        self.chan_ind = stream.channel_names.index(chan)
        self.down_sampling_factor = down_sampling_factor
        self.lowcut = lowcut
        self.highcut = highcut
        self.centering = centering       

    def plot(self, ax, t0, t1):
        sigs, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        srate = self.stream.sample_rate
        sig = sigs[:, self.chan_ind]
        if 'ECoG' in self.chan_name:
            sig = sig / 1000 # µV to mV
            unit = 'mV'
        else:
            unit = 'µV'
        if self.centering:
            sig = sig - np.mean(sig)
            # sig = sig - sig[0]
        if not self.lowcut is None or not self.highcut is None:
            sig = iirfilt(sig, srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel')
        sig = scipy.signal.decimate(sig, q=self.down_sampling_factor)
        dates = dates[::self.down_sampling_factor]
        
        ax.plot(dates, sig, color='r', lw = 0.8)
        ax.set_ylim(sig.min(), sig.max())
        ax.set_ylabel(f'{self.chan_name}\n[{unit}]')        

class Bipolar:
    name = 'Bipolar'

    def __init__(self, stream, chan1, chan2, down_sampling_factor = 1, lowcut = None, highcut = None, centering = True):
        self.stream = stream
        self.chan1_name = chan1
        self.chan2_name = chan2
        if isinstance(chan1, str):
            chan1 = stream.channel_names.index(chan1)
        if isinstance(chan2, str):
            chan2 = stream.channel_names.index(chan2)
        self.ind1 = chan1
        self.ind2 = chan2
        self.down_sampling_factor = down_sampling_factor
        self.lowcut = lowcut
        self.highcut = highcut
        self.centering = centering
       
    def plot(self, ax, t0, t1):
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        srate = self.stream.sample_rate
        sig = sigs[:, self.ind2] - sigs[:, self.ind1]
        # if 'ECoG' in self.chan_name1:
        sig = sig / 1000 # µV to mV
        unit = 'mV'

        if self.centering:
            sig = sig - np.mean(sig)
            # sig = sig - sig[0]
        if not self.lowcut is None or not self.highcut is None:
            sig = iirfilt(sig, srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel')
        sig = scipy.signal.decimate(sig, q=self.down_sampling_factor)
        times = times[::self.down_sampling_factor]
        
        ax.plot(times, sig, color='r', lw = 0.8)
        ax.set_ylim(sig.min(), sig.max())
        ax.set_ylabel(f'{self.chan1_name}\n-\n{self.chan2_name}\n[{unit}]')
           
class AC_Envelope:
    name = 'AC Amp'

    def __init__(self, stream, chan_name, lowcut_ac = 0.5, highcut_ac = 40, highcut_amp = 0.01, down_sampling_factor = 100):
        self.stream = stream
        self.lowcut_ac = lowcut_ac
        self.highcut_ac = highcut_ac
        self.highcut_amp = highcut_amp
        self.srate = stream.sample_rate
        self.chan_name = chan_name
        self.down_sampling_factor = down_sampling_factor
        
    def plot(self, ax, t0, t1):
        stream = self.stream
        sigs, times = stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        chan_name = self.chan_name
        if not '-' in chan_name: # monopolar case
            sig = sigs[:,stream.channel_names.index(chan_name)]
        else:
            chan_name1, chan_name2 = chan_name.split('-')
            sig = sigs[:,stream.channel_names.index(chan_name1)] - sigs[:,stream.channel_names.index(chan_name2)]
        
        sig = sig / 1000 # µV to mV
        ac = iirfilt(sig, self.srate, lowcut = self.lowcut_ac, highcut = self.highcut_ac, ftype = 'bessel')
        ac_envelope = get_amp(ac)
        ac_env_smoothed = iirfilt(ac_envelope, self.srate, lowcut = None, highcut = self.highcut_amp)
        # ac_env_smoothed = scipy.signal.decimate(ac_env_smoothed,q=self.down_sampling_factor)
        ac_env_smoothed = ac_env_smoothed[::self.down_sampling_factor]
        # ac_env_smoothed = ac_env_smoothed - np.mean(ac_env_smoothed)
        times = times[::self.down_sampling_factor]
        ax.plot(times, ac_env_smoothed, color='k', lw = 0.5)
        # ax.set_ylabel('Amplitude (mV)')
        ax.set_ylim(ac_env_smoothed.min(), ac_env_smoothed.max())
        ax.set_ylabel(f'AC Amp\n{self.chan_name}\n[mV]')
        
class Spectrogram_eeg:
    name = 'Spectro eeg'

    def __init__(self, stream, chan_name, wsize, lf=None, hf=None, scaling = 'log', saturation_quantile = None, cmap = 'viridis'):
        self.stream = stream
        self.wsize = wsize
        self.chan_name = chan_name
        self.lf = lf
        self.hf = hf
        self.saturation_quantile = saturation_quantile
        self.scaling = scaling
        self.cmap = cmap
        
    def plot(self, ax, t0, t1):
        stream = self.stream
        srate = stream.sample_rate
        chan_name = self.chan_name
        if not '-' in chan_name:
            chan_ind = stream.channel_names.index(chan_name)
            sigs, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                              apply_gain=True)
            sig = sigs[:,chan_ind]
        else:
            chan1, chan2 = chan_name.split('-')
            chan_ind1, chan_ind2 = stream.channel_names.index(chan1), stream.channel_names.index(chan2)
            sigs, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                              apply_gain=True)
            sig = sigs[:,chan_ind1] -  sigs[:,chan_ind2]     
        
        lf = self.lf
        hf = self.hf
        
        down_sample_factor = int(srate / (hf * 3))
        if down_sample_factor >=1:
            sig = scipy.signal.decimate(sig, q = down_sample_factor)
            srate /= down_sample_factor
        freqs, times_spectrum_s, Sxx = scipy.signal.spectrogram(sig, fs = srate, nperseg = int(self.wsize * srate))

        times_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + dates[0]

        if lf is None and hf is None:
            f_mask = (freqs>=freqs[0]) and (freqs<=freqs[-1])
        elif lf is None and not hf is None:
            f_mask = (freqs<=hf)
        elif not lf is None and hf is None:
            f_mask = (freqs>=lf)
        else:
            f_mask = (freqs>=lf) & (freqs<=hf)
        
        relative_cumsum = np.cumsum(Sxx, axis = 0) / np.sum(Sxx, axis = 0)
        median_freq = np.apply_along_axis(lambda x:freqs[np.searchsorted(x, 0.5)], arr = relative_cumsum, axis = 0)
        spectral_edge_freq = np.apply_along_axis(lambda x:freqs[np.searchsorted(x, 0.95)], arr = relative_cumsum, axis = 0)
        
        data = Sxx[f_mask,:]
        
        assert self.scaling in ['log','dB',None], f'{self.scaling} is not a valid parameter (log or dB or None)'
        if self.scaling == 'log':
            data = np.log(data)
        elif self.scaling == 'dB':
            data = 10 * np.log10(data)
        elif self.scaling is None:
            data = data
            
        if not self.saturation_quantile is None:
            vmin = np.quantile(data, self.saturation_quantile)
            vmax = np.quantile(data, 1 - self.saturation_quantile)
            ax.pcolormesh(times_spectrum, freqs[f_mask], data, vmin=vmin , vmax=vmax, cmap = self.cmap)
        else:
            ax.pcolormesh(times_spectrum, freqs[f_mask], data, cmap = self.cmap)
        # ax.plot(times_spectrum, median_freq, color = 'k', ls = '--', alpha = 0.3)
        # ax.plot(times_spectrum, spectral_edge_freq, color = 'k', alpha = 0.3)
        ax.set_ylim(lf, hf)
        ax.set_ylabel(f'Spectro EEG\n{chan_name}\nFrequency (Hz)')
        # ax.set_yscale('log')
        # for i in [4,8,12]:
        #     ax.axhline(i, color = 'r')
        # ax.set_ylim(lf, hf)
    
class Spectrogram_bio:
    name = 'Spectro bio'

    def __init__(self, stream, wsize, lf=None, hf=None, log_transfo = False, saturation_quantile = None, overlap_prct = 0.5, nfft_factor = 1, power_or_amplitude = 'power'):
        self.stream = stream
        self.srate = stream.sample_rate
        self.wsize = wsize
        self.lf = lf
        self.hf = hf
        self.overlap_prct = overlap_prct
        self.nfft_factor = nfft_factor
        self.log_transfo = log_transfo
        self.saturation_quantile = saturation_quantile
        self.power_or_amplitude = power_or_amplitude
        
    def plot(self, ax, t0, t1):
        lf = self.lf
        hf = self.hf
        
        times = self.stream.get_times()
        
        i0 = np.searchsorted(times, np.datetime64(t0))
        i1 = np.searchsorted(times, np.datetime64(t1))
        
        sig, times = self.stream.get_data(isel=slice(i0, i1), with_times=True,
                                          apply_gain=True)
        
        if sig.shape[0] == 0:
            ax.plot(times, sig, color='k')
        else:
            nperseg = int(self.wsize * self.srate)
            noverlap = int(nperseg * self.overlap_prct)
            nfft = int(nperseg * self.nfft_factor)
            freqs, times_spectrum_s, Sxx = scipy.signal.spectrogram(sig, fs = self.srate, nperseg =  nperseg, noverlap = noverlap, nfft = nfft)
            if self.power_or_amplitude == 'amplitude': 
                Sxx = np.sqrt(Sxx)
            times_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + times[0]
            
            if lf is None and hf is None:
                f_mask = (freqs>=freqs[0]) and (freqs<=freqs[-1])
            elif lf is None and not hf is None:
                f_mask = (freqs<=hf)
            elif not lf is None and hf is None:
                f_mask = (freqs>=lf)
            else:
                f_mask = (freqs>=lf) & (freqs<=hf)

            if self.log_transfo:
                data = np.log(Sxx[f_mask,:])
            else:
                data = Sxx[f_mask,:]

            if not self.saturation_quantile is None:
                vmin = np.quantile(data, self.saturation_quantile)
                vmax = np.quantile(data, 1 - self.saturation_quantile)
                ax.pcolormesh(times_spectrum, freqs[f_mask], data, vmin=vmin , vmax=vmax)
            else:
                ax.pcolormesh(times_spectrum, freqs[f_mask], data)
            
class Respi_Rate:
    name = 'Respi_Rate'

    def __init__(self, resp_features, rate_bins_resp = np.arange(5, 30, 0.5), resp_wsize_in_mins = 4, ratio_sat = 4, units = 'bpm'):
        self.rate_bins_resp = rate_bins_resp
        self.resp_wsize_in_mins = resp_wsize_in_mins
        self.ratio_sat = ratio_sat
        self.resp_features = resp_features
        self.units = units
        
    def plot(self, ax, t0, t1):

        resp_features = self.resp_features
        local_resp_features = resp_features[(resp_features['inspi_date'] > t0) & (resp_features['inspi_date'] < t1)]
        
        if not local_resp_features.shape[0] == 0:
            res = get_rate_variablity(cycles = local_resp_features, 
                                      rate_bins = self.rate_bins_resp, 
                                      bin_size_min = self.resp_wsize_in_mins, 
                                      colname_date = 'inspi_date', 
                                      colname_time = 'inspi_time', 
                                      units = self.units
                                     )

            plot_variability(res, ax=ax, ratio_saturation = self.ratio_sat, plot_type = '2d')
            ax.set_ylim(self.rate_bins_resp[0], self.rate_bins_resp[-1])
            ax.set_ylabel(f'Respi\nrate\n[{self.units}]')
            
class Heart_Rate:
    name = 'Heart_Rate'

    def __init__(self, ecg_peaks, step_bins_ecg = 2,  hrv_wsize_in_mins = 2, ratio_sat = 4, plot_type='2d'):
        self.step_bins_ecg = step_bins_ecg
        self.hrv_wsize_in_mins = hrv_wsize_in_mins
        self.ratio_sat = ratio_sat
        self.ecg_peaks = ecg_peaks
        self.plot_type = plot_type
        
    def plot(self, ax, t0, t1):
        
        ecg_peaks = self.ecg_peaks
        local_peaks = ecg_peaks[(ecg_peaks['peak_date'] > t0) & (ecg_peaks['peak_date'] < t1)]
        rri = 60 / np.diff(local_peaks['peak_time'].values)
        min_rate = np.quantile(rri, 0.001) - 5
        max_rate = np.quantile(rri, 0.999) + 5
        
        rate_bins_ecg = np.arange(min_rate, max_rate, self.step_bins_ecg)
        
        if not local_peaks.shape[0] == 0:
            res = get_rate_variablity(cycles = local_peaks, 
                                      rate_bins = rate_bins_ecg, 
                                      bin_size_min = self.hrv_wsize_in_mins, 
                                      colname_date = 'peak_date', 
                                      colname_time = 'peak_time', 
                                      units = 'bpm')
            

            plot_variability(res, ax=ax, ratio_saturation = self.ratio_sat, plot_type = self.plot_type)
            if self.plot_type == '2d':
                ax.set_ylim(min_rate, max_rate)
            elif self.plot_type == '1d':
                # ax.set_ylim(0, np.quantile(res['rate_variability'], 0.999))
                ax.set_ylim(0, 10)
            ax.set_ylabel(f'Heart\nrate\n[bpm]')
        
class Spreading_depol_mono:
    name = 'Spreading_depol_mono'

    def __init__(self, stream, detections = None, down_sampling_factor = 5, lowcut_dc = 0.0008):
        self.stream = stream
        self.down_sampling_factor = down_sampling_factor
        self.lowcut_dc = lowcut_dc
        self.detections = detections

    def plot(self, ax, t0, t1):
        chans = self.stream.channel_names
        srate = self.stream.sample_rate
        ecog_chans = [chan for chan in chans if 'ECoG' in chan]
        ecog_chan_inds = [chans.index(chan) for chan in ecog_chans]
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        times = times[::self.down_sampling_factor]
        sigs = sigs / 1000 # µV to mV
        new_srate = self.stream.sample_rate / self.down_sampling_factor
        
        raw_sigs = sigs[:,ecog_chan_inds]
        dcs = iirfilt(raw_sigs, srate, lowcut = self.lowcut_dc, highcut = 0.1, ftype = 'bessel', axis = 0)
        acs = iirfilt(raw_sigs, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel', axis = 0)
        
        meds, mads = physio.compute_median_mad(dcs, axis = 0)
        gains_dcs = mads / np.max(mads)
        
        meds, mads = physio.compute_median_mad(acs, axis = 0)
        gains_acs = mads / np.max(mads)
        
        jitters = []
        mins = []
        maxs = []
        for i, chan in enumerate(ecog_chans):
            dc = dcs[:,i]
            ac = acs[:,i]

            dc = (dc - np.mean(dc)) / np.std(dc)
            ac = (ac - np.mean(ac)) / np.std(ac)

            dc = scipy.signal.decimate(dc, q=self.down_sampling_factor)
            ac = scipy.signal.decimate(ac, q = self.down_sampling_factor)
            
            gain_ac = gains_acs[i] * 2
            gain_dc = gains_dcs[i] * 2
            
            jitter = i * 20
            ac_plot = ac * gain_ac + jitter
            dc_plot = dc * gain_dc  + jitter
            mins.append(ac_plot.min())
            maxs.append(ac_plot.max())
            ax.plot(times, ac_plot, color = 'k', lw = 0.5, alpha = 0.9)
            ax.plot(times, dc_plot, color = 'r', lw = 1)
            jitters.append(jitter)
        
        ax.set_yticks(jitters, labels=ecog_chans)
        ax.set_ylim(np.min(mins) - 5, np.max(maxs) + 5)
        ax.set_ylabel(f'ECoG\n[mV]')
        
        detections = self.detections
        if not detections is None:
            detections = detections[detections['Name'].apply(lambda x:'SD' in x)]
            detections = detections.sort_values(by = 'StartTimeGMT')
            local_detections = detections[(detections['StartTimeGMT'].values > t0) & (detections['StartTimeGMT'].values < t1)]
            if local_detections.shape[0] > 0:
                for i, row in local_detections.iterrows():
                    start = row['StartTimeGMT']
                    duration = float(row['Duration'])
                    stop = start + pd.Timedelta(duration, 's')
                    ax.axvspan(start, stop, color = 'k', alpha = 0.05)
        
class Spreading_depol_bipol:
    name = 'ECoG Bipolar'

    def __init__(self, stream, detections = None, down_sampling_factor = 5, lowcut_dc = 0.0008):
        self.stream = stream
        self.down_sampling_factor = down_sampling_factor
        self.lowcut_dc = lowcut_dc
        self.detections = detections

    def plot(self, ax, t0, t1):
        chans = self.stream.channel_names
        srate = self.stream.sample_rate
        ecog_chan_names = [chan for chan in chans if 'ECoG' in chan]
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        times = times[::self.down_sampling_factor]
        sigs = sigs / 1000 # µV to mV
        new_srate = self.stream.sample_rate / self.down_sampling_factor
        
        bipol_ecog_chan_names = [f'{ecog_chan_names[i]}-{ecog_chan_names[i-1]}' for i in np.arange(len(ecog_chan_names)-1,0,-1)]
        
        raw_sigs = np.zeros((sigs.shape[0], len(bipol_ecog_chan_names)))
        for i, bipol_chan in enumerate(bipol_ecog_chan_names):
            chan1, chan2 = bipol_chan.split('-')
            chan1_ind, chan2_ind = chans.index(chan1), chans.index(chan2)
            raw_sigs[:,i] =  sigs[:,chan1_ind] -  sigs[:,chan2_ind]
        
        dcs = iirfilt(raw_sigs, srate, lowcut = self.lowcut_dc, highcut = 0.1, ftype = 'bessel', axis = 0)
        acs = iirfilt(raw_sigs, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel', axis = 0)
        
        meds, mads = physio.compute_median_mad(dcs, axis = 0)
        gains_dcs = mads / np.max(mads)
        
        meds, mads = physio.compute_median_mad(acs, axis = 0)
        gains_acs = mads / np.max(mads)
        
        jitters = []
        mins = []
        maxs = []
        names = []
        for i, chan in enumerate(bipol_ecog_chan_names):
            chan1, chan2 = chan.split('-')
            names.append(f'{chan1[-1]}-{chan2[-1]}')
            
            dc = dcs[:,i]
            ac = acs[:,i]

            dc = (dc - np.mean(dc)) / np.std(dc)
            ac = (ac - np.mean(ac)) / np.std(ac)

            dc = scipy.signal.decimate(dc, q=self.down_sampling_factor)
            ac = scipy.signal.decimate(ac, q = self.down_sampling_factor)
            
            gain_ac = gains_acs[i] * 1
            gain_dc = gains_dcs[i] * 1
            
            jitter = i * 20
            ac_plot = ac * gain_ac + jitter
            dc_plot = dc * gain_dc  + jitter
            # mins.append(ac_plot.min())
            # maxs.append(ac_plot.max())
            mins.append(np.quantile(ac_plot, q = 0.001))
            maxs.append(np.quantile(ac_plot, q = 0.999))
            ax.plot(times, ac_plot, color = 'k', lw = 0.5, alpha = 0.9)
            ax.plot(times, dc_plot, color = 'r', lw = 1)
            jitters.append(jitter)
        
        ax.set_yticks(jitters, labels=names)
        ax.set_ylim(np.min(mins) - 5, np.max(maxs) + 5)
        ax.set_ylabel(f'ECoG\n[mV]')
        
        detections = self.detections
        if not detections is None:
            detections = detections[(detections['Name'].apply(lambda x:'SD' in x)) | (detections['Name'].apply(lambda x:'NUP' in x))]
            detections = detections.sort_values(by = 'StartTimeGMT')
            local_detections = detections[(detections['StartTimeGMT'].values > t0) & (detections['StartTimeGMT'].values < t1)]
            if local_detections.shape[0] > 0:
                for i, row in local_detections.iterrows():
                    start = row['StartTimeGMT']
                    duration = float(row['Duration'])
                    stop = start + pd.Timedelta(duration, 's')
                    ax.axvspan(start, stop, color = 'k', alpha = 0.05)
                    
class Spreading_depol_mono2:
    name = 'Spreading_depol_mono'

    def __init__(self, stream, detections = None, down_sampling_factor = 5, lowcut_dc = 0.0008):
        self.stream = stream
        self.down_sampling_factor = down_sampling_factor
        self.lowcut_dc = lowcut_dc
        self.detections = detections

    def plot(self, ax, t0, t1):
        chans = self.stream.channel_names
        srate = self.stream.sample_rate
        ecog_chans = [chan for chan in chans if 'ECoG' in chan]
        ecog_chan_inds = [chans.index(chan) for chan in ecog_chans]
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        times = times[::self.down_sampling_factor]
        sigs = sigs / 1000 # µV to mV
        new_srate = self.stream.sample_rate / self.down_sampling_factor
        
        raw_sigs = sigs[:,ecog_chan_inds]
        dcs = iirfilt(raw_sigs, srate, lowcut = self.lowcut_dc, highcut = 0.1, ftype = 'bessel', axis = 0)
        acs = iirfilt(raw_sigs, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel', axis = 0)
        
        meds, mads = physio.compute_median_mad(dcs, axis = 0)
        gains_dcs = mads / np.max(mads)
        
        meds, mads = physio.compute_median_mad(acs, axis = 0)
        gains_acs = mads / np.max(mads)
        
        jitters = []
        for i, chan in enumerate(ecog_chans):
            dc = dcs[:,i]
            ac = acs[:,i]

            dc = (dc - np.mean(dc)) / np.std(dc)
            ac = (ac - np.mean(ac)) / np.std(ac)

            dc = scipy.signal.decimate(dc, q=self.down_sampling_factor)
            ac = scipy.signal.decimate(ac, q = self.down_sampling_factor)
  
            jitter = i * 20
            ac_plot = ac + jitter
            dc_plot = dc  + jitter
            ax.plot(times, ac_plot, color = 'k', lw = 0.5, alpha = 0.9)
            ax.plot(times, dc_plot, color = 'r', lw = 1)
            jitters.append(jitter)
        
        ax.set_yticks(jitters, labels=ecog_chans)
        ax.set_ylim(np.min(jitters), np.max(jitters))
        ax.set_ylabel(f'ECoG\n[mV]')
        
        detections = self.detections
        if not detections is None:
            detections = detections[detections['Name'].apply(lambda x:'SD' in x)]
            detections = detections.sort_values(by = 'StartTimeGMT')
            local_detections = detections[(detections['StartTimeGMT'].values > t0) & (detections['StartTimeGMT'].values < t1)]
            if local_detections.shape[0] > 0:
                for i, row in local_detections.iterrows():
                    start = row['StartTimeGMT']
                    duration = float(row['Duration'])
                    stop = start + pd.Timedelta(duration, 's')
                    ax.axvspan(start, stop, color = 'k', alpha = 0.05)
        
class Spreading_depol_bipol2:
    name = 'ECoG Bipolar 2'

    def __init__(self, stream, detections = None, down_sampling_factor = 5, lowcut_dc = 0.0008):
        self.stream = stream
        self.down_sampling_factor = down_sampling_factor
        self.lowcut_dc = lowcut_dc
        self.detections = detections

    def plot(self, ax, t0, t1):
        chans = self.stream.channel_names
        srate = self.stream.sample_rate
        ecog_chan_names = [chan for chan in chans if 'ECoG' in chan]
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        times = times[::self.down_sampling_factor]
        sigs = sigs / 1000 # µV to mV
        new_srate = self.stream.sample_rate / self.down_sampling_factor
        
        bipol_ecog_chan_names = [f'{ecog_chan_names[i]}-{ecog_chan_names[i-1]}' for i in np.arange(len(ecog_chan_names)-1,0,-1)]
        jitters = []
        names = []
        mins = []
        maxs = []
        for i, bipol_chan in enumerate(bipol_ecog_chan_names):
            chan1, chan2 = bipol_chan.split('-')
            name = f'{chan1[-1]}-{chan2[-1]}'
            names.append(name)
            chan1_ind, chan2_ind = chans.index(chan1), chans.index(chan2)
            raw_sig =  sigs[:,chan1_ind] -  sigs[:,chan2_ind]
        
            dc = iirfilt(raw_sig, srate, lowcut = self.lowcut_dc, highcut = 0.1, ftype = 'bessel', axis = 0)
            ac = iirfilt(raw_sig, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel', axis = 0)

            dc = (dc - np.mean(dc)) / np.std(dc)
            ac = (ac - np.mean(ac)) / np.std(ac)

            dc = scipy.signal.decimate(dc, q = self.down_sampling_factor)
            ac = scipy.signal.decimate(ac, q = self.down_sampling_factor)

            jitter = i * 20
            ac_plot = ac + jitter
            dc_plot = dc + jitter

            ax.plot(times, ac_plot, color = 'k', lw = 0.5, alpha = 0.9)
            ax.plot(times, dc_plot, color = 'r', lw = 1)
            mins.append(np.min([np.min(ac_plot), np.min(dc_plot)]))
            maxs.append(np.max([np.max(ac_plot), np.max(dc_plot)]))
            jitters.append(jitter)
        
        ax.set_yticks(jitters, labels=names)
        ax.set_ylabel(f'ECoG_\n[mV]')
        ax.set_ylim(np.min(mins), np.max(maxs))
        
        detections = self.detections
        if not detections is None:
            detections = detections[detections['Name'].apply(lambda x:'SD' in x)]
            detections = detections.sort_values(by = 'StartTimeGMT')
            local_detections = detections[(detections['StartTimeGMT'].values > t0) & (detections['StartTimeGMT'].values < t1)]
            if local_detections.shape[0] > 0:
                for i, row in local_detections.iterrows():
                    start = row['StartTimeGMT']
                    duration = float(row['Duration'])
                    stop = start + pd.Timedelta(duration, 's')
                    ax.axvspan(start, stop, color = 'k', alpha = 0.05)
        
class Spreading_depol_scalp:
    name = 'Spreading_depol_scalp'

    def __init__(self, stream, down_sampling_factor = 5):
        self.stream = stream
        self.down_sampling_factor = down_sampling_factor

    def plot(self, ax, t0, t1):
        chans = self.stream.channel_names
        srate = self.stream.sample_rate
        scalp_chans = [chan for chan in chans if not 'ECoG' in chan]
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        times = times[::self.down_sampling_factor]
        sigs = sigs / 1000 # µV to mV
        new_srate = self.stream.sample_rate / self.down_sampling_factor
        
        jitters = []
        mins = []
        maxs = []
        for i, chan in enumerate(scalp_chans):
            chan_ind = chans.index(chan)
            raw_sig = sigs[:, chan_ind]
            dc = iirfilt(raw_sig, srate, lowcut = 0.0008, highcut = 0.1, ftype = 'bessel')
            ac = iirfilt(raw_sig, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel')
            dc = (dc - np.mean(dc)) / np.std(dc)
            ac = (ac - np.mean(ac)) / np.std(ac)
            dc = dc - dc[0]
            dc = scipy.signal.decimate(dc, q = self.down_sampling_factor)
            ac = scipy.signal.decimate(ac, q=self.down_sampling_factor)
            
            jitter = i * 20
            ac_plot = ac + jitter
            dc_plot = dc + jitter
            mins.append(ac_plot.min())
            maxs.append(ac_plot.max())
            ax.plot(times, ac_plot, color = 'k', lw = 0.5, alpha = 0.9)
            ax.plot(times, dc_plot, color = 'r', lw = 1)
            jitters.append(jitter)
        
        ax.set_yticks(jitters, labels=scalp_chans)
        ax.set_ylim(np.min(mins) - 5, np.max(maxs) + 5)
        ax.set_ylabel(f'Scalp EEG\n[mV]')
        
class Wavelet_Power:
    name = 'Wavelet_Power'

    def __init__(self, stream, chan,  f_start, f_stop, n_steps, n_cycles, amplitude_exponent=2, down_samp_compute = 2, down_samp_plot = 5, quantile_saturation = 0.01, scaling = None):
        self.stream = stream
        self.chan_name = chan
        self.chan_ind = stream.channel_names.index(chan)
        self.f_start = f_start
        self.f_stop = f_stop
        self.n_steps = n_steps
        self.n_cycles = n_cycles
        self.amplitude_exponent = amplitude_exponent
        self.down_samp_compute = down_samp_compute
        self.down_samp_plot = down_samp_plot
        self.quantile_saturation = quantile_saturation
        self.scaling = scaling

    def plot(self, ax, t0, t1):
        sigs, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        srate = self.stream.sample_rate
        sig = sigs[:, self.chan_ind]
        sig = scipy.signal.decimate(sig, q = self.down_samp_compute)
        dates = dates[::self.down_samp_compute]
        new_srate = srate / self.down_samp_compute
        padding_ones = np.ones(int(sig.size * 0.2))
        padding_before = padding_ones * sig[0]
        padding_after = padding_ones * sig[-1]
        sig_padded = np.concatenate([padding_before, sig, padding_after])
        f, power_padded = morlet_power(sig_padded, new_srate, self.f_start, self.f_stop, self.n_steps, self.n_cycles, self.amplitude_exponent)
        power = power_padded[:,padding_ones.size:sig_padded.size - padding_ones.size]
        power_plot = scipy.signal.decimate(power, q = self.down_samp_plot, axis = 1)
        if self.scaling == 'log':
            power_plot = np.log(power_plot)
        elif self.scaling == 'fit':
            f_log = np.log(f)
            spectrum = np.mean(power_plot, axis = 1)
            spectrum_log = np.log(spectrum)
            res = scipy.stats.linregress(f_log, spectrum_log)
            a = res.slope
            b = res.intercept
            fit_log = a * f_log + b
            fit = np.exp(a * f_log + b)
            power_plot = power_plot - fit[:,None]
        dates_plot = dates[::self.down_samp_plot]
        vmin = np.quantile(power, self.quantile_saturation)
        vmax = np.quantile(power, 1-self.quantile_saturation)
        ax.pcolormesh(dates_plot, f, power_plot, vmin = vmin, vmax=vmax)
        ax.set_ylabel(f'{self.chan_name}\nPower\n[µV]')  
        
        
class Cereral_Perfusion_Pressure:
    name = 'Cereral_Perfusion_Pressure'

    def __init__(self, icp_mean_stream, abp_mean_stream, down_sampling_factor=10):
        self.icp_mean_stream = icp_mean_stream
        self.abp_mean_stream = abp_mean_stream
        self.down_sampling_factor = down_sampling_factor
        
    def plot(self, ax, t0, t1):

        icp, dates_icp = self.icp_mean_stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        
        abp, dates_abp = self.abp_mean_stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        
        ymax = 200
        if icp.size == abp.size:
                
            cpp = abp - icp

            ax.plot(dates_abp, cpp, color='k', lw = 1)
            ax.set_ylabel(f'CPP (mmHg)')
            ax.set_ylim(0, ymax)
            
        else:
            ax.plot()
            # ax.text(x = dates_abp[dates_abp.size // 2], y = 100, s = 'ICP or ABP do not have the same shape', ha = 'center')
            ax.set_ylabel(f'CPP (mmHg)')
            ax.set_ylim(0, ymax)
            
        # ax.axhspan(0,50, color = 'r', alpha = 0.1)
        # ax.axhspan(50,150, color = 'tab:blue', alpha = 0.1)
        # ax.axhspan(150,ymax, color = 'r', alpha = 0.1)
        
        ax.axhline(50, color = 'r')
        ax.axhline(150, color = 'r')
        
        
        

        
        
        
        
  