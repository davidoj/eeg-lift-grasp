import numpy as np
import pandas as pd

from scipy.signal import butter, lfilter, firwin, kaiserord, filtfilt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

### Read and preprocess datafiles

def data_preprocess_train(X, events=[], fbands=[], sca=None, cpast=None, bpass=None, 
                          lpass=None, pca=None, csp_filt=None):
    if csp_filt:
        fit_CSP(data, csp, events)
        X = (csp.filters_[0:nfilters].dot(X.T)).T
    X = np.array(X)
    if sca:
        X=scaler.fit_transform(X)
    Xs = []
    if lpass:
        #Xs.append(butter_lowpass_filter(X,lpass,500))
        Xs.append(fir_lowpass_filter(X,lpass,3,500,20))
    if len(fbands)>0:
        Xs.append(prepare_freq_bands(X,fbands)[cpast[0]*cpast[1]:])
    if cpast:
        Xs[0] = concat_past(Xs[0],interval=cpast[0],num_past=cpast[1])
    return np.concatenate(Xs,axis=1)

def data_preprocess_test(X, fbands = [], sca=None, cpast=None, bpass=None, 
                         lpass=None, pca=None, csp_filt=None):
    if csp_filt:
        X = (csp.filters_[0:nfilters].dot(X.T)).T
    X = np.array(X)
    if sca:
        X=scaler.transform(X)
    Xs = []
    if lpass:
        #Xs.append(butter_lowpass_filter(X,lpass,500))
        Xs.append(fir_lowpass_filter(X,lpass,3,500,20))
    if len(fbands)>0:
        Xs.append(prepare_freq_bands(X,fbands)[cpast[0]*cpast[1]:])
    if cpast:
        Xs[0] = concat_past(Xs[0],interval=cpast[0],num_past=cpast[1])
    return np.concatenate(Xs,axis=1)


### CSP with MNE

def creat_mne_raw_object(data, events=[]):
    """Create a mne raw instance from csv file"""

    ch_names = list(data.columns)
    
    # read EEG standard montage from mne
    montage = read_montage('standard_1005',ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = np.array(data[ch_names]).T

    if len(events)>0:
        events_names =list(events.columns)
        events_data = np.array(events[events_names]).T     
        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data,events_data))

    # create and populate MNE info structure
    info = create_info(ch_names,sfreq=500.0, ch_types=ch_type, montage=montage)
    #info['filename'] = fname
    
    # create raw object 
    raw = RawArray(data,info,verbose=False)
    
    return raw

def fit_CSP(data,csp,events=[]):
    epochs_tot = []
    y = []
    
    raw = creat_mne_raw_object(data,events=events)
 
    # pick eeg signal
    picks = pick_types(raw.info,eeg=True)

    events = find_events(raw,stim_channel='HandStart', verbose=False)

    epochs = Epochs(raw, events, {'during' : 1}, 0, 2, proj=False,
                    picks=picks, baseline=None, preload=True,
                    add_eeg_ref=False, verbose=False)

    epochs_tot.append(epochs)
    y.extend([1]*len(epochs))
    
    epochs_rest = Epochs(raw, events, {'before' : 1}, -2, 0, proj=False,
                    picks=picks, baseline=None, preload=True,
                    add_eeg_ref=False, verbose=False)
    
    epochs_rest.times = epochs.times
    
    y.extend([-1]*len(epochs_rest))
    epochs_tot.append(epochs_rest)
        
    epochs = concatenate_epochs(epochs_tot)

    X = epochs.get_data()
    y = np.array(y)
    

    csp.fit(X,y)

### Frequency stuff

def fir_lowpass(highcut,width,fs,atten):
    nyq = float(fs)/2
    width = float(width)/nyq
    N, beta = kaiserord(atten,width)
    taps = firwin(N, highcut/nyq,window=('kaiser',beta))
    return taps

def fir_lowpass_filter(data,highcut,width,fs,atten):
    taps = fir_lowpass(highcut,width,fs,atten)
    y = lfilter(taps,1.0,data,axis=0)
    return y

def fir_lowpass_filtfilter(data,highcut,width,fs,atten):
    taps = fir_lowpass(highcut,width,fs,atten)
    y = filtfilt(taps,1.0,data,axis=0)
    return y


def fir_bandpass(lowcut,highcut,width,fs,atten):
    nyq = float(fs)/2
    width = width/nyq
    N, beta = kaiserord(atten,width)
    taps = firwin(N, [lowcut/nyq,highcut/nyq],window=('kaiser',beta),pass_zero=False)
    return taps

def fir_bandpass_filter(data,lowcut,highcut,width,fs,atten):
    taps = fir_bandpass(lowcut,highcut,width,fs,atten)
    y = lfilter(taps,1.0,data,axis=0)
    return y
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data,axis=0)
    return y

def butter_lowpass(highcut,fs,order=5):
    nyq = 0.5*fs
    high = highcut/nyq
    b,a = butter(order,high,btype="low")
    return b,a

def butter_lowpass_filter(data,highcut,fs,order=5):
    b, a = butter_lowpass(highcut,fs,order=order)
    y = lfilter(b,a,data,axis=0)
    return y


def freq_bands(data,bands):
    nchans = data.shape[1]
    f_band = np.empty((len(data),len(bands)*nchans))
    for j, band in enumerate(bands):
        f_band[:,nchans*j:nchans*(j+1)] = butter_bandpass_filter(data,band[0],band[1],500)
    #for j in range(f_band.shape[1]):
        #f_band[:,j] = smooth(f_band[:,j])
    return f_band**2

def smooth(data,window_len=11,window="hamming"):
    if data.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
   
    if data.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
   
   
    if window_len<3:
        return data
   
   
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
   
  
    s=np.r_[data[window_len-1:0:-1],data,data[-1:-window_len:-1]]
       
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
   
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[:-window_len+1]

def prepare_freq_bands(data,bands):
    f = freq_bands(data,bands)
    fsmo = np.zeros(f.shape)
    for j in range(f.shape[1]):
        fsmo[:,j] = smooth(f[:,j],window='flat',window_len=50)
    return fsmo

### Miscellaneous added features

def concat_past(X,interval=20,num_past=5):
    frames = []
    for i in range(num_past):
        X_trunc  = X[i*interval:-(num_past-i)*interval]
        frames.append(X_trunc)
    X_out = np.concatenate(frames,axis=1)
    return X_out


def pd_concat_past(X,interval=20,num_past=5):
    frames = []
    for i in range(num_past):
        X_trunc  = X[i*interval:-(num_past-i)*interval]
        X_trunc = X_trunc.rename(index=lambda x: x-i*interval)
        frames.append(X_trunc)
    X_out = pd.concat(frames,axis=1)
    return X_out
