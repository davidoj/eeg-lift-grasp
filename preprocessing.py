import numpy as np
import pandas as pd

from scipy.signal import butter, lfilter

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP


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