import numpy as np
import os
import pandas as pd
from glob import glob
from scipy import signal
from scipy.signal import windows
from scipy.io import loadmat


class compute_psd:
    
    def __init__(self, root):

        self.root = root
        self.folders = glob(self.root + '/ambient')
        self.files = {}
        self.build_file_dicts()

    def build_file_dicts(self):

        for folder in self.folders:
            key = folder.split('_', folder.count('_'))[-1]
            self.files[key] = {}
            curr_dir = os.path.join(self.root, folder)

            for day in range(1, 32):
                self.files[key][day] = [os.path.join(curr_dir, file) for file in os.listdir(curr_dir) 
                                        if (os.path.isfile(os.path.join(curr_dir, file)) and int(file[14:16]) == day)]
                
    def init_sensors(Self):

        sensors = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
        freq = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
        return sensors, freq
    
    def compute_psd(Self, data, sensor):

        tdata = data['predat_a']['tdata'][0][0]
        fs = data['predat_a']['fs'][0][0]
        hann_window = windows.hann(2**14)
        n_overlap = round(0.66*len(hann_window))
        f, psd = signal.welch(tdata[:, sensor - 1], fs, window = hann_window, noverlap = n_overlap)
        return psd, f
    
    def build_data(self):

        for key, month in self.files.items():
            save_path = os.path.join(self.root, 'sensor_data_' + key)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for day, files in month.items():
                sensors, freq = self.init_sensors()

                for file in files:
                    data = loadmat(file)

                    for sensor in sensors.keys():
                        psd, f = self.compute_psd(data, sensor)
                        sensors[sensor].append(psd)
                        freq[sensor].append(f)

                for sensor in sensor.keys():
                    data_entry = []
                    
                    for obs in range(0, len(sensors[sensor])):
                        if not np.isnan(sensors[sensor][obs]).any():
                            data_entry.append(sensors[sensor][obs])
                    
                    if data_entry:
                        pd.DataFrame(data_entry).to_csv(os.path.join(save_path, key + '_day_' + str(day) + '_sensor_' + str(sensor) + '.csv'))
