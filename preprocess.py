import os
import pandas as pd
from glob import glob

class preprocess:

    def __init__(self, root):

        self.root = root
        self.folders = glob(self.root + '/sensor_data_*')

    def scale_data(self):

        rows = ['obs_1','obs_2','obs_3','obs_4','obs_5','obs_6']

        for folder in self.folders:
            key = folder.split('_', folder.count('_'))[-1]
            file_path = os.path.join(self.root, 'sensor_data_' + key)
            files = glob(file_path + '/' + key + '*.csv')

            for file in files:
                raw_data = pd.DataFrame(pd.read_csv(os.path.join(file_path, file)))
                n_rows = raw_data.shape[0]
                raw_data.index = rows[0:n_rows]
                raw_data = raw_data.drop(columns = 'Unnamed: 0')
                scaled_data = raw_data.apply(lambda x: ((x - x.mean()) / x.std()))
                scaled_data.to_csv(os.path.join(file_path, 'scaled_' + file))

    def write_hdf(self):

        for folder in self.folders:
            key = folder.split('_', folder.count('_'))[-1]
            file_path = os.path.join(self.root, 'sensor_data_' + key)
            files = glob(file_path + '/scaled_*.csv')

            for file in files:
                data = pd.DataFrame(pd.read_csv(os.path.join(file_path, file))).iloc[:, 1:]
                data = data.transpose()
                data.to_hdf(file.replace('.csv', '_parallel.h5'), key = 'data', format = 'table')