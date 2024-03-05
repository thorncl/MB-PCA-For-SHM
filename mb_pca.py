import dask.dataframe as daskdf
import pandas as pd
import numpy as np
from compute_delayed import *
from glob import glob


class NIPALS_MBPCA:

    def __init__ (self, root):

        self.root = root
        self.n_rows, self.n_features = self.get_shape()
        self.t_T = np.asarray([-0.40824829, -0.16738194, -0.82674259,  0.06009628, -0.19901919,-0.28034463]).reshape(-1, 1)
        self.folders = glob(self.root + '/sensor_data_*')
        self.folders.sort()
    
    def get_shape(self) -> int:

        sample = pd.read_hdf(self.root + 'sensor_data_201901/scaled_201901_day_1_sensor_1_parallel_padded.h5', key='data').transpose()
        return sample.shape[0], sample.shape[1]
    
    def pad_missing_values(self, read_keys):

        self.block_files = []

        for folder in self.folders:
            key = folder.split('_', folder.count('_'))[-1]

            if key in read_keys:
                self.block_files.extend(glob(self.root + '/sensor_data_' + key + 'scaled_*_parallel.h5'))

        for file in self.block_files:
            df = pd.read_hdf(file, key = 'data', mode = 'r')
            
            if df.shape[1] != self.n_rows:
                cols = df.shape[1]

                for i in range(cols, self.n_rows):
                    df.loc[:, i] = 0

            df.to_hdf(file.replace('.h5', '_padded.h5'), key = 'data', format = 'table')

    def compute_block_loadings(self, no_norm = False):

        p_b = []
        df_map = daskdf.map_partitions(dot_p_b, self.t_T, self.df, self.n_rows, self.n_features, no_norm)
        delayed = df_map.compute()

        for i in range(len(delayed)):
            p_b.append(delayed[i].compute())

        self.p_b = p_b

    def compute_block_scores(self):

        self.t_b = []
        
        for i in range(len(self.p_b[0])):
            df_map = daskdf.map_partitions(dot_t_b, self.p_b[i][0], self.df, self.n_rows, self.n_features, align_dataframes = True)
            delayed = df_map.compute()

            for j in range (len(delayed)):
                self.t_b.append(delayed[j].compute())

        self.T = np.reshape(self.t_b, (self.n_rows, len(self.t_b)))

    def compute_super_weights(self):

        num = self.T.transpose().dot(self.t_T)
        denom = self.t_T.transpose().dot(self.t_T)
        w_T = num / denom
        self.w_T = w_T / np.linalg.norm(w_T)

    def update_super_score(self) -> np.ndarray:

        self.t_T_new = self.T.dot(self.w_T)
        return self.t_T_new

    def deflate(self) -> daskdf:

        E = []
        start_ptr = 0
        self.compute_block_loadings(no_norm = True)
        self.p_b = np.reshape(self.p_b, (self.n_partitions, self.n_features))
        df = self.df.compute()

        for i in range(self.p_b.shape[0]):
            p_b = self.p_b[i].reshape(-1, 1)
            E.append(df[start_ptr:start_ptr + self.n_features].transpose() - self.t_T.dot(p_b.transpose()))
            start += self.n_features
        
        E = np.reshape(E, (self.n_partitions, self.n_features, self.n_rows))
        E = np.vstack(E)
        df = pd.DataFrame(E, columns = pd.Int64Index([0, 1, 2, 3, 4, 5], dtype = 'int64'))
        df = daskdf.from_pandas(df, npartitions = self.n_partitions)
        self.df = df
        return self.df

    def init_model(self, X, n_components, read_ckp) -> dict:

        X = X.reset_index(drop = True)
        self.df = X
        self.n_partitions = self.df.npartitions
        model = {'t_T_scores':np.empty((0,self.n_rows)),
                 'p_b':np.empty((0,self.n_partitions,self.n_features)),
                 't_b':np.empty((0,self.n_partitions,self.n_rows)),
                 'w_T':np.empty((0,self.n_partitions)),
                 'residuals':np.empty((0,self.n_features,self.n_rows)),
                 'cum_var_exp':np.empty((0,self.n_partitions))}
        ckp_components = 0

        if read_ckp is not None:
            model['cum_var_exp'] = pd.read_hdf(glob(self.root + '/cum_var_exp_ckp.h5')[0], key = 'data').values
            ckp_components = int(model['cum_var_exp'].shape[0])

            if ckp_components + n_components > self.n_features:
                return
            
            model['t_T_scores'] = pd.read_hdf(glob(self.root + '/t_T_score_ckp.h5')[0], key = 'data').values

            model['p_b'] = pd.read_hdf(glob(self.root + '/p_b_stack_ckp.h5')[0], key = 'data').values
            model['p_b'] = model['p_b'].reshape(ckp_components, self.n_partitions, self.n_features)

            model['t_b'] = pd.read_hdf(glob(self.root + '/t_b_stack_ckp.h5')[0], key = 'data').values
            model['t_b'] = model['t_b'].reshape(ckp_components, self.n_partitions, self.n_rows)

            model['w_T'] = pd.read_hdf(glob(self.root + '/w_T_ckp.h5')[0], key = 'data').values

            residuals = pd.read_hdf(glob(self.root + '/residuals_stack_ckp.h5')[0], key = 'data')
            residuals = residuals.reset_index(drop=True)

            model['residuals'] = daskdf.from_pandas(residuals, self.n_partitions)
            model['residuals'] = model['residuals'].reset_index(drop = True)

            self.t_T = np.array(model['t_T_scores'][-1]).reshape(-1, 1)
            self.df = model['residuals']
            
        return model
            
    def update_model(self, model) -> dict:

        self.p_b = np.reshape(self.p_b, (1, self.n_partitions, self.n_features))
        model['p_b'] = np.concatenate((model['p_b'], self.p_b), axis = 0)
        model['t_T_scores'] = np.append(model['t_T_scores'], self.t_T.transpose(), axis = 0)
        model['t_b'] = np.append(model['t_b'], np.reshape(self.t_b, (1, self.n_partitions, self.n_rows)), axis = 0)
        model['w_T'] = np.append(model['w_T'], self.w_T.transpose(), axis = 0)

        self.residuals = self.deflate()
        self.residuals = self.residuals.reset_index(drop = True)

    def fit(self, X, tolerance, max_iter, n_components, read_ckp = None, write_ckp = None) -> dict:

        model = self.init_model(X, n_components, read_ckp)

        ckp_components = int(model['cum_var_exp'].shape[0])
        total_components = int(n_components + ckp_components)

        for i in range(ckp_components, total_components):
            print('Fitting Component: ', i + 1)

            for j in range(max_iter):
                print('Current Iteration: ', j)
                self.compute_block_loadings()
                self.compute_block_scores()
                self.compute_super_weights()
                t_T = self.update_super_score()
                eps = abs(t_T - self.t_T)

                if np.all(eps < tolerance):
                    print('Component Converged Early at Iteration: ', j)
                    self.t_T = t_T
                    break

                self.t_T = t_T
            
            model = self.update_model(model)

            df_map = daskdf.map_partitions(compute_variance_explained, self.residuals, X)
            delayed = df_map.compute()
            delayed_vals = []

            for func in range(len(delayed)):
                delayed_vals.append(delayed[func].compute())

            delayed_vals = np.reshape(delayed_vals, (1, self.n_partitions))
            model['cum_var_exp'] = np.append(model['cum_var_exp'], delayed_vals, axis = 0)

        residuals = self.residuals.compute()
        residuals = residuals.values
        residuals = residuals.reshape(self.n_partitions, self.n_features, self.n_rows)
        model['residuals'] = residuals

        if write_ckp is not None:
            self.write_checkpoint(model, total_components)

        return model
    
    def write_checkpoint(self, model, n_components):

        cum_var_exp = np.reshape(model['cum_var_exp'], (n_components, self.n_partitions))
        self.write_hdf(cum_var_exp, 'cum_var_exp')

        residuals = np.reshape(model['residuals'], (self.n_partitions, self.n_features, self.n_rows))
        self.write_hdf(residuals, 'residuals', stack = True)

        p_b = np.reshape(model['p_b'], (n_components, self.n_partitions, self.n_features))
        self.write_hdf(p_b, 'p_b', stack = True)

        t_T_score = np.reshape(model['t_T_scores'], (n_components, self.n_rows))
        self.write_hdf(t_T_score, 't_T_scores')

        t_b = np.reshape(model['t_b'], (n_components, self.n_partitions, self.n_rows))
        self.write_hdf(t_b, 't_b', stack = True)

        w_T = np.reshape(model['w_T'], (n_components, self.n_partitions))
        self.write_hdf(w_T, 'w_T')

    def write_hdf(self, param, param_name, stack = False):

        if stack:
            param_name = param_name + '_stack'
            param = np.vstack(param)
            
        param_df = pd.DataFrame(param)
        param_df.to_hdf(self.root + param_name + '_ckp.h5', key = 'data', format = 'fixed')