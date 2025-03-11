import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from pta_learn.tpmr import TPMR

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score

import pickle
import os



def load_series(series_path):
    df = pd.read_csv(series_path, delimiter='\t', header=None, names=['Time', 'Pressure'])
    return df


def split_series_by_gap(df,gap):
    segments = []
    start_idx = 0
    ts = df['Time'].values
    for i in range(1, len(ts)):
        if (ts[i] - ts[i-1]) > gap:
            segments.append(df.iloc[start_idx:i].copy())
            start_idx = i
    if start_idx < len(ts):
        segments.append(df.iloc[start_idx:].copy())
    return segments


def preprocess_segment(df, grid_timestep=0.01, smoothing_time_window=5, interp_method='time'):
    seg = df.copy()
    seg.set_index('Time', inplace=True)
    stt, endt = np.floor(seg.index.min()), np.ceil(seg.index.max())
    uniform_times = np.arange(stt, endt + 1, grid_timestep)
    
    seg.index = pd.to_timedelta(seg.index, unit='h')
    new_times = pd.to_timedelta(uniform_times, unit='h')
    union_idx = seg.index.union(new_times)
    
    # Interpolate and fill remaining NaNs
    seg_interp = seg.reindex(union_idx).interpolate(interp_method).ffill().bfill()
    
    index_window_size = int(float(smoothing_time_window) / float(grid_timestep))
    index_window_size = int(index_window_size)  # Ensure it's an integer
    
    seg_uniform = seg_interp.reindex(new_times)
    
    # Apply rolling mean and fill NaNs resulting from the window
    seg_uniform = seg_uniform.rolling(window=index_window_size, center=True, min_periods=1).mean()
    
    # Handle any remaining edge NaNs (though min_periods=1 should prevent them)
    seg_uniform = seg_uniform.ffill().bfill()
    
    seg_uniform.index = seg_uniform.index.total_seconds() / 3600
    seg_uniform.reset_index(inplace=True)
    seg_uniform.columns = ['Time', 'Pressure']
    seg_uniform['Timestamp'] = pd.to_timedelta(seg_uniform['Time'], unit='h')
    return seg_uniform


def get_recovery_intervals(df_bhp, p = 1, int_len_treshhold = 70):
    df = df_bhp.copy()
    df['Pressure'] = -df['Pressure'].values + np.max(df['Pressure'].values)
    shutin_bp_all, shutin_bp_interval, shutin_transient_all, shutin_transient_interval = TPMR(df, p, int_len_treshhold)
    
    if shutin_transient_interval.empty:
        shutin_transient_interval = pd.DataFrame(columns = ['start','end'])
    else:
        shutin_transient_interval = shutin_transient_interval[['start/hr', 'end/hr']]
        shutin_transient_interval.columns = ['start','end']
    return shutin_transient_interval


def get_drop_intervals(df_bhp, p = 1, int_len_treshhold = 70):
    shutin_bp_all, shutin_bp_interval, shutin_transient_all, shutin_transient_interval = TPMR(df_bhp, p, int_len_treshhold)
    if shutin_transient_interval.empty:
        shutin_transient_interval = pd.DataFrame(columns = ['start','end'])
    else:
        shutin_transient_interval = shutin_transient_interval[['start/hr', 'end/hr']]
        shutin_transient_interval.columns = ['start','end']
    return shutin_transient_interval


def fit_ln_curve(df):
    df = df[df['Time'] > 0].copy()
    
    def objective(epsilon_shift):
        df['lnTime'] = np.log(df['Time'] + epsilon_shift)
        
        X = np.vstack([df['lnTime'], np.ones(len(df))]).T
        y = df['Pressure'].values
        
        coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        q, b = coeff
        
        y_pred = q * df['lnTime'] + b
        
        return mean_squared_error(y, y_pred)
    
    result = minimize_scalar(objective, bounds=(-min(df['Time']), 1), method='bounded')
    best_epsilon_shift = result.x
    
    df['lnTime'] = np.log(df['Time'] + best_epsilon_shift)
    X = np.vstack([df['lnTime'], np.ones(len(df))]).T
    y = df['Pressure'].values
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    q, b = coeff
    y_pred = q * df['lnTime'] + b
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {'q': q, 'b': b, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Epsilon Shift': best_epsilon_shift}


def compute_tsfresh_features(df):
    # For a single time series, add an id column (all rows belong to the same series)
    df['id'] = 1  
    
    # Extract features; the result is a DataFrame with one row per id
    features_df = extract_features(df, 
                                   column_id='id',
                                   column_sort='Time',
                                   column_value='Pressure',
                                   default_fc_parameters = MinimalFCParameters(),
                                   n_jobs = 1)
    
    # Convert the features of our single time series to a dict
    features_dict = features_df.iloc[0].to_dict()
    return features_dict


def build_test(dataset_path):
    df = []
    for index,filename in enumerate(os.listdir(dataset_path)):
        print(f"{index}: {filename}")
        series_path = os.path.join(dataset_path, filename)
            
        series = load_series(series_path)
        segments = split_series_by_gap(series, 50)
        
        recovery_ints_pred = []
        drop_ints_pred = []
        for seg in segments:
            if len(seg) < 20:
                continue
            proc_seg = preprocess_segment(seg, smoothing_time_window = 7)
            recovery_ints_pred.append(get_recovery_intervals(proc_seg))
            drop_ints_pred.append(get_drop_intervals(proc_seg))

        recovery_ints_pred = pd.concat(recovery_ints_pred)
        drop_ints_pred = pd.concat(drop_ints_pred)
        
        recovery_ints_pred['type'] = 'recovery'
        drop_ints_pred['type'] = 'drop'
        
        all_ints = pd.concat([recovery_ints_pred, drop_ints_pred])

        print(f"{len(recovery_ints_pred)} recovery intervals")
        print(f"{len(drop_ints_pred)} drop intervals")
        print("|-------------------------------------------|")
        
        for _, interval in all_ints.iterrows():
            segment = series[(series['Time'] >= interval['start']) & (series['Time'] <= interval['end'])]
            if len(segment) < 20:
                continue
            seg = segment.copy()
            seg['Time'] = seg['Time'].values - np.min(seg['Time'].values)
            output = fit_ln_curve(seg)
            output['file'] = filename
            output['interval_start'] = interval['start']
            output['interval_end'] = interval['end']
            output['type'] = interval['type']

            # features
            output['duration'] = interval['end'] - interval['start']
            output['len'] = len(seg)
            output['type_binary'] = 1 if interval['type'] == 'recovery' else 0
            output['amplitude'] = seg['Pressure'].max() - seg['Pressure'].min()

            tsfresh_feat = compute_tsfresh_features(seg)
            output.update(tsfresh_feat)
            
            df.append(output)
            
    return pd.DataFrame(df)


if __name__ == "__main__":
    dataset_path = r'./dataset'
    test_path = r'./test.csv'
    output_path = r'./output.csv'
    model_path = r'./models/random_forest_model.pkl' 
    
    
    with open(model_path, 'rb') as file:
        rfc = pickle.load(file)
        
        
    if os.path.exists(test_path):
        print("load existing test set")
        df_test = pd.read_csv(test_path)
    else:
        df_test = build_test(dataset_path)
        df_test.to_csv(test_path, index=False)
        
    
    feats = df_test.drop(['file','interval_start','interval_end','type'], axis = 1)
    df_test['class'] = rfc.predict(feats)
    
    
    result = df_test[df_test['class'] == 1].groupby('file').apply(lambda group: pd.Series({
        'recovery': group.loc[(group['class'] == 1) & (group['type'] == 'recovery'), ['interval_start', 'interval_end']].values.tolist(),
        'drop': group.loc[(group['class'] == 1) & (group['type'] == 'drop'), ['interval_start', 'interval_end']].values.tolist()
    })).reset_index()
    test_filenames = os.listdir(dataset_path)
    missing_files = [filename for filename in test_filenames if filename not in result['file'].values]
    new_rows = pd.DataFrame({
        'file': missing_files,
        'recovery': [[] for _ in missing_files],
        'drop': [[] for _ in missing_files]
    })
    result = pd.concat([result, new_rows], ignore_index=True)
    result = result.sort_values(by='file').reset_index(drop=True)
    result.to_csv(output_path, sep = ',', index=False)
