import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from scipy.optimize import minimize_scalar
from pta_learn.tpmr import TPMR
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_series(series_path):
    df = pd.read_csv(series_path, delimiter='\t', header=None, names=['Time', 'Pressure'])
    return df

def split_series_by_gap(df, gap):
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
    index_window_size = int(index_window_size)  # Ensure it's integer
    
    seg_uniform = seg_interp.reindex(new_times)
    
    # Apply rolling mean
    seg_uniform = seg_uniform.rolling(window=index_window_size, center=True, min_periods=1).mean()
    
    # Fill edges
    seg_uniform = seg_uniform.ffill().bfill()
    
    seg_uniform.index = seg_uniform.index.total_seconds() / 3600
    seg_uniform.reset_index(inplace=True)
    seg_uniform.columns = ['Time', 'Pressure']
    seg_uniform['Timestamp'] = pd.to_timedelta(seg_uniform['Time'], unit='h')
    return seg_uniform

def get_recovery_intervals(df_bhp, p=1, int_len_treshhold=70):
    df = df_bhp.copy()
    df['Pressure'] = -df['Pressure'].values + np.max(df['Pressure'].values)
    _, _, _, shutin_transient_interval = TPMR(df, p, int_len_treshhold)
    
    if shutin_transient_interval.empty:
        shutin_transient_interval = pd.DataFrame(columns=['start', 'end'])
    else:
        shutin_transient_interval = shutin_transient_interval[['start/hr', 'end/hr']]
        shutin_transient_interval.columns = ['start', 'end']
    return shutin_transient_interval

def get_drop_intervals(df_bhp, p=1, int_len_treshhold=70):
    _, _, _, shutin_transient_interval = TPMR(df_bhp, p, int_len_treshhold)
    if shutin_transient_interval.empty:
        shutin_transient_interval = pd.DataFrame(columns=['start', 'end'])
    else:
        shutin_transient_interval = shutin_transient_interval[['start/hr', 'end/hr']]
        shutin_transient_interval.columns = ['start', 'end']
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
    
    return {
        'q': q, 'b': b,
        'MSE': mse, 'RMSE': rmse,
        'MAE': mae, 'R2': r2,
        'Epsilon Shift': best_epsilon_shift
    }

def compute_tsfresh_features(df):
    df['id'] = 1
    features_df = extract_features(
        df, 
        column_id='id',
        column_sort='Time',
        column_value='Pressure',
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=1
    )
    return features_df.iloc[0].to_dict()

def build_test_for_single_file(file_path, model):
    
    series = load_series(file_path)
    segments = split_series_by_gap(series, 50)
    
    interval_records = []
    for seg in segments:
        if len(seg) < 20:
            continue
        
        proc_seg = preprocess_segment(seg, smoothing_time_window=7)
        recovery_ints_pred = get_recovery_intervals(proc_seg)
        drop_ints_pred = get_drop_intervals(proc_seg)
        
        recovery_ints_pred['type'] = 'recovery'
        drop_ints_pred['type'] = 'drop'
        
        all_ints = pd.concat([recovery_ints_pred, drop_ints_pred])
        
        for _, interval in all_ints.iterrows():
            segment_df = series[(series['Time'] >= interval['start']) & (series['Time'] <= interval['end'])]
            if len(segment_df) < 20:
                continue
            
            seg_ = segment_df.copy()
            seg_['Time'] = seg_['Time'].values - np.min(seg_['Time'].values)
            ln_curve = fit_ln_curve(seg_)
            
            ln_curve['interval_start'] = interval['start']
            ln_curve['interval_end'] = interval['end']
            ln_curve['type'] = interval['type']
            ln_curve['duration'] = interval['end'] - interval['start']
            ln_curve['len'] = len(seg_)
            ln_curve['type_binary'] = 1 if interval['type'] == 'recovery' else 0
            ln_curve['amplitude'] = seg_['Pressure'].max() - seg_['Pressure'].min()
            
            feats = compute_tsfresh_features(seg_)
            ln_curve.update(feats)
            
            interval_records.append(ln_curve)
    
    if not interval_records:
        return pd.DataFrame([])  
    
    df_test = pd.DataFrame(interval_records)
    
    drop_cols = ['interval_start','interval_end','type']
    feats_df = df_test.drop(drop_cols, axis=1)

    df_test['class'] = model.predict(feats_df)
    return df_test


def main():
    st.markdown(
        """
        <style>
        /* Title (h1) should be #30d5c8 */
        h1 {
            color: #30d5c8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("PressureAI")
    st.markdown(
        """
        <div class="explanation">
          Загрузите одиночный файл, состоящий из двух столбцов:
          <br/>Time (ч) и Pressure (бар), разделённые табуляцией.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    @st.cache_resource
    def load_rf_model():
        model_path = r'./models/random_forest_model.pkl'
        with open(model_path, 'rb') as file:
            rf = pickle.load(file)
        return rf
    
    rf_model = load_rf_model()
    
    uploaded_file = st.file_uploader("Выберите файл с временным рядом")
    if uploaded_file is not None:
        temp_file_path = "./_temp_inference_file.txt"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.success("Файл загружен. Нажмите 'Выполнить обработку' для старта.")
        
        if st.button("Выполнить обработку"):
            with st.spinner("Обработка..."):
                df_result = build_test_for_single_file(temp_file_path, rf_model)
                
                if df_result.empty:
                    st.warning("Не удалось обнаружить пригодные интервалы в этом файле.")
                else:
                    
                    intervals_df = df_result[df_result['class'] == 1][['interval_start','interval_end','type']]
                    
                    original_series = load_series(temp_file_path)
                    
                    fig, ax = plt.subplots()
                    ax.plot(original_series['Time'], original_series['Pressure'], label='Pressure', lw=1)
                    
                    for _, row in intervals_df.iterrows():
                        c = 'blue' if row['type'] == 'recovery' else 'red'
                        ax.axvspan(row['interval_start'], row['interval_end'], color=c, alpha=0.2)
                    
                    ax.set_xlabel("Time (hr)")
                    ax.set_ylabel("Pressure")
                    ax.set_title("Выделенные интервалы (синие = recovery, красные = drop)")
                    st.pyplot(fig)
                    

                    intervals_df = intervals_df.sort_values(by='interval_start').reset_index(drop=True)
                    csv_data = intervals_df.to_csv(index=False, sep=',')
                    
                    st.download_button(
                        label="Скачать результат",
                        data=csv_data,
                        file_name="intervals_result.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
