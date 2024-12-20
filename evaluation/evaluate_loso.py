import os
import io
import glob

import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from .post_process import calculate_metric_per_video
from .metrics import _reform_data_from_dict
from joblib import Parallel, delayed
tqdm.pandas()

# Function to generate the folder structure for unsupervised and supervised data
def generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=True):
    unsupervised_flag = 'unsupervised' if is_unsupervised else ''
    label_type = 'DiffNormalized' if data_type == 'DiffNormalized_Standardized' else data_type 
    return f"AriaPPG_SizeW{dim}_SizeH{dim}_ClipLength{clip_length}_DataType{data_type}_DataAugNone_LabelType{label_type}_Crop_faceFalse_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse{unsupervised_flag}"



# Paths for supervised data
SUPERVISED_ROOT_PATH = '/cluster/scratch/boehis/runs/train_configs/'
SUPERVISED_PATHS = [
    # ('DeepPhys_finetune',   os.path.join(SUPERVISED_ROOT_PATH, 'DeepPhys',    'median',  'loso_cv_finetune')) ,
    # ('DeepPhys_retrain',    os.path.join(SUPERVISED_ROOT_PATH, 'DeepPhys',    'median',  'loso_cv_retrain')) ,
    # ('PhysFormer_finetune', os.path.join(SUPERVISED_ROOT_PATH, 'PhysFormer',  'median',  'loso_cv_finetune')) ,
    # ('PhysFormer_retrain',  os.path.join(SUPERVISED_ROOT_PATH, 'PhysFormer',  'median',  'loso_cv_retrain')) ,
    # ('PhysNet_finetune',    os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'loso_cv_finetune')) ,
    # ('PhysNet_retrain',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'loso_cv_retrain')) ,
    # ('Tscan_finetune',      os.path.join(SUPERVISED_ROOT_PATH, 'Tscan',       'median',  'loso_cv_finetune')) ,
    # ('Tscan_retrain',       os.path.join(SUPERVISED_ROOT_PATH, 'Tscan',       'median',  'loso_cv_retrain')) ,
    # ('iBVPNet_finetune',    os.path.join(SUPERVISED_ROOT_PATH, 'iBVPNet',     'median',  'loso_cv_finetune')) ,
    # ('iBVPNet_retrain',     os.path.join(SUPERVISED_ROOT_PATH, 'iBVPNet',     'median',  'loso_cv_retrain')) ,

    # ('PhysNet_loso_mcc',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'loss_function_comparison', 'loso_mcc')) ,        
    # ('PhysNet_loso_soft_macc',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'loss_function_comparison', 'loso_soft_macc')) ,  
    # ('PhysNet_loso_soft_msacc',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'loss_function_comparison', 'loso_soft_msacc')) , 
    # ('PhysNet_loso_talos',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'loss_function_comparison', 'loso_talos')) ,      
    # ('PhysNet_loso_val_macc',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'loss_function_comparison', 'loso_val_macc')) ,

    # ('PhysFormer_loso_mcc',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysFormer',     'median',  'loss_function_comparison', 'loso_mcc')) ,        
    # ('PhysFormer_loso_soft_macc',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysFormer',     'median',  'loss_function_comparison', 'loso_soft_macc')) ,  
    # ('PhysFormer_loso_soft_msacc',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysFormer',     'median',  'loss_function_comparison', 'loso_soft_msacc')) , 
    # ('PhysFormer_loso_talos',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysFormer',     'median',  'loss_function_comparison', 'loso_talos')) ,
    
    # ('PhysNet_multim_adaptive_norm_10_imu',    os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'adaptive_norm_10_imu')),
    # ('PhysNet_multim_adaptive_norm_10_quaternion',    os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'adaptive_norm_10_quaternion')),
    # ('PhysNet_multim_adaptive_norm_2_imu',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'adaptive_norm_2_imu')),
    # ('PhysNet_multim_adaptive_norm_2_quaternion',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'adaptive_norm_2_quaternion')),
    # ('PhysNet_multim_adaptive_norm_4_imu',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'adaptive_norm_4_imu')),
    # ('PhysNet_multim_adaptive_norm_4_quaternion',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'adaptive_norm_4_quaternion')),
    # ('PhysNet_multim_adaptive_norm_8_imu',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'adaptive_norm_8_imu')),
    # ('PhysNet_multim_adaptive_norm_8_quaternion',     os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'adaptive_norm_8_quaternion')),
    # ('PhysNet_multim_add_10_imu',              os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'add_10_imu')),
    # ('PhysNet_multim_add_10_quaternion',              os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'add_10_quaternion')),
    # ('PhysNet_multim_add_2_imu',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'add_2_imu')),
    # ('PhysNet_multim_add_2_quaternion',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'add_2_quaternion')),
    # ('PhysNet_multim_add_4_imu',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'add_4_imu')),
    # ('PhysNet_multim_add_4_quaternion',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'add_4_quaternion')),
    # ('PhysNet_multim_add_8_imu',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'add_8_imu')),
    # ('PhysNet_multim_add_8_quaternion',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'add_8_quaternion')),
    # ('PhysNet_multim_cat_10_imu',              os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cat_10_imu')),
    # ('PhysNet_multim_cat_10_quaternion',              os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cat_10_quaternion')),
    # ('PhysNet_multim_cat_2_imu',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cat_2_imu')),
    # ('PhysNet_multim_cat_2_quaternion',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cat_2_quaternion')),
    # ('PhysNet_multim_cat_4_imu',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cat_4_imu')),
    # ('PhysNet_multim_cat_4_quaternion',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cat_4_quaternion')),
    # ('PhysNet_multim_cat_8_imu',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cat_8_imu')),
    # ('PhysNet_multim_cat_8_quaternion',               os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cat_8_quaternion')),
    # ('PhysNet_multim_cross_attention_10_imu',  os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_10_imu')),
    # ('PhysNet_multim_cross_attention_10_quaternion',  os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_10_quaternion')),
    # ('PhysNet_multim_cross_attention_2_imu',   os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_2_imu')),
    # ('PhysNet_multim_cross_attention_2_quaternion',   os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_2_quaternion')),
    # ('PhysNet_multim_cross_attention_4_imu',   os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_4_imu')),
    # ('PhysNet_multim_cross_attention_4_quaternion',   os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_4_quaternion')),
    # ('PhysNet_multim_cross_attention_8_imu',   os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_8_imu')),
    # ('PhysNet_multim_cross_attention_8_quaternion',   os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_8_quaternion')),
    ('PhysNet_multim_cross_attention_2_4_8_10_imu',   os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_2_4_8_10_imu')),
    ('PhysNet_multim_cross_attention_skip_8_imu',   os.path.join(SUPERVISED_ROOT_PATH, 'PhysNet',     'median',  'multimodal', 'cross_attention_skip_8_imu')),
]


# Reform function to reshape the data
def reform_data(df_slice):
    predictions = np.concatenate(df_slice.sort_values(by='slice')['prediction'].values)
    labels = np.concatenate(df_slice.sort_values(by='slice')['label'].values)
    return {
        "index": df_slice['index'].iloc[0],
        "method": df_slice['method'].iloc[0],
        "labels": labels,
        "predictions": predictions
    }

# Function to read CSV files (unsupervised data)
def read_csv_files(path):
    data_frames = []
    for result_file in os.listdir(path):
        if result_file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, result_file), index_col=0)
            df['method'] = result_file.split('_')[0]
            data_frames.append(df)
    combined_df = pd.concat(data_frames)
    combined_df['prediction'] = combined_df['prediction'].apply(eval)
    combined_df['label'] = combined_df['label'].apply(eval)

    result_list = []
    for (index, method), group in tqdm(combined_df.groupby(['index', 'method']), desc='Reforming Data'):
        result_list.append(reform_data(group))
    
    return pd.DataFrame(result_list).reset_index(drop=True), None

# Function to read Pickle files (supervised data)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def read_pickle_file(path):
    with open(path, 'rb') as f:
        data = CPU_Unpickler(f).load()
    return data

# Unified data loader for both unsupervised and supervised datasets
def load_data(path):
    all_data = dict()
    files = glob.glob(os.path.join(path,"fold_*","*","saved_test_outputs","*.pickle"))
    assert len(files)==22, f"Should be 22 files!, found {len(files)}"
    for file in files:
        new_dict = read_pickle_file(file)       
        for key, value in new_dict.items():
            if key in all_data:
                if isinstance(value, dict) and isinstance(all_data[key], dict):
                    all_data[key].update(value)
                # Otherwise, you can append to a list or handle based on your needs
                else:
                    if not key in ['label_type', 'fs']:
                        print(f"Conflict for key: {key}, consider handling")
            else:
                all_data[key] = value
    assert len(all_data['predictions'].keys()) == 216, f"Expected 216 keys. Got {len(all_data['predictions'].keys())}"
    
    reformed_data = []
    for index in sorted(all_data['predictions'].keys()):
        predictions = _reform_data_from_dict(all_data['predictions'][index], False)
        labels = _reform_data_from_dict(all_data['labels'][index], False)
        predictions = np.reshape(predictions, (-1))
        labels = np.reshape(labels, (-1))
        reformed_data.append({
            "index": index,
            "labels": labels,
            "predictions": predictions
        })
    return pd.DataFrame(reformed_data), all_data['label_type']


# Function to calculate metrics
def get_metrics(row, label_type=None):
    # print(label_type)
    diff_flag = label_type == 'DiffNormalized' 

    hr_label_FFT, hr_pred_FFT, SNR_FFT, macc = calculate_metric_per_video(
        row['predictions'], row['labels'], fs=30, diff_flag=diff_flag, use_bandpass=True, hr_method='FFT'
    )
    hr_label_Peak, hr_pred_Peak, SNR_Peak, _ = calculate_metric_per_video(
        row['predictions'], row['labels'], fs=30, diff_flag=diff_flag, use_bandpass=True, hr_method='Peak'
    )
    PID = row['index'][4:8]
    TID = row['index'][8:]
    CAM = row['index'][:4]
    
    return pd.Series({
        'PID': PID,
        'TID': TID,
        'CAM': CAM,
        'method': row['method'],
        "hr_label_Peak": hr_label_Peak,
        "hr_pred_Peak": hr_pred_Peak,
        "SNR_Peak": SNR_Peak,
        "hr_label_FFT": hr_label_FFT,
        "hr_pred_FFT": hr_pred_FFT,
        "SNR_FFT": SNR_FFT,
        "macc": macc
    })

# Function to process data in parallel and calculate metrics
def process_data(data_frame, label_type):
    # print(label_type)
    metrics_list = Parallel(n_jobs=-1)(
        delayed(get_metrics)(row, label_type=label_type) for _, row in tqdm(data_frame.iterrows(), total=len(data_frame), desc="Calculating metrics")
    )
    return pd.DataFrame(metrics_list)

# Function to save the resulting DataFrame as CSV
def save_as_csv(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")

# Function to process and save data (combined step)
def process_and_save_data(data_frame, path, label_type):
    metrics_df = process_data(data_frame, label_type)
    output_file = os.path.join(path, 'metrics_output.csv')
    save_as_csv(metrics_df, output_file)

# Main function to process all paths
def main():    
    # Process supervised paths
    for method, path in SUPERVISED_PATHS:
        if os.path.exists(path):
            print(f"Processing supervised path: {path}")
            data_frame, label_type = load_data(path)
            data_frame['method'] = method
            process_and_save_data(data_frame, path, label_type)
        else:
            print(f"Path does not exist: {path}")

# Run the main function
if __name__ == "__main__":
    main()
