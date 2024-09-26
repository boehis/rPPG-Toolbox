import os
import io

import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from .post_process import calculate_metric_per_video
from .metrics import _reform_data_from_dict

tqdm.pandas()

# Function to generate the folder structure for unsupervised and supervised data
def generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=True):
    unsupervised_flag = 'unsupervised' if is_unsupervised else ''
    label_type = 'DiffNormalized' if data_type == 'DiffNormalized_Standardized' else data_type 
    return f"AriaPPG_SizeW{dim}_SizeH{dim}_ClipLength{clip_length}_DataType{data_type}_DataAugNone_LabelType{label_type}_Crop_faceFalse_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse{unsupervised_flag}"

# Paths for unsupervised data
UNSUPERVISED_ROOT_PATH = './runs/infer_configs/UNSUPERVISED/'
UNSUPERVISED_PATHS = [
    # os.path.join(UNSUPERVISED_ROOT_PATH, 'none', 'len128', 'res128', generate_folder_structure(), 'saved_outputs'),
    # os.path.join(UNSUPERVISED_ROOT_PATH, 'first_frame', 'len128', 'res128', generate_folder_structure(), 'saved_outputs'),
    # os.path.join(UNSUPERVISED_ROOT_PATH, 'median', 'len128', 'res128', generate_folder_structure(), 'saved_outputs'),
    # os.path.join(UNSUPERVISED_ROOT_PATH, 'lowpass', 'len128', 'res128', generate_folder_structure(), 'saved_outputs'),
]

# Paths for supervised data
SUPERVISED_ROOT_PATH = './runs/'
SUPERVISED_PATHS = [
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'PURE_PhysNet_DiffNormalized', 'none', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs', 'PURE_PhysNet_DiffNormalized_AriaPPG_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'PURE_PhysNet_DiffNormalized', 'first_frame', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs', 'PURE_PhysNet_DiffNormalized_AriaPPG_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'PURE_PhysNet_DiffNormalized', 'median', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs', 'PURE_PhysNet_DiffNormalized_AriaPPG_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'PURE_PhysNet_DiffNormalized', 'lowpass', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs', 'PURE_PhysNet_DiffNormalized_AriaPPG_outputs.pickle'),
    
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'UBFC-rPPG_PhysNet_DiffNormalized', 'none', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs', 'UBFC-rPPG_PhysNet_DiffNormalized_AriaPPG_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'UBFC-rPPG_PhysNet_DiffNormalized', 'first_frame', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs', 'UBFC-rPPG_PhysNet_DiffNormalized_AriaPPG_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'UBFC-rPPG_PhysNet_DiffNormalized', 'median', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs', 'UBFC-rPPG_PhysNet_DiffNormalized_AriaPPG_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'UBFC-rPPG_PhysNet_DiffNormalized', 'lowpass', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs', 'UBFC-rPPG_PhysNet_DiffNormalized_AriaPPG_outputs.pickle'),
    


    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'PURE_DeepPhys',                    'median',                                   generate_folder_structure(dim=72, clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),   'saved_test_outputs',   'PURE_DeepPhys_AriaPPG_outputs.pickle'),
    os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'PURE_PhysFormer_DiffNormalized',   'median',                                   generate_folder_structure(dim=128, clip_length=160, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'PURE_PhysFormer_DiffNormalized_AriaPPG_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'PURE_TSCAN',                       'median',                                   generate_folder_structure(dim=72, clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),   'saved_test_outputs',   'PURE_TSCAN_AriaPPG_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'infer_configs', 'PURE_iBVPNet_DiffNormalized',      'median',                                   generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'PURE_iBVPNet_DiffNormalized_AriaPPG_outputs.pickle'),

    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'DeepPhys',                         'median',                       'pxt1',     generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_deepphys_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'DeepPhys',                         'median',                       'pxt1stat', generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_deepphys_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'DeepPhys',                         'median',                       'pxtx',     generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_deepphys_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysFormer',                       'median',                       'pxt1',     generate_folder_structure(dim=128, clip_length=160, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_physformer_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysFormer',                       'median',                       'pxt1stat', generate_folder_structure(dim=128, clip_length=160, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_physformer_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysFormer',                       'median',                       'pxtx',     generate_folder_structure(dim=128, clip_length=160, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_physformer_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysNet',                          'median',                       'pxt1',     generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_physnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysNet',                          'median',                       'pxt1stat', generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_physnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysNet',                          'median',                       'pxtx',     generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_physnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'Tscan',                            'median',                       'pxt1',     generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_tscan_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'Tscan',                            'median',                       'pxt1stat', generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_tscan_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'Tscan',                            'median',                       'pxtx',     generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_tscan_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'iBVPNet',                          'median',                       'pxt1',     generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_ibvpnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'iBVPNet',                          'median',                       'pxt1stat', generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_ibvpnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'iBVPNet',                          'median',                       'pxtx',     generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs',   'AriaPPG_AriaPPG_ibvpnet_outputs.pickle'),

    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysFormer',                       'median',   'diffnormalized',   'pxt1',     generate_folder_structure(dim=128, clip_length=160, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_physformer_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysFormer',                       'median',   'diffnormalized',   'pxt1stat', generate_folder_structure(dim=128, clip_length=160, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_physformer_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysFormer',                       'median',   'diffnormalized',   'pxtx',     generate_folder_structure(dim=128, clip_length=160, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_physformer_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysNet',                          'median',   'diffnormalized',   'pxt1',     generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_physnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysNet',                          'median',   'diffnormalized',   'pxt1stat', generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_physnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'PhysNet',                          'median',   'diffnormalized',   'pxtx',     generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_physnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'iBVPNet',                          'median',   'diffnormalized',   'pxt1',     generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_ibvpnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'iBVPNet',                          'median',   'diffnormalized',   'pxt1stat', generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_ibvpnet_outputs.pickle'),
    # os.path.join(SUPERVISED_ROOT_PATH, 'train_configs', 'iBVPNet',                          'median',   'diffnormalized',   'pxtx',     generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs',   'AriaPPG_AriaPPG_ibvpnet_outputs.pickle'),
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

    reformed_data = []
    for index in sorted(data['predictions'].keys()):
        predictions = _reform_data_from_dict(data['predictions'][index], False)
        labels = _reform_data_from_dict(data['labels'][index], False)
        predictions = np.reshape(predictions, (-1))
        labels = np.reshape(labels, (-1))
        reformed_data.append({
            "index": index,
            "method": os.path.splitext(os.path.basename(path))[0],
            "labels": labels,
            "predictions": predictions
        })
        # print(len(predictions), len(labels))
    print(data.keys())
    return pd.DataFrame(reformed_data), data['label_type']

# Unified data loader for both unsupervised and supervised datasets
def load_data(path, data_type='unsupervised'):
    if data_type == 'unsupervised':
        return read_csv_files(path)
    elif data_type == 'supervised':
        return read_pickle_file(path)

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
    # Process unsupervised paths
    for path in UNSUPERVISED_PATHS:
        if os.path.exists(path):
            print(f"Processing unsupervised path: {path}")
            data_frame, label_type = load_data(path, data_type='unsupervised')
            process_and_save_data(data_frame, path, label_type)
        else:
            print(f"Path does not exist: {path}")
    
    # Process supervised paths
    for path in SUPERVISED_PATHS:
        if os.path.exists(path):
            print(f"Processing supervised path: {path}")
            data_frame, label_type = load_data(path, data_type='supervised')
            process_and_save_data(data_frame, os.path.dirname(path), label_type)
        else:
            print(f"Path does not exist: {path}")

# Run the main function
if __name__ == "__main__":
    main()
