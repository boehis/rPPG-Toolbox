import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

import scipy.stats as stats
from itertools import combinations


COLOR_STAT_CLOSE = '#8663DB'
COLOR_STAT_FAR = '#D663DB'
COLOR_HEAD_STAT = '#638DDB'
COLOR_HEAD_MOV = '#63D9DB'

TEXT_HEAD_STILL = 'Still'
TEXT_HEAD_MOVING = 'Natural Movements'
TEXT_STAT_CLOSE = 'Close'
TEXT_STAT_FAR = 'Far'

SKIP_STATS = False
# Y_LIST = ['abs_err_fft', 'abs_err_peak', 'macc']
Y_LIST = ['abs_err_peak', 'macc']

# Function to generate folder structure (as used in the processing script)
def generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=True):
    unsupervised_flag = '_unsupervised' if is_unsupervised else ''
    label_type = 'DiffNormalized' if data_type == 'DiffNormalized_Standardized' else data_type 
    return f"AriaPPG_SizeW{dim}_SizeH{dim}_ClipLength{clip_length}_DataType{data_type}_DataAugNone_LabelType{label_type}_Crop_faceFalse_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse{unsupervised_flag}"

# Static variables
ROOT_PATH_UNSUPERVISED = '/cluster/scratch/boehis/runs/infer_configs/UNSUPERVISED/'
ROOT_PATH_SUPERVISED = '/cluster/scratch/boehis/runs/'

PATHS = [
    # ('none',        os.path.join(ROOT_PATH_UNSUPERVISED, 'none', 'len128', 'res128', generate_folder_structure(), 'saved_outputs')),
    # ('first_frame', os.path.join(ROOT_PATH_UNSUPERVISED, 'first_frame', 'len128', 'res128', generate_folder_structure(), 'saved_outputs')),
    # ('median',      os.path.join(ROOT_PATH_UNSUPERVISED, 'median', 'len128', 'res128', generate_folder_structure(), 'saved_outputs')),
    # ('lowpass',     os.path.join(ROOT_PATH_UNSUPERVISED, 'lowpass', 'len128', 'res128', generate_folder_structure(), 'saved_outputs')),

    # ('none',        os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'PURE_PhysNet_DiffNormalized', 'none', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs')),
    # ('first_frame', os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'PURE_PhysNet_DiffNormalized', 'first_frame', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs')),
    # ('median',      os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'PURE_PhysNet_DiffNormalized', 'median', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs')),
    # ('lowpass',     os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'PURE_PhysNet_DiffNormalized', 'lowpass', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs')),
    
    # ('none',        os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'UBFC-rPPG_PhysNet_DiffNormalized', 'none', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs')),
    # ('first_frame', os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'UBFC-rPPG_PhysNet_DiffNormalized', 'first_frame', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs')),
    # ('median',      os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'UBFC-rPPG_PhysNet_DiffNormalized', 'median', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs')),
    # ('lowpass',     os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'UBFC-rPPG_PhysNet_DiffNormalized', 'lowpass', generate_folder_structure(data_type='DiffNormalized', is_unsupervised=False), 'saved_test_outputs')),


    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'PURE_DeepPhys',                    'median',                                   generate_folder_structure(dim=72, clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'PURE_PhysFormer_DiffNormalized',   'median',                                   generate_folder_structure(dim=128, clip_length=160, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'PURE_TSCAN',                       'median',                                   generate_folder_structure(dim=72, clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'infer_configs', 'PURE_iBVPNet_DiffNormalized',      'median',                                   generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),


    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'DeepPhys',   'median', 'loso_cv_finetune')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'DeepPhys',   'median', 'loso_cv_retrain')),

    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer', 'median', 'loso_cv_finetune')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer', 'median', 'loso_cv_retrain')),

    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',    'median', 'loso_cv_finetune')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',    'median', 'loso_cv_retrain')),

    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'Tscan',      'median', 'loso_cv_finetune')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'Tscan',      'median', 'loso_cv_retrain')),

    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'iBVPNet',    'median', 'loso_cv_finetune')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'iBVPNet',    'median', 'loso_cv_retrain')),

    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',    'median', 'loss_function_comparison', 'loso_mcc')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',    'median', 'loss_function_comparison', 'loso_soft_macc')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',    'median', 'loss_function_comparison', 'loso_soft_msacc')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',    'median', 'loss_function_comparison', 'loso_talos')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',    'median', 'loss_function_comparison', 'loso_val_macc')),

    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',    'median', 'loss_function_comparison', 'loso_mcc')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',    'median', 'loss_function_comparison', 'loso_soft_macc')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',    'median', 'loss_function_comparison', 'loso_soft_msacc')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',    'median', 'loss_function_comparison', 'loso_talos')),

    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'adaptive_norm_10_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'adaptive_norm_2_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'adaptive_norm_4_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'adaptive_norm_8_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'add_10_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'add_2_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'add_4_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'add_8_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'cat_10_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'cat_2_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'cat_4_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'cat_8_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'cross_attention_10_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'cross_attention_2_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'cross_attention_4_imu')),
    ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',  'median',  'multimodal', 'cross_attention_8_imu')),

    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'DeepPhys',                         'median',                       'pxt1',     generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'DeepPhys',                         'median',                       'pxt1stat', generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'DeepPhys',                         'median',                       'pxtx',     generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',                       'median',                       'pxt1',     generate_folder_structure(dim=128, clip_length=160, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',                       'median',                       'pxt1stat', generate_folder_structure(dim=128, clip_length=160, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',                       'median',                       'pxtx',     generate_folder_structure(dim=128, clip_length=160, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',                          'median',                       'pxt1',     generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',                          'median',                       'pxt1stat', generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',                          'median',                       'pxtx',     generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'Tscan',                            'median',                       'pxt1',     generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'Tscan',                            'median',                       'pxt1stat', generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'Tscan',                            'median',                       'pxtx',     generate_folder_structure(dim=72,  clip_length=128, data_type='DiffNormalized_Standardized', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'iBVPNet',                          'median',                       'pxt1',     generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'iBVPNet',                          'median',                       'pxt1stat', generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'iBVPNet',                          'median',                       'pxtx',     generate_folder_structure(dim=128, clip_length=128, data_type='Raw', is_unsupervised=False),              'saved_test_outputs')),

    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',                       'median',   'diffnormalized',   'pxt1',     generate_folder_structure(dim=128, clip_length=160, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',                       'median',   'diffnormalized',   'pxt1stat', generate_folder_structure(dim=128, clip_length=160, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysFormer',                       'median',   'diffnormalized',   'pxtx',     generate_folder_structure(dim=128, clip_length=160, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',                          'median',   'diffnormalized',   'pxt1',     generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',                          'median',   'diffnormalized',   'pxt1stat', generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'PhysNet',                          'median',   'diffnormalized',   'pxtx',     generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'iBVPNet',                          'median',   'diffnormalized',   'pxt1',     generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'iBVPNet',                          'median',   'diffnormalized',   'pxt1stat', generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
    # ('median', os.path.join(ROOT_PATH_SUPERVISED, 'train_configs', 'iBVPNet',                          'median',   'diffnormalized',   'pxtx',     generate_folder_structure(dim=128, clip_length=128, data_type='DiffNormalized', is_unsupervised=False),   'saved_test_outputs')),
]


# Function to load the CSV file generated in the previous step
def load_csv(path):
    csv_path = os.path.join(path, 'metrics_output.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"CSV file not found at {csv_path}")
        return None



def plot_boxenplot_metrics(x, y_list, hue, data, palette, output_path, plot_name, method=None, smoothing_method=None,
                           k_depth=5, hue_order=None, name_map=dict()):
    xlabel = name_map.get(x, x.replace('_', ' ').title())
    legend_title = name_map.get(hue, hue.replace('_', ' ').title())
    title = name_map.get(plot_name, plot_name.replace('_', ' ').title())
    if method:
        title += f" - Method: {method}"
    if smoothing_method:
        title += f" - Smoothing: {smoothing_method}"
    for y in y_list:
        ylabel = name_map.get(y, y.replace('_', ' ').title())
        plt.figure(figsize=(10, 6))
        ax = sns.boxenplot(
            x=x, y=y, hue=hue, data=data,
            palette=palette, k_depth=k_depth, hue_order=hue_order
        )

        pair_list = []
        for x_group in data[x].unique():
            group = data[data[x] == x_group]
            hue_values = hue_order if not hue_order == None else group[hue].unique()
            pairs = list(combinations(hue_values, 2))
            for pair in pairs:
                pair_list.append(((x_group, pair[0]), (x_group, pair[1])))  # Correct format for grouped comparison

        annotator = Annotator(ax, pair_list, data=data, x=x, y=y, hue=hue, hue_order=hue_order, verbose=False)
        annotator.configure(
            test='Mann-Whitney', text_format='star', loc='inside',
            hide_non_significant=True,
            pvalue_thresholds=[
                [0.001, '****'],               # P < 0.001: 4 stars
                [0.01, '***'],                 # P < 0.01: 3 stars
                [0.05, '**'],                  # P < 0.05: 2 stars
                [0.1, '*'],                    # P < 0.1: 1 star
                [1, 'ns']                      # Otherwise: 'ns' (not significant)
            ])
        annotator.apply_and_annotate()

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(title=legend_title)
        plt.tight_layout()
        import warnings
        from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        plt.savefig(os.path.join(output_path, f'{method}_{y}_{plot_name}_boxen_plot.png'))
        plt.close()


def statistical_tests(x, y_list, hue, df, output_path, plot_name, method,  alternative='two-sided'):
    if SKIP_STATS == True:
        return
    for y in y_list:
        ylabel = y.replace('_', ' ').title()
        stats_results = []
        for x_group in df[x].unique():
            group = df[df[x] == x_group]
            hue_values = group[hue].unique()
            pairs = list(combinations(hue_values, 2))
            num_tests = len(pairs)
            for pair in pairs:
                errors1 = group[group[hue] == pair[0]][y]
                errors2 = group[group[hue] == pair[1]][y]
                stat_pair, p_value_pair = stats.mannwhitneyu(errors1, errors2, alternative=alternative)
                stats_results.append({
                    x: x_group,
                    'Comparison': f"{pair[0]} vs {pair[1]}",
                    'Test': 'Mann-Whitney U',
                    'Statistic': stat_pair,
                    'Raw p-value': p_value_pair,
                })

        # Save stats results to file
        stats_filename = os.path.join(output_path, f'{method}_{y}_{plot_name}_stats.txt')
        with open(stats_filename, 'w') as f:
            f.write(f'Statistical Test Results for {ylabel}\n')
            f.write(f"Plot Name: {plot_name}\n\n")
            for result in stats_results:
                f.write(f"{x}: {result[x]}\n")
                f.write(f"Comparison: {result['Comparison']}\n")
                f.write(f"Test: {result['Test']}\n")
                f.write(f"Statistic: {result['Statistic']}\n")
                f.write(f"Raw p-value: {result['Raw p-value']}\n")
                f.write('\n')

def generate_full_plots(df, method, output_path, smoothing_method):
    df = df.copy()
    
    x = 'Subject Motion'
    y_list = Y_LIST
    hue = 'Camera Type'
    plot_name = 'full_cmp'
    plot_boxenplot_metrics(
        x=x, y_list=y_list, hue=hue, data=df, plot_name=plot_name, method=method, smoothing_method=smoothing_method,
        palette=[COLOR_STAT_CLOSE,COLOR_STAT_FAR, COLOR_HEAD_STAT, COLOR_HEAD_MOV], output_path=output_path,
        hue_order=[TEXT_STAT_CLOSE, TEXT_STAT_FAR,TEXT_HEAD_STILL, TEXT_HEAD_MOVING],
        name_map={
            'abs_err_fft': 'Absolute Error (FFT)',
            'abs_err_peak': 'Absolute Error (Peak)',
            'macc': 'Maximum Amplitude of Cross Correlation (MACC)',
            'full_cmp': 'Full Comparison'
            })
    
    statistical_tests(
        x=x, y_list=y_list, hue=hue, df=df, output_path=output_path, plot_name=plot_name, method=method
    )


def generate_stationary_camera_plots(df, method, output_path, smoothing_method):
    df = df.copy()

    x = 'Subject Motion'
    y_list = Y_LIST
    hue = 'Camera Type'
    plot_name = 'camera_proximity'
    plot_boxenplot_metrics(
        x=x, y_list=y_list, hue=hue, data=df,
        palette=[COLOR_STAT_CLOSE, COLOR_STAT_FAR], output_path=output_path, plot_name=plot_name, method=method, smoothing_method=smoothing_method,
        hue_order=[TEXT_STAT_CLOSE, TEXT_STAT_FAR],
        name_map={
            'abs_err_fft': 'Absolute Error (FFT)',
            'abs_err_peak': 'Absolute Error (Peak)',
            'macc': 'Maximum Amplitude of Cross Correlation (MACC)',
            'camera_proximity': 'Camera Proximity Comparison',
        })
    

def generate_head_movement_comparison_plots(df, method, output_path, smoothing_method):
    # Filter relevant data
    df = df.copy()


    plot_name = "head_movement"
    y_list = Y_LIST
    x = 'Subject Motion'
    hue = 'Camera Type'
    plot_boxenplot_metrics(
        x=x, y_list=y_list, hue=hue, data=df,
        palette=[COLOR_HEAD_STAT, COLOR_HEAD_MOV], output_path=output_path, plot_name=plot_name, method=method, smoothing_method=smoothing_method,
        hue_order=[TEXT_HEAD_STILL, TEXT_HEAD_MOVING],
        name_map={
            'abs_err_fft': 'Absolute Error (FFT)',
            'abs_err_peak': 'Absolute Error (Peak)',
            'macc': 'Maximum Amplitude of Cross Correlation (MACC)',
            'head_movement': 'Head Movement Comparison',
        })

def generate_boxen_plots(df, method, output_path, smoothing_method):

    df = df.copy()
    
    subject_motion_map = {
        'T1': 'Still',
        'T2': 'Natural Movements',
        'T3': 'Natural Movements',
        'T5': 'Still',
        'T6': 'Natural Movements'
    }
    def map_camera_type(row):
        if row['CAM'] == 'stat' and row['TID'] in ['T1', 'T2','T3']: 
            return TEXT_STAT_CLOSE
        elif row['CAM'] == 'stat' and row['TID'] in ['T5','T6']: 
            return TEXT_STAT_FAR
        elif row['CAM'] == 'head' and row['TID'] in ['T1', 'T2']:
                return TEXT_HEAD_STILL
        elif row['CAM'] == 'head' and row['TID'] in ['T3', 'T5', 'T6']:
                return TEXT_HEAD_MOVING
        else:
            raise ValueError(f"Invalid value: {row['CAM']},{row['TID']}")

    
    df['Subject Motion'] = df['TID'].map(subject_motion_map)
    df['Camera Type'] = df.apply(map_camera_type, axis=1)

    # Calculate absolute errors
    df['abs_err_fft'] = np.abs(df['hr_label_FFT'] - df['hr_pred_FFT'])
    df['abs_err_peak'] = np.abs(df['hr_label_Peak'] - df['hr_pred_Peak'])

    # Ensure no negative values in the calculated absolute errors
    assert (df['abs_err_fft'] >= 0).all(), "Negative values found in abs_err_fft!"
    assert (df['abs_err_peak'] >= 0).all(), "Negative values found in abs_err_peak!"

    # Generate the original plots
    generate_full_plots(df, method, output_path, smoothing_method)

    # Generate the stationary camera plots
    generate_stationary_camera_plots(df, method, output_path, smoothing_method)

    # Generate the head movement comparison plots
    generate_head_movement_comparison_plots(df, method, output_path, smoothing_method)

# Function to generate summary table with proper sorting based on TID and CAM
def generate_summary_table(df):
    # Pivot-like summary based on PID, TID, and CAM
    summary = df.groupby(['TID', 'CAM']).agg({
        'macc': 'mean',
        'hr_label_Peak': lambda x: np.mean(np.abs(x - df['hr_pred_Peak'])),
        'SNR_Peak': 'mean',
        'hr_label_FFT': lambda x: np.mean(np.abs(x - df['hr_pred_FFT'])),
        'SNR_FFT': 'mean'
    }).reset_index()

    # Add summary rows (by TID and overall)
    tid_summary = df.groupby('TID').agg({
        'macc': 'mean',
        'hr_label_Peak': lambda x: np.mean(np.abs(x - df['hr_pred_Peak'])),
        'SNR_Peak': 'mean',
        'hr_label_FFT': lambda x: np.mean(np.abs(x - df['hr_pred_FFT'])),
        'SNR_FFT': 'mean'
    }).reset_index()
    tid_summary['CAM'] = ''  # No camera for TID summary

    # Overall summary row
    overall_summary = pd.DataFrame({
        'TID': [''],
        'CAM': [''],
        'macc': [df['macc'].mean()],
        'hr_label_Peak': [np.mean(np.abs(df['hr_label_Peak'] - df['hr_pred_Peak']))],
        'SNR_Peak': [df['SNR_Peak'].mean()],
        'hr_label_FFT': [np.mean(np.abs(df['hr_label_FFT'] - df['hr_pred_FFT']))],
        'SNR_FFT': [df['SNR_FFT'].mean()]
    })

    # Concatenate the full summary
    full_summary = pd.concat([summary, tid_summary, overall_summary], ignore_index=True)
    
    # Sort by TID first, then by CAM to get the correct order
    full_summary['TID'] = pd.Categorical(
        full_summary['TID'], 
        categories=['T1', 'T2', 'T3', 'T5', 'T6', ''], 
        ordered=True
    )
    full_summary['CAM'] = pd.Categorical(
        full_summary['CAM'], 
        categories=['stat', 'head', ''], 
        ordered=True
    )
    
    # Sort by TID and CAM to ensure correct ordering
    full_summary = full_summary.sort_values(by=['TID', 'CAM']).reset_index(drop=True)
    full_summary = full_summary.rename(columns={"hr_label_Peak": "MAE_Peak", "hr_label_FFT": "MAE_FFT"})
    
    return full_summary

# Function to save plots and summary table
def save_results(df, method, output_path, smoothing_method):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Generate boxen plots
    generate_boxen_plots(df, method, output_path, smoothing_method)

    # Generate summary table
    summary_df = generate_summary_table(df)

    # Save the summary table as a CSV
    summary_df.to_csv(os.path.join(output_path, f'{method}_summary_table.csv'), index=False)
    print(f"Results saved for method: {method}")

# Main function to process each method from the CSV file and generate plots and summary tables
def main():
    
    for smoothing_method, path in PATHS:
        
        # Step 1: Load the generated CSV file
        df = load_csv(path)
        
        if df is not None:
            # Step 2: Define the output path for this method
            output_path = os.path.join(path, 'aggregated_results')
            print(smoothing_method)
            
            # Step 3: For each distinct 'method' in the CSV, save the plots and summary table
            for method, df_method in df.groupby('method'):
                print(f"Processing method: {method}")
                save_results(df_method, method, output_path,smoothing_method)
            
            # Optionally save the combined results for all methods under "all"
            if len(df['method'].unique()) > 1:
                save_results(df, 'all', output_path,smoothing_method) 
        else:
            print(f"Skipping file: {path}, as CSV file was not found.")


# Run the main function
if __name__ == "__main__":
    main()
