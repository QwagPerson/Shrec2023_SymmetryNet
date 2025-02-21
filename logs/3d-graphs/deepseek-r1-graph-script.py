#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path

# Set an interactive backend for matplotlib
plt.switch_backend('Qt5Agg')  # Use 'Qt5Agg' or 'MacOSX' if TkAgg doesn't work

# Configuration
EXPERIMENT_GROUPS = ['rotations', 'noise-undersampling']
#CLASSES = ['astroid', 'citrus', 'cylinder', 'egg_keplero', 'geometric_petal', 'lemniscate', 'm_convexities', 'mouth_curve', 'revolution', 'square']
CLASSES = ['cylinder', 'egg_keplero', 'geometric_petal', 'lemniscate', 'm_convexities', 'mouth_curve', 'revolution', 'square']
BASE_DIR = 'logs'
CLIP_LOSS_AT = 5  # Clip loss at this value
USE_EPOCHS = True  # Use epochs instead of steps for the x-axis

# Columns to extract from the metrics.csv files
columns_dict = {
			'step':						[30, 'Samples'],
			'epoch':					[ 1, 'Epoch'],

			'plane_train_loss_confidence_step':		[ 3, 'plane_train_loss_confidence_step'],
			'plane_train_loss_distance_step':		[ 5, 'plane_train_loss_distance_step'],
			'plane_train_loss_normal_step':			[ 8, 'plane_train_loss_normal_step'],
			'plane_train_loss_ref_sym_distance_step':	[10, 'plane_train_loss_ref_sym_distance_step'],
			'plane_train_loss_step':			[11, 'plane_train_loss_step'],
			'plane_train_map_step':				[13, 'plane_train_map_step'],
			'plane_train_phc_step':				[15, 'plane_train_phc_step'],

			'total_train_loss_step':			[32, 'total_train_loss_step'],

			'plane_train_loss_confidence_epoch':		[ 2, 'plane_train_loss_confidence_epoch'],
			'plane_train_loss_distance_epoch':		[ 4, 'plane_train_loss_distance_epoch'],
			'plane_train_loss_epoch':			[ 6, 'Loss'],
			'plane_train_loss_normal_epoch':		[ 7, 'plane_train_loss_normal_epoch'],
			'plane_train_loss_ref_sym_distance_epoch':	[ 9, 'plane_train_loss_ref_sym_distance_epoch'],
			'plane_train_map_epoch':			[12, 'mAP'],
			'plane_train_phc_epoch':			[14, 'PHC'],

			'plane_val_loss_confidence_epoch':		[16, 'plane_val_loss_confidence_epoch'],
			'plane_val_loss_confidence_step':		[17, 'plane_val_loss_confidence_step'],
			'plane_val_loss_distance_epoch':		[18, 'plane_val_loss_distance_epoch'],
			'plane_val_loss_distance_step':			[19, 'plane_val_loss_distance_step'],
			'plane_val_loss_epoch':				[20, 'Loss'],
			'plane_val_loss_normal_epoch':			[21, 'plane_val_loss_normal_epoch'],
			'plane_val_loss_normal_step':			[22, 'plane_val_loss_normal_step'],
			'plane_val_loss_ref_sym_distance_epoch':	[23, 'plane_val_loss_ref_sym_distance_epoch'],
			'plane_val_loss_ref_sym_distance_step':		[24, 'plane_val_loss_ref_sym_distance_step'],
			'plane_val_loss_step':				[25, 'plane_val_loss_step'],
			'plane_val_map_epoch':				[26, 'mAP'],
			'plane_val_map_step':				[27, 'plane_val_map_step'],
			'plane_val_phc_epoch':				[28, 'PHC'],
			'plane_val_phc_step':				[29, 'plane_val_phc_step'],

			'total_train_loss_epoch':			[31, 'total_train_loss_epoch'],

			'total_val_loss_epoch':				[33, 'total_val_loss_epoch'],

			'total_val_loss_step':				[34, 'total_val_loss_step'],
}

# Function to load metrics from a directory
def load_metrics(directory):
    metrics_file = os.path.join(directory, 'metrics.csv')
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        return df
    return None

# Function to extract relevant data from the metrics
def extract_data(df, use_epochs=True):
	if use_epochs:
		x_axis   = 'epoch'
		loss_col = 'plane_val_loss_epoch'
		map_col  = 'plane_val_map_epoch'
		phc_col  = 'plane_val_phc_epoch'
	else:
		x_axis   = 'step'
		loss_col = 'plane_val_loss_step'
		map_col  = 'plane_val_map_step'
		phc_col  = 'plane_val_phc_step'

	x    = df[x_axis].values
	loss = df[loss_col].values
	map_ = df[map_col].values
	phc  = df[phc_col].values
	
	# Clip loss values
	loss = np.clip(loss, None, CLIP_LOSS_AT)
	
	#print(f'{x[:5] = } - {loss[:5] = } - {map_[:5] = } - {phc[:5]= }')
	loss_non_null = np.logical_not(np.isnan(loss))
	map_non_null = np.logical_not(np.isnan(map_))
	phc_non_null = np.logical_not(np.isnan(phc))
	x    = x[loss_non_null & map_non_null & phc_non_null]
	loss = loss[loss_non_null & map_non_null & phc_non_null]
	map_ = map_[loss_non_null & map_non_null & phc_non_null]
	phc  = phc[loss_non_null & map_non_null & phc_non_null]

	return x, loss, map_, phc

# Function to plot 3D graphs
def plot_3d_graphs(experiment_group, class_data):
    fig = plt.figure(figsize=(18, 6))
    
    # Plot for Loss
    ax1 = fig.add_subplot(131, projection='3d')
    for class_name, data in class_data.items():
        x, loss, _, _ = data
        ax1.plot(x, [CLASSES.index(class_name)] * len(x), loss, label=class_name)
    ax1.set_xlabel('Epochs' if USE_EPOCHS else 'Steps')
    ax1.set_ylabel('Class')
    ax1.set_zlabel('Loss')
    ax1.set_title(f'{experiment_group} - Validation Loss')
    
    # Plot for mAP
    ax2 = fig.add_subplot(132, projection='3d')
    for class_name, data in class_data.items():
        x, _, map_, _ = data
        ax2.plot(x, [CLASSES.index(class_name)] * len(x), map_, label=class_name)
    ax2.set_xlabel('Epochs' if USE_EPOCHS else 'Steps')
    ax2.set_ylabel('Class')
    ax2.set_zlabel('mAP')
    ax2.set_title(f'{experiment_group} - Validation mAP')
    
    # Plot for PHC
    ax3 = fig.add_subplot(133, projection='3d')
    for class_name, data in class_data.items():
        x, _, _, phc = data
        ax3.plot(x, [CLASSES.index(class_name)] * len(x), phc, label=class_name)
    ax3.set_xlabel('Epochs' if USE_EPOCHS else 'Steps')
    ax3.set_ylabel('Class')
    ax3.set_zlabel('PHC')
    ax3.set_title(f'{experiment_group} - Validation PHC')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Main function to process each experiment group
def process_experiment_group(experiment_group, debug=False, max_points=10):
	class_data = {}
	for class_name in CLASSES:
		#class_dir = os.path.join(BASE_DIR, experiment_group, class_name)
		class_dir = Path(BASE_DIR) / experiment_group / class_name
		print(f'Processing {class_name} in {class_dir}...')
		#if os.path.exists(class_dir):
		experiment_class = Path(experiment_group) / class_name
		if (experiment_class).is_dir():
			for root, dirs, files in os.walk(str(experiment_class)):		#, topdown=False):
				if debug:
					print(f'{root = } - {dirs = } - {files = }')
				for run_idx, run_dir in enumerate(dirs):
					print(f'Processing run no.: {run_idx} in {run_dir}...')
					experiment_run = Path(experiment_class) / run_dir
					if experiment_run.is_dir():
						print(f'Loading metrics in {run_dir}...')
						run_path = experiment_run / 'version_0'
						df = load_metrics(run_path)
						if df is not None:
							if debug:
								print(f'{df.head(5) = }')
								print(f'{class_name = }')
								print(f'{len(df) = }')
							x, loss, map_, phc = extract_data(df, USE_EPOCHS)
							if class_name not in class_data:
								class_data[class_name] = (x[:max_points], loss[:max_points], map_[:max_points], phc[:max_points])
							else:
								# Average the metrics if there are multiple runs
								old_x, old_loss, old_map, old_phc = class_data[class_name]
								if debug:
									print(f'{x.shape = } - {old_x.shape = }')
									print(f'{loss.shape = } - {old_loss.shape = }')
									print(f'{map_.shape = } - {old_map.shape = }')
									print(f'{phc.shape = } - {old_phc.shape = }')
								class_data[class_name] = (x[:max_points], (old_loss + loss[:max_points]) / 2, (old_map + map_[:max_points]) / 2, (old_phc + phc[:max_points]) / 2)
	
	plot_3d_graphs(experiment_group, class_data)

# Process each experiment group
for experiment_group in EXPERIMENT_GROUPS:
    process_experiment_group(experiment_group)
