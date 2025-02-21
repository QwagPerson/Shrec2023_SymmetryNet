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
CLASSES = ['cylinder', 'egg_keplero', 'geometric_petal', 'lemniscate', 'm_convexities', 'mouth_curve', 'revolution', 'square']
BASE_DIR = 'logs'
CLIP_LOSS_AT = 5  # Clip loss at this value
USE_EPOCHS = True  # Use epochs instead of steps for the x-axis

# Columns to extract from the metrics.csv files
columns_dict = {
    'step': [30, 'Samples'],
    'epoch': [1, 'Epoch'],
    'plane_val_loss_epoch': [20, 'Loss'],
    'plane_val_map_epoch': [26, 'mAP'],
    'plane_val_phc_epoch': [28, 'PHC'],
    'plane_val_loss_step': [25, 'Loss'],
    'plane_val_map_step': [27, 'mAP'],
    'plane_val_phc_step': [29, 'PHC'],
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
        x_axis = 'epoch'
        loss_col = 'plane_val_loss_epoch'
        map_col = 'plane_val_map_epoch'
        phc_col = 'plane_val_phc_epoch'
    else:
        x_axis = 'step'
        loss_col = 'plane_val_loss_step'
        map_col = 'plane_val_map_step'
        phc_col = 'plane_val_phc_step'

    x = df[x_axis].values
    loss = df[loss_col].values
    map_ = df[map_col].values
    phc = df[phc_col].values

    # Clip loss values
    loss = np.clip(loss, None, CLIP_LOSS_AT)

    # Filter out rows where any of the metrics are NaN
    mask = ~np.isnan(loss) & ~np.isnan(map_) & ~np.isnan(phc)
    x = x[mask]
    loss = loss[mask]
    map_ = map_[mask]
    phc = phc[mask]

    return x, loss, map_, phc

# Function to generate a "friendly" name for the run
def generate_friendly_name(run_name):
    if 'noise-undersampling' in run_name:
        parts = run_name.split('-')
        class_name = parts[4]
        prob = parts[6].split('_')[-1]
        transform = parts[-1]
        return f"{class_name}-prob={prob}-{transform}"
    elif 'rotations' in run_name or 'rotprob' in run_name:
        parts = run_name.split('-')
        class_name = parts[2]
        prob = parts[4].split('_')[-1]
        rotations = []
        if 'rotx-true' in run_name:
            rotations.append('rotx')
        if 'roty-true' in run_name:
            rotations.append('roty')
        if 'rotz-true' in run_name:
            rotations.append('rotz')
        return f"{class_name}-prob={prob}-{'-'.join(rotations)}"
    return run_name

# Function to plot 3D graphs for a single class
def plot_3d_graphs_for_class(experiment_group, class_name, runs_data):
	fig = plt.figure(figsize=(18, 6))
	fig.suptitle(f'{experiment_group} - {class_name}', fontsize=16)

	# Plot for Loss
	ax1 = fig.add_subplot(131, projection='3d')
	for run_name, data in runs_data.items():
		x, loss, _, _ = data
		friendly_name = generate_friendly_name(run_name)
		ax1.plot(x, [list(runs_data.keys()).index(run_name)] * len(x), loss, label=friendly_name)
	ax1.set_xlabel('Epochs' if USE_EPOCHS else 'Steps')
	ax1.set_ylabel('Run')
	ax1.set_zlabel('Loss')
	ax1.set_title('Validation Loss')

	# Plot for mAP
	ax2 = fig.add_subplot(132, projection='3d')
	for run_name, data in runs_data.items():
		print(f'{run_name = }')
		x, _, map_, _ = data
		friendly_name = generate_friendly_name(run_name)
		ax2.plot(x, [list(runs_data.keys()).index(run_name)] * len(x), map_, label=friendly_name)
	ax2.set_xlabel('Epochs' if USE_EPOCHS else 'Steps')
	ax2.set_ylabel('Run')
	ax2.set_zlabel('mAP')
	ax2.set_title('Validation mAP')

	# Plot for PHC
	ax3 = fig.add_subplot(133, projection='3d')
	for run_name, data in runs_data.items():
		x, _, _, phc = data
		friendly_name = generate_friendly_name(run_name)
		ax3.plot(x, [list(runs_data.keys()).index(run_name)] * len(x), phc, label=friendly_name)
	ax3.set_xlabel('Epochs' if USE_EPOCHS else 'Steps')
	ax3.set_ylabel('Run')
	ax3.set_zlabel('PHC')
	ax3.set_title('Validation PHC')

	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	fig.savefig(f'{experiment_group}-{class_name}.png')
	plt.show()

# Main function to process each experiment group
def process_experiment_group(experiment_group, debug=False):
	for class_name in CLASSES:
		class_dir = Path(BASE_DIR) / experiment_group / class_name
		print(f'Processing {class_name} in {class_dir}...')
		experiment_class = Path(experiment_group) / class_name
		print(f'{experiment_class = }')
		if (experiment_class).is_dir():
			for root, dirs, files in os.walk(str(experiment_class)):				#, topdown=False):
				if debug:
					print(f'{root = } - {dirs = } - {files = }')
				runs_data = {}
				for run_idx, run_dir in enumerate(dirs):
					print(f'Processing run no.: {run_idx} in {run_dir}...')
					experiment_run = Path(experiment_class) / run_dir
					if experiment_run.is_dir():
						print(f'Loading metrics in {run_dir}...')
						run_path = experiment_run / 'version_0'
						df = load_metrics(run_path)
						if df is not None:
							x, loss, map_, phc = extract_data(df, USE_EPOCHS)
							runs_data[run_dir] = (x, loss, map_, phc)
				if runs_data:
					plot_3d_graphs_for_class(experiment_group, class_name, runs_data)
















		'''
		if class_dir.is_dir():
			runs_data = {}
			for run_dir in os.listdir(class_dir):
				run_path = class_dir / run_dir / 'version_0'
				df = load_metrics(run_path)
				if df is not None:
					x, loss, map_, phc = extract_data(df, USE_EPOCHS)
					runs_data[run_dir] = (x, loss, map_, phc)
			if runs_data:
				plot_3d_graphs_for_class(experiment_group, class_name, runs_data)
		'''

# Process each experiment group
for experiment_group in EXPERIMENT_GROUPS:
    process_experiment_group(experiment_group)
