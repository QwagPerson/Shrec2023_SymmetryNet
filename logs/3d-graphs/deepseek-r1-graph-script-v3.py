#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from collections import defaultdict
import matplotlib.cm as cm

# Set an interactive backend for matplotlib
plt.switch_backend('Qt5Agg')

# Configuration
EXPERIMENT_GROUPS = ['rotations', 'noise-undersampling']
CLASSES = ['cylinder', 'egg_keplero', 'geometric_petal', 'lemniscate', 
          'm_convexities', 'mouth_curve', 'revolution', 'square']
BASE_DIR = 'logs'
CLIP_LOSS_AT = 5
USE_EPOCHS = True
DEBUG = True

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

def parse_run_name(run_name, debug=False):
	"""Extract parameters from run directory name"""
	parts = run_name.split('-')
	if debug:
		print(f'{parts = }')
	if 'noise-undersampling' in run_name:
		return {
			'class': parts[4],
			'prob': float(parts[6].split('_')[-1]),
			'transform': parts[-1]
		}
	elif 'rotations' in run_name or 'rotprob' in run_name:
		return {
			'class': parts[2],
			'prob': float(parts[5].split('_')[-1]),
			'axes': [p.split('-')[0] for p in parts[6:] if 'true' in p]
		}
	return {}

def get_color_and_label(params, experiment_type):
    """Generate color and friendly label based on parameters"""
    if experiment_type == 'noise-undersampling':
        label = f"{params['class']}-prob={params['prob']:.1f}-{params['transform']}"
        # Color by transform type
        colors = {'gaussian': 'red', 'uniform': 'blue', 'undersampling': 'green', 'clean': 'gray'}
        base_color = colors.get(params['transform'], 'black')
        alpha = 0.5 + (params['prob'] * 0.5)  # Higher prob = more opaque
    else:
        axes_str = '-'.join(params['axes'])
        label = f"{params['class']}-prob={params['prob']:.1f}-{axes_str}"
        # Color by rotation axes combination
        color_map = {'rotx': 0, 'roty': 1, 'rotz': 2}
        hue = sum(color_map.get(ax, 0) for ax in params['axes']) / 3
        base_color = cm.rainbow(hue)
        alpha = 0.3 + (params['prob'] * 0.7)  # Higher prob = more opaque
    
    return label, (*base_color[:3], alpha) if isinstance(base_color, tuple) else (base_color, alpha)

def plot_3d_graphs(experiment_group, class_data, debug=False):
	"""Plot 3D graphs for each class separately"""
	for class_name, runs in class_data.items():
		fig = plt.figure(figsize=(18, 6))
		fig.suptitle(f'{experiment_group} - {class_name}', fontsize=16)

		if debug:
			print(f'{class_name = } - {runs = }')
		
		axes = [
			fig.add_subplot(131, projection='3d'),
			fig.add_subplot(132, projection='3d'),
			fig.add_subplot(133, projection='3d')
		]
		titles = ['Validation Loss', 'Validation mAP', 'Validation PHC']
		
		for run_idx, run_data in enumerate(runs):
			if debug:
				print(f'{class_name = } - Run[{run_idx}]: {run_data = }')
				#print(f'{runs.index(run_data) = }')
			x, loss, map_, phc, label, color = run_data
			#y_pos = np.full_like(x, runs.index(run_data))  # Unique Y position per run
			y_pos = np.full_like(x, run_idx)  # Unique Y position per run
			
			for ax_idx, (ax, values) in enumerate(zip(axes, [loss, map_, phc])):
				ax.plot(x, y_pos, values, color=color, label=label if ax_idx == 0 else None)
				ax.set_title(titles[ax_idx])
				ax.set_xlabel('Epochs' if USE_EPOCHS else 'Steps')
				ax.set_ylabel('Run')
				ax.set_zlabel(titles[ax_idx].split()[-1])
		
		handles, labels = axes[0].get_legend_handles_labels()
		fig.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
		plt.tight_layout()
		plt.show()

def process_experiment_group(experiment_group, debug=False):
	for class_name in CLASSES:
		#class_dir = Path(BASE_DIR) / experiment_group / class_name
		class_dir = Path(experiment_group) / class_name
		print(f'Processing {class_name} in {class_dir}...')
		
		if not class_dir.exists():
			print(f"Directory not found: {class_dir}")
			continue
			
		class_runs = []
		for root, dirs, files in os.walk(str(class_dir)):
			if debug:
				print(f'{root = } - {dirs = } - {files = }')
			
			for run_dir in dirs:
				run_path = Path(root) / run_dir / 'version_0'
				if debug:
					print(f'Processing run: {run_path}')
				
				df = load_metrics(run_path)
				if df is None:
					continue
				
				# Extract metrics
				x, loss, map_, phc = extract_data(df, USE_EPOCHS)
				
				# Parse run parameters
				params = parse_run_name(run_dir, debug=debug)
				if debug:
					print(f'{params = }')
				label, color = get_color_and_label(params, experiment_group)
				
				class_runs.append((x, loss, map_, phc, label, color))
		
		# Plot for this class
		if class_runs:
			plot_3d_graphs(experiment_group, {class_name: class_runs}, debug=debug)

# Process each experiment group
for experiment_group in EXPERIMENT_GROUPS:
    process_experiment_group(experiment_group, debug=DEBUG)
