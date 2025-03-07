#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import colorsys

from scipy.ndimage.filters import gaussian_filter1d

# Configuration
EXPERIMENT_GROUPS = ['rotations-2k-samples', 'noise-undersampling-2k-samples', 'PointNeXt_XXL-rotations-2k-samples']
#EXPERIMENT_GROUPS = ['PointNeXt_XXL-rotations-2k-samples']
'''
CLASSES = ['astroid', 'citrus', 'cylinder', 'egg_keplero', 'geometric_petal', 
           'lemniscate', 'm_convexities', 'mouth_curve', 'revolution', 'square']
'''
CLASSES = ['astroid', 'citrus', 'cylinder', 'egg_keplero', 'geometric_petal', 'lemniscate', ]
BASE_DIR = '../../logs'
CLIP_LOSS_AT = 2  # Clip loss at this value
SMOOTH_SIGMA = 2
ENABLE_NOISE_UNDERSAMPLING_DASHED_LINES = False

def parse_run_name(run_name, experiment_group):
	parts = run_name.split('-')
	if 'noise-undersampling' in experiment_group:
		# e.g. symmetria-ablation-noise-undersampling-2k-samples-astroid-2000-transform_prob-1.0-transform-gaussian
		transform_prob = None
		transform_type = None
		for i in range(len(parts)):
			if parts[i] == 'transform_prob':
				transform_prob = float(parts[i+1])
			if parts[i] == 'transform' and i+1 < len(parts):
				transform_type = parts[i+1]
		friendly_name = f"Transform prob. {transform_prob} - {transform_type}" if transform_prob > 0.0 else f"Transform prob. {transform_prob} (clean)"
		params = {
			'prob': transform_prob,
			'type': transform_type
		}
	elif 'PointNeXt_XXL' in experiment_group:
		# e.g. symmetria-ablation-PointNeXt_XXL-rotations-2k-samples-lemniscate-2000-rotprob-0.4-rotx-true-roty-true-rotz-true
		rot_prob = None
		axes = []
		for i in range(len(parts)):
			if parts[i] == 'rotprob':
				rot_prob = float(parts[i+1])
			if parts[i].startswith('rot') and parts[i] != 'rotprob' and i+1 < len(parts):
				axis = parts[i]
				state = parts[i+1]
				if state == 'true':
					axes.append(axis.replace('rot', ''))
		friendly_name = f"Rotation prob. {rot_prob} - axes: {', '.join(axes)}" if axes else f"Rotation prob. {rot_prob} (clean)"
		params = {
			'prob': rot_prob,
			'axes': axes
		}
	elif 'rotations' in experiment_group:
		# e.g. symmetria-ablation-rotations-2k-samples-astroid-2000-rotprob-0.8-rotx-true-roty-true-rotz-false
		rot_prob = None
		axes = []
		for i in range(len(parts)):
			if parts[i] == 'rotprob':
				rot_prob = float(parts[i+1])
			if parts[i].startswith('rot') and parts[i] != 'rotprob' and i+1 < len(parts):
				axis = parts[i]
				state = parts[i+1]
				if state == 'true':
					axes.append(axis.replace('rot', ''))
		friendly_name = f"Rotation prob. {rot_prob} - axes: {','.join(axes)}" if axes else f"Rotation prob. {rot_prob} (clean)"
		params = {
			'prob': rot_prob,
			'axes': axes
		}
	else:
		friendly_name = run_name
		params = {}
	return friendly_name, params

def get_color(experiment_group, params):
	if 'PointNeXt_XXL' in experiment_group:
		prob		= params['prob']
		axes		= params['axes']
		color_var	= axes
		variations	= [['x'], ['x', 'y'], ['x', 'y', 'z']]
	elif 'rotations' in experiment_group:
		prob		= params['prob']
		axes		= params['axes']
		color_var	= axes
		variations	= [['x'], ['x', 'y'], ['x', 'y', 'z']]
	elif 'noise-undersampling' in experiment_group:
		prob		= params['prob']
		transform_type	= params['type']
		color_var	= transform_type
		variations	= ['undersampling', 'gaussian', 'uniform']
	else:
		return (0, 0, 0)  # Fallback

	if prob == 0.0:
		#return (0.2, 0.2, 0.2)  # Gray
		# for name, hex in matplotlib.colors.cnames.items():
		return '#DDB700' # a darker gold
	else:
		if color_var == variations[0]:
			hue = 120  # Green
		elif color_var == variations[1]:
			hue = 240  # Blue
		elif color_var == variations[2]:
			hue = 0  # Red
		else:
			hue = 180  # Cyan for others
		value      = 0.1 + 0.9 * prob
		saturation = 0.5 + 0.5 * prob
		r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
		return (r, g, b)
	return (0, 0, 0)  # Fallback

def load_metrics(metrics_file):
	if metrics_file.exists():
		return pd.read_csv(metrics_file)
	else:
		return None

def process_class(experiment_group, class_name, clip_loss_at=-1, smooth_sigma=-1):
	class_dir = Path(BASE_DIR) / experiment_group / class_name
	if not class_dir.exists():
		print(f'No runs found for {experiment_group} - {class_name} in {class_dir}')
		return

	runs = []
	for run_dir in class_dir.iterdir():
		if not run_dir.is_dir():
			continue
		print(f'Collecting run: {run_dir}...')
		if run_dir.is_dir() and run_dir.name.startswith('symmetria-ablation'):
			runs.append(run_dir)

	if not runs:
		print(f'No runs found for {experiment_group} - {class_name}')
		return
	else:
		print(f'Found {len(runs)} runs for {experiment_group} - {class_name}')

	fig, (ax_loss, ax_map, ax_phc) = plt.subplots(1, 3, figsize=(18, 6))
	if 'noise-undersampling' in str(runs[0]):
		exp_title = f"noise/undersampling (PointNet) - 2k samples"
	elif 'PointNeXt_XXL' in str(runs[0]):
		exp_title = f"rotations (PointNeXt XXL) - 2k samples"
	else:
		exp_title = f"rotations (PointNet) - 2k samples"
	fig.suptitle(f"Experiment: {exp_title}, Class: {class_name}")

	ax_loss.set_ylabel('Validation Loss'+ f' (clipped at {CLIP_LOSS_AT})' if CLIP_LOSS_AT > 0 else '')
	ax_map.set_ylabel('Val. mAP')
	ax_phc.set_ylabel('Val. PHC')

	handles, labels = [], []

	if 'noise-undersampling' in str(runs[0]):
		tmp_runs = sorted(runs, key=lambda x: str(x).split('-')[-1])
	else:
		tmp_runs = sorted(runs, key=lambda x: parse_run_name(str(x), experiment_group)[1]['axes'])

	runs = tmp_runs[1:]
	runs.append(tmp_runs[0])		# we want the "clean" run to be last

	max_loss_vals = [-1] * len(runs)
	for run_idx, run_dir in enumerate(runs):
		print(f'Processing run: {run_dir}...')
		run_name = run_dir.name
		friendly_name, params = parse_run_name(run_name, experiment_group)
		print(f'============================================================== {friendly_name = }')
		color = get_color(experiment_group, params)

		metrics_file = run_dir / 'version_0' / 'metrics.csv'
		df = load_metrics(metrics_file)
		if df is None:
			print(f'No metrics found in {metrics_file}')
			continue

		epoch = df['epoch'].values
		loss = df['plane_val_loss_epoch'].values
		map_vals = df['plane_val_map_epoch'].values
		phc_vals = df['plane_val_phc_epoch'].values

		print(f'{epoch.shape = } - {loss.shape = } - {map_vals.shape = } - {phc_vals.shape = }')

		valid = ~np.isnan(loss) & ~np.isnan(map_vals) & ~np.isnan(phc_vals)
		x = epoch[valid]
		if len(x) == 0:
			continue

		if clip_loss_at > 0:
			loss_filtered  = np.clip(loss[valid], None, CLIP_LOSS_AT)
		else:
			loss_filtered = loss[valid]
		map_filtered  = map_vals[valid]
		phc_filtered  = phc_vals[valid]

		if smooth_sigma > 0:
			loss_filtered = gaussian_filter1d(loss_filtered, sigma=smooth_sigma)
			map_filtered  = gaussian_filter1d(map_filtered , sigma=smooth_sigma)
			phc_filtered  = gaussian_filter1d(phc_filtered , sigma=smooth_sigma)

		max_loss_vals[run_idx] = np.max(loss_filtered)

		if 'noise-undersampling' in str(runs[0]) and ENABLE_NOISE_UNDERSAMPLING_DASHED_LINES:
			if params['prob'] == 0.0:
				dashes = (None, None)
			else:
				pp = int(params['prob'] * 16)
				dashes = [pp, pp/4, pp/2, pp/4, pp/2, pp/4]
			#if params['type'] == 0.5:
		else:
			dashes = (None, None)
		print(f'{dashes = }')
		line, = ax_loss.plot(x, loss_filtered, color=color, linewidth=3, label=friendly_name, dashes=dashes)
		ax_map.plot(x,		map_filtered,  color=color, linewidth=3, label=friendly_name, dashes=dashes)
		ax_phc.plot(x,		phc_filtered,  color=color, linewidth=3, label=friendly_name, dashes=dashes)

		handles.append(line)
		labels.append(friendly_name)

	for ax_idx, ax in enumerate([ax_loss, ax_map, ax_phc]):
		ax.set_xlabel('Epochs')
		ax.grid(True)
		if ax_idx == 0:
			if clip_loss_at > 0:
				ax.set_ylim(-0.004, clip_loss_at + 0.01*clip_loss_at)
			else:
				ax.set_ylim(-0.004, max_loss_vals[ax_idx] + 0.01*max_loss_vals[ax_idx])
		else:
			ax.set_ylim(-0.05, 1.05)

	fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title='Runs')
	plt.tight_layout()
	plt.savefig(f"{experiment_group}-{class_name}.png", bbox_inches='tight')
	plt.close()

for experiment_group in EXPERIMENT_GROUPS:
	for class_name in CLASSES:
		print(f"Processing {experiment_group} - {class_name}")
		process_class(experiment_group, class_name, clip_loss_at=CLIP_LOSS_AT, smooth_sigma=SMOOTH_SIGMA)
