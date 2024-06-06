#!/usr/bin/env python3

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from colors import colors, hex_to_rgba, select_color, apply_wandb_graph_style

from scipy.ndimage.filters import gaussian_filter1d

import argparse
argparse = argparse.ArgumentParser()

'''
├── easy
│   ├── 100k
│   │   ├── config.yaml
│   │   ├── hparams.yaml
│   │   └── metrics.csv
│   └── 10k
│       ├── config.yaml
│       ├── hparams.yaml
│       └── metrics.csv
├── hard
│   ├── 100k
│   │   ├── config.yaml
│   │   ├── hparams.yaml
│   │   └── metrics.csv
│   └── 10k
│       ├── config.yaml
│       ├── hparams.yaml
│       └── metrics.csv
├── intermediate-1
│   ├── 100k
│   │   ├── config.yaml
│   │   ├── hparams.yaml
│   │   └── metrics.csv
│   └── 10k
│       ├── config.yaml
│       ├── hparams.yaml
│       ├── metrics.csv
│       └── metrics.csv
└── intermediate-2
    ├── 100k
    │   ├── config.yaml
    │   ├── hparams.yaml
    │   └── metrics.csv
    └── 10k
        ├── config.yaml
        ├── hparams.yaml
        └── metrics.csv
'''

def print_x_y_lists(x_vals, y_vals, n=3):
	for x, y in zip(x_vals[:n], y_vals[:n]):
		print(x, y, sep='\t\t')
	for x, y in zip(x_vals[-n:], y_vals[-n:]):
		print(x, y, sep='\t\t')

def place_legend(args_legend=None):
	legend_loc = None
	if args_legend:
		if args_legend == 'll' or args_legend == 'bl':
			legend_loc = 'lower left'
		elif args_legend == 'lr' or args_legend == 'br':
			legend_loc = 'lower right'
		elif args_legend == 'tl' or args_legend == 'ul':
			legend_loc = 'upper left'
		elif args_legend == 'tr' or args_legend == 'ur':
			legend_loc = 'upper right'
		elif args_legend == 'tc' or args_legend == 'uc':
			legend_loc = 'upper center'
		elif args_legend == 'bc' or args_legend == 'lc':
			legend_loc = 'lower center'
		elif args_legend == 'cl' or args_legend == 'cl':
			legend_loc = 'center left'
		elif args_legend == 'cr' or args_legend == 'cr':
			legend_loc = 'center right'
	else:
		legend_loc = 'lower right'
	return legend_loc

argparse.add_argument('dir' , type=str, help='Metrics directory')	# positionals are always required
argparse.add_argument('x'   , type=str, help='x axis')			# e.g. "epoch" or "step"
argparse.add_argument('y1'  , type=str, help='y axis no. 1')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y2', type=str, help='y axis no. 2')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y3', type=str, help='y axis no. 3')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y4', type=str, help='y axis no. 4')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y5', type=str, help='y axis no. 5')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y6', type=str, help='y axis no. 6')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--legend', type=str, help='Where to place the legend (e.g. lr, tl, etc.)')		# e.g. "lr = lower right, etc."
argparse.add_argument('--out_fn', type=str, help='Output filename')
argparse.add_argument('--smooth_sigma', type=int, default=2, help='Smooth segmented lines using splines with sigma = <smooth_sigma>')
args = argparse.parse_args()

# /mnt/btrfs-big/dataset/geometric-primitives-classification/neurips-final/metrics/andrea/easy/10k/metrics.csv

sizes = ['10k', '100k']
difficulties = ['easy', 'intermediate-1', 'intermediate-2', 'hard']

'''
sizes = ['10k'] #, '100k']
difficulties = ['easy'] # , 'intermediate-1', 'intermediate-2', 'hard']
'''

metrics = {'10k': {}, '100k': {}}
for size in sizes:
	for difficulty in difficulties:
		print(f'------------------------------------------')
		print(f'------------------------------------------')
		print(f'Reading metrics for: {size} - {difficulty}')
		print(f'------------------------------------------')
		print(f'------------------------------------------')
		fn = f'{args.dir}/{difficulty}/{size}/metrics.csv'
		print(f'{fn = }')
		if not Path(fn).is_file():
			print(f'File not found: {fn}')
			break
		else:
			metrics[size][difficulty] = {'fn': fn, 'metrics': None}
			metric = pd.read_csv(fn)
			metrics[size][difficulty]['metrics'] = metric
print(f'{metrics = }')

# Set fivethirtyeight style
plt.style.use('fivethirtyeight')

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

#selected_columns = ['step', 'plane_train_map_step'] # almost all 1.0 since the beginning
#selected_columns = ['step', 'plane_train_loss_step']
selected_columns = [args.x, args.y1]
if args.y2:
	selected_columns.append(args.y2)
if args.y3:
	selected_columns.append(args.y3)
if args.y4:
	selected_columns.append(args.y4)
if args.y5:
	selected_columns.append(args.y5)
if args.y6:
	selected_columns.append(args.y6)

out_fn = args.out_fn
print(f'Generating {out_fn} for column {selected_columns[1]}')

max_epochs = 40
y_maxes = []
for sz in sizes:
	for col in selected_columns:
		for diff in difficulties:
			metric		= metrics[sz][diff]['metrics']
			df		= metric[metric[col].notnull()]
			x_vals		= df[selected_columns[0]]
			idx_max_min	= df[col].idxmax() if not 'loss' in col else df[col].idxmin()
			df_max_min	= df[col].max() if not 'loss' in col else df[col].min()
			x_vals_max_min	= x_vals[idx_max_min]
			max_min_str	= 'max' if not 'loss' in col else 'min'
			y_maxes.append(f'Symmetria {diff}-{sz} - {col} has {max_min_str}: {df_max_min} at idx: {x_vals_max_min}')
for ymax in y_maxes:
	print(f'{ymax}')
#sys.exit(0)


for sz in sizes:
	for col in selected_columns:
		max_y_val = -1
		max_x_val = -1
		label     = ''
		#plt.clf()
		fig, ax = plt.subplots()
		# Who knows what's this for...
		ax.tick_params(axis='x', which='major', bottom=True)

		diff_counter = 0
		for diff in difficulties:	# we want to see all the (e.g.) mAP curves for all the 10k datasets: easy, intermediate-1, intermediate-2, hard
			label = columns_dict[col][1]
			run = 'Training' if 'train' in selected_columns[1] else 'Validation'
			out_fn = f'symmetria-{size}-{run.lower()}-{label.lower()}.png'

			print(f'##########################################')
			print(f'##########################################')
			print(f'Generating graph for: {size} - {difficulty} - {col} - {out_fn}')
			print(f'##########################################')
			print(f'##########################################')

			if sz not in metrics or diff not in metrics[sz]:
				continue
			metric = metrics[sz][diff]['metrics']

			difficulty = 'Easy' if 'easy' in diff else 'Hard' if 'hard' in diff else 'Intermediate-1' if 'intermediate-1' in diff else 'Intermediate-2'
			size = '10k' if '10k' in sz else '100k'
			print(f'{out_fn} - {diff} - {sz} - {run}')

			if col == 'step' or col == 'epoch':
				continue

			df = metric[metric[col].notnull()]
			if max_epochs:
				df = df[df['epoch'] <= max_epochs]

			x_vals = df[selected_columns[0]]
			y_vals = df[col]

			print(f'Plotting column: {col} along x-axis: {selected_columns[0]}')
			print_x_y_lists(x_vals, y_vals, n=3)

			max_y_val = np.max(y_vals) if max_y_val < np.max(y_vals) else max_y_val
			max_x_val = np.max(x_vals) if max_x_val < np.max(x_vals) else max_x_val
			color = select_color(f'{col}-{diff}', selector=0, debug=True)
			if args.smooth_sigma != -1:
				y_vals = gaussian_filter1d(y_vals, sigma=args.smooth_sigma)
			img   = ax.plot(x_vals, y_vals, label=f'{difficulty}-{size}', color=color)
			diff_counter += 1
	
			# set imshow outline
			for spine in img[0].axes.spines.values():
				spine.set_edgecolor(hex_to_rgba(colors['axis']))    

		legend_loc = place_legend(args.legend)

		if 'loss' in label or 'Loss' in label:
			legend_loc = 'upper right'
			
		apply_wandb_graph_style(ax, plt, loc=legend_loc, title=f'Symmetria-{size} {run} {label}', xlabel=columns_dict[selected_columns[0]][1], ylabel=f'{run} {label}', title_fontsize=32, legend_fontsize=18)
		# Set x and y axis limits
		plt.xlim(-0.001*max_x_val, max_x_val)
		plt.ylim(-0.004*max_y_val, max_y_val + 0.01*max_y_val)
			
		plt.subplots_adjust(bottom=0.15)
		#if run == "valid":
		#	plt.subplots_adjust(left=0.11)
			
		# Save figure
		fig.set_size_inches(19.2, 10.8)
		#plt.tight_layout()
			
		#out_fn = args.out_fn if args.out_fn else 'graph.png'
		fig.savefig(f'{out_fn}', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))
	plt.show()
		
