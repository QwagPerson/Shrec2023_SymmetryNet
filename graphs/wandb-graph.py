#!/usr/bin/env python3

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from scipy.ndimage.filters import gaussian_filter1d

from colors import colors, hex_to_rgba, select_color, apply_wandb_graph_style

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
import argparse
argparse = argparse.ArgumentParser()
argparse.add_argument('fn'  , type=str, help='Metrics filename')	# positionals are always required
argparse.add_argument('x'   , type=str, help='x axis')			# e.g. "epoch" or "step"
argparse.add_argument('y1'  , type=str, help='y axis no. 1')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y2', type=str, help='y axis no. 2')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y3', type=str, help='y axis no. 3')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y4', type=str, help='y axis no. 4')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y5', type=str, help='y axis no. 5')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--y6', type=str, help='y axis no. 6')		# e.g. "plane_train_loss_epoch"
argparse.add_argument('--smooth_sigma', type=int, default=2, help='Smooth segmented lines using splines with sigma = <smooth_sigma>')
argparse.add_argument('--legend', type=str, help='Where to place the legend (e.g. lr, tl, etc.)')		# e.g. "lr = lower right, etc."
argparse.add_argument('--out_fn', type=str, help='Output filename')
args = argparse.parse_args()

# /mnt/btrfs-big/dataset/geometric-primitives-classification/neurips-final/metrics/andrea/easy/10k/metrics.csv

metrics = pd.read_csv(args.fn)

print(f'{metrics.columns = }')
print(f'{metrics.head() = }')

# Set fivethirtyeight style
plt.style.use('fivethirtyeight')

# Create figure and axis objects
fig, ax = plt.subplots()

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

			'plane_train_loss_confidence_epoch':		[ 2, 'Conf. Loss'],
			'plane_train_loss_distance_epoch':		[ 4, 'Dist. Loss'],
			'plane_train_loss_epoch':			[ 6, 'Loss'],
			'plane_train_loss_normal_epoch':		[ 7, 'Norm. Loss'],
			'plane_train_loss_ref_sym_distance_epoch':	[ 9, 'Ref Sym Dist Loss'],
			'plane_train_map_epoch':			[12, 'mAP'],
			'plane_train_phc_epoch':			[14, 'PHC'],

			'plane_val_loss_confidence_epoch':		[16, 'Conf. Loss'],
			'plane_val_loss_confidence_step':		[17, 'plane_val_loss_confidence_step'],
			'plane_val_loss_distance_epoch':		[18, 'Dist. loss'],
			'plane_val_loss_distance_step':			[19, 'plane_val_loss_distance_step'],
			'plane_val_loss_epoch':				[20, 'Loss'],
			'plane_val_loss_normal_epoch':			[21, 'Norm. Loss'],
			'plane_val_loss_normal_step':			[22, 'plane_val_loss_normal_step'],
			'plane_val_loss_ref_sym_distance_epoch':	[23, 'Ref Sym Dist Loss'],
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

# Who knows what's this for...
ax.tick_params(axis='x', which='major', bottom=True)

max_y_val = -1
max_x_val = -1

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

run = 'Training' if 'train' in selected_columns[1] else 'Validation'

for col in selected_columns:
	if col == 'step' or col == 'epoch':
		continue

	df = metrics[metrics[col].notnull()]

	x_vals = df[selected_columns[0]]
	print(f'{x_vals = }')
	y_vals = df[col]
	print(f'Plotting column: {col} with values: {y_vals = }')

	if True:
		max_y_val = np.max(y_vals) if max_y_val < np.max(y_vals) else max_y_val
		max_x_val = np.max(x_vals) if max_x_val < np.max(x_vals) else max_x_val
		color = select_color(col, selector=0)
		#color = select_color(col, selector=0 if run == 'Training' else 1)
		label = columns_dict[col][1]
		if args.smooth_sigma != -1:
			y_vals = gaussian_filter1d(y_vals, sigma=args.smooth_sigma)
		img   = ax.plot(x_vals, y_vals, label=label, color=color)

		# set imshow outline
		for spine in img[0].axes.spines.values():
			spine.set_edgecolor(hex_to_rgba(colors['axis']))    

difficulty = 'Easy' if 'easy' in out_fn else 'Hard' if 'hard' in out_fn else 'Intermediate-1' if 'intermediate-1' in out_fn else 'Intermediate-2'
size = '10k' if '10k' in out_fn else '100k'
print(f'{out_fn} - {difficulty} - {size} - {run}')

legend_loc = place_legend(args.legend)
apply_wandb_graph_style(ax, plt, loc=legend_loc, title=f'Symmetria {difficulty} {size}', xlabel=columns_dict[selected_columns[0]][1], ylabel=f'{run} losses', title_fontsize=32, legend_fontsize=18, label_fontsize=24)

# Set x and y axis limits
plt.xlim(-0.001*max_x_val, max_x_val)
plt.ylim(-0.004*max_y_val, max_y_val + 0.01*max_y_val)

plt.subplots_adjust(bottom=0.15)
#if run == "valid":
#	plt.subplots_adjust(left=0.11)

# Save figure
fig.set_size_inches(19.2, 10.8)
out_fn = args.out_fn if args.out_fn else 'graph.png'
fig.savefig(f'{out_fn}', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))

#plt.tight_layout()

# Show the plot
plt.show()

