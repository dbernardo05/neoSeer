#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.width = 0
import sys

from datetime import date
import joblib
from optparse import OptionParser
from time import time
from typing import Dict, List, Tuple, Any

import tqdm
from tqdm.auto import trange
from tqdm.contrib import tzip

import sys
import preproc

# ML
import torch

from fastai.callback.progress import CSVLogger
from mixed_patch import *

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder
from tsai.all import *


def total_params(model):
	return sum(p.numel() for p in model.parameters())

def y_state(o): return o[:,-1]

def get_archs(default_lr, default_patience):

	# exp20230119--  +HUH (HIE only) and temp scaling
	archs = [
			('XGBoost', {}, [], []),
	]

	return archs


def save_preds(X_test, y_test, y_preds, win_key, kfold, arch_name, featclass, epoch_winlen):
	preds_df = pd.DataFrame.from_dict({'arch': arch_name,
									'kfold': kfold,
									'y_test': y_test, 
									'y_preds': y_preds,
									'win_key': win_key}) # win_key is snd_grp
	preds_df_fn = 'results/preds_df_' + arch_name + '_kf' + str(kfold) + '_' + str(featclass) + '_winlen' + str(epoch_winlen) + '.csv'
	preds_df.to_csv(preds_df_fn, index=False)


def save_MCdropout_preds(X_test, y_test, all_preds, win_key, kfold, arch_name, featclass, epoch_winlen):
	preds_df = pd.DataFrame.from_dict({'arch': arch_name,
									'kfold': kfold,
									'y_test': y_test, 
									'win_key': win_key}) # win_key is snd_grp
	for i, yp in enumerate(all_preds):
		preds_df['y_preds_run' + str(i)] = yp
	preds_df_fn = 'results/preds_df_' + arch_name + '_kf' + str(kfold) + '_' + str(featclass) + '_winlen' + str(epoch_winlen) + '.csv'
	preds_df.to_csv(preds_df_fn, index=False)



def get_clinvars_Xy(df, clin_var_cols, stride_, y_state, epoch_winlen, y_arr, drop_start=0, drop_end=0):
	X_clinvar, y = TSUnwindowedDataset(X=df[clin_var_cols].values, y=df['preictal'].values, y_func=y_state,
						   window_size=epoch_winlen, stride=stride_, seq_first=True,
						   drop_start=drop_start, drop_end=drop_end)[:]
	X_clinvar = np.array(X_clinvar)
	clin_vars_lst = []
	missing_rep_nan = False  # Replace nan with zeros or not
	for arr in X_clinvar:
		# CORK data has all zeros so swap out nan
		if missing_rep_nan:
			if np.isnan(arr).all():
				arr = np.zeros_like(arr)
		featvec = arr[:,0]
		arr = arr.T 
		clin_vars_lst.append(featvec)

		assert (arr == arr[0]).all()

	tab_y_arr = np.array(y).astype(np.int64)
	y_arr = np.array(y_arr).astype(np.int64)

	assert np.array_equal(y_arr, tab_y_arr)

	clin_vars_arr = np.vstack(clin_vars_lst)

	return clin_vars_arr


def get_fold_data(kfolds_dir, kfold, datatype, resume_from_secStats=False, secondary_stats_mode=False):
	# Load folds data
	df_kf_ss_fn = '{}_secStats/{}_kf{}.hdf'.format(kfolds_dir, datatype, kfold)

	if resume_from_secStats and os.path.isfile(df_kf_ss_fn):
		df_kf = pd.read_csv(df_kf_ss_fn) 
	else:
		df_t_fn = '{}/{}_kf{}.hdf'.format(kfolds_dir, datatype, kfold)
		df_kf = pd.read_hdf(df_t_fn) 

	return df_kf


def fetch_training_objs(kfolds_dir, kfold, resume_from_secStats, secondary_stats_mode, 
	add_clinvars_mode, featclass, time_mode, reuse_training_objs, debug_mode):
	df_kf_pp_fn = '{}_postprocTrainData/train_kf{}.csv'.format(kfolds_dir, kfold)
	kf_ss_fn = '{}_postprocTrainData/train_kf{}_scaler.joblib'.format(kfolds_dir, kfold)
	kf_imp_fn = '{}_postprocTrainData/train_kf{}_imputer.joblib'.format(kfolds_dir, kfold)
	kf_rawdatacols_fn = '{}_postprocTrainData/train_kf{}_rawdatacols.pkl'.format(kfolds_dir, kfold)
	kf_clinvarcols_fn = '{}_postprocTrainData/train_kf{}_clinvarcols.pkl'.format(kfolds_dir, kfold)


	if os.path.isfile(df_kf_pp_fn) and os.path.isfile(kf_ss_fn) and os.path.isfile(kf_imp_fn) and reuse_training_objs:
		df_train = pd.read_csv(df_kf_pp_fn)
		scaler = joblib.load(kf_ss_fn)
		imp = joblib.load(kf_imp_fn)
		with open(kf_rawdatacols_fn, "rb") as input_file:
			raw_data_cols = pkl.load(input_file)
		with open(kf_clinvarcols_fn, "rb") as input_file:
			clin_var_cols = pkl.load(input_file)
		print("\tdf_train shape:", df_train.shape)
		print("\tNum raw_data_cols:", len(raw_data_cols))
	else:
		df_train = get_fold_data(kfolds_dir, kfold, "train", resume_from_secStats, secondary_stats_mode)

		df_train_subjs = df_train.subject_id.unique()

		if debug_mode:
			print("# Debug mode")
			huh_subjs = [s for s in df_train_subjs if s<100]
			sel_subjs = random.sample(huh_subjs, 7)
			df_train = df_train[df_train['subject_id'].isin(sel_subjs)]
			df_train_subjs = sel_subjs
		print("# Total {} training subjects.".format(len(df_train_subjs)))
		print("# \tdf_train_subjs:", df_train_subjs)

		# Add clinical variables
		clin_var_cols = []

		# Select feature columns
		raw_data_cols = [ f for f in df_train.columns if f not in ['subject_id', 'abs_time_idx', 'preictal', 'augmentation', 'snd_grp', 'unq_subj_id'] ]

		# Select features if using per feature models
		if featclass != 'all':
			raw_data_cols = [ f for f in raw_data_cols if featclass in f]

		if time_mode:
			df_train['time_vec'] = 0 
			for ns, s in enumerate(df_train.subject_id.unique()):
				df_train.loc[df_train.subject_id == s,['time_vec']] = np.arange(np.sum(df_train.subject_id == s))		
			raw_data_cols.append('time_vec')

		# Scaler would ideally be here... but since doing TSUnwindowed needed to move here.
		# scaler = RobustScaler().fit(df_train[data_cols].values)
		df_train.replace(np.nan, 0, inplace=True) # temp fix
		scaler = PowerTransformer().fit(df_train[raw_data_cols].values)

		df_train[raw_data_cols] = scaler.transform(df_train[raw_data_cols])
		df_train = df_train.sort_values(['subject_id', 'snd_grp', 'abs_time_idx'], ascending=[True, True, True])

		# Save stuff (if reuse mode on)
		if reuse_training_objs:
			df_train.to_csv(df_kf_pp_fn, index=False)
			joblib.dump(scaler, kf_ss_fn)
			with open(kf_rawdatacols_fn, "wb") as output_file:
				pkl.dump(raw_data_cols, output_file)
			with open(kf_clinvarcols_fn, "wb") as output_file:
				pkl.dump(clin_var_cols, output_file)




	return df_train, scaler, raw_data_cols, clin_var_cols



def get_metadataloader(kfold, epoch_winlen, bs, kfolds_dir, featclass, 
									add_clinvars_mode=False, 
									multimodal_mode=False, 
									secondary_stats_mode=False, 
									time_mode=False, 
									neg_pos_ratio=None,
									resume_from_secStats=False,
									JK_paper_mode=True,
									reuse_training_objs=False,
									debug_mode=False,
									AFE_mode=False):

	print("# Fetching training data")
	df_train, scaler, raw_data_cols, clin_var_cols = fetch_training_objs(kfolds_dir, kfold, resume_from_secStats, secondary_stats_mode, 
																add_clinvars_mode, featclass, time_mode, reuse_training_objs, debug_mode)
	print("\tdf_train shape:", df_train.shape)


	# If multimodal mode, remove clinvars from data cols, after scaling completed
	# raw_data_cols include all data_cols that require scaling
	# data_cols subsequently only includes qEEG features that are already scaled
	if multimodal_mode:
		data_cols = [ f for f in raw_data_cols if f not in clin_var_cols]
	else:
		data_cols = raw_data_cols

	if 0:
		# Debugging stuff
		print(df_train.subject_id.unique())
		print(df_train.augmentation.unique())
		for subj in df_train.subject_id.unique():
			plt.figure()
			df_samp_x = df_train[(df_train.subject_id==subj) & (df_train.augmentation == 'none')].abs_time_idx
			df_samp_y = df_train[(df_train.subject_id==subj) & (df_train.augmentation == 'none')].preictal
			# df_samp_snd = df_train[(df_train.subject_id==subj) & (df_train.augmentation == 'none')].ii_safe
			plt.plot(df_samp_x, df_samp_y)
			# plt.plot(df_samp_x)
			plt.savefig('subj_tracings/{}.png'.format(subj))
		sys.exit()

	# Reset unq_subj_id, parses time epochs (e.g. for individual seizures)
	df_train["unq_subj_id"] = df_train["subject_id"].astype(str) + '_' + df_train["augmentation"] + '_' + df_train["snd_grp"].astype(str)

	# Estimate class imbalance
	neg_samples = 0
	pos_samples = 0
	for s in df_train.unq_subj_id.unique():
		y = df_train[df_train.unq_subj_id == s].preictal.values
		y_arr = np.array(y).astype(np.int64)
		neg_samples += np.sum(y_arr==0)
		pos_samples += np.sum(y_arr==1)
	adj_factor = 1 # higher reduces stride size

	if neg_pos_ratio is None:
		neg_pos_ratio = int((neg_samples // pos_samples)/adj_factor)
	print("Neg to Pos ratio (train):", neg_pos_ratio, "to be used as stride.")

	clin_vars_X = []
	clin_vars_y = []

	datasets = []
	neg_samples = 0
	pos_samples = 0
	srate = 20

	for ns, s in enumerate(df_train.unq_subj_id.unique()):

		df_S = df_train[df_train.unq_subj_id == s]

		len_data = len(df_S[data_cols].values)
		curr_subj = df_S.subject_id.unique()
		
		if len_data >= epoch_winlen:

			ii_pi_edge = np.where(np.diff(df_S['preictal'].values) == 1)[0]

			if ii_pi_edge.size == 0:
				# All PI or ALL II
				# print('\t\t:',np.unique(df_S['preictal'].values)[0])
				if np.unique(df_S['preictal'].values)[0] == 1:
					# All PI
					# stride_ = srate * 3
					stride_ = 1

				else:
					# All II
					# stride_ = srate * (neg_pos_ratio // 5)
					stride_ = neg_pos_ratio * 4

				X, y = TSUnwindowedDataset(X=df_S[data_cols].values, y=df_S['preictal'].values, y_func=y_state,
									   window_size=epoch_winlen, stride=int(stride_), seq_first=True)[:]
				y_arr = np.array(y).astype(np.int64)

				if multimodal_mode:
					curr_clin_vars  = get_clinvars_Xy(df_S, clin_var_cols, stride_, y_state, epoch_winlen, y_arr)
					clin_vars_X.extend(curr_clin_vars)
					clin_vars_y.extend(y_arr.tolist())

				dset = TSDatasets(X, y_arr)
				datasets.append(dset)
				neg_samples += np.sum(y_arr==0)
				pos_samples += np.sum(y_arr==1)
				# assert np.sum(y_arr==1) == 0

			else:

				# Cut location is 8*epoch_winlen (20 sec * 15 * 12 == 1 hr) before preictal period
				cut_loc_ii = ii_pi_edge[0] - epoch_winlen*6

				# print(len_data, epoch_winlen, ii_pi_edge, cut_loc, len_data-cut_loc)

				# Check if interictal period is long enough
				if cut_loc_ii >= epoch_winlen:
					# Interictal
					X, y = TSUnwindowedDataset(X=df_S[data_cols].values, y=df_S['preictal'].values, y_func=y_state,
										   window_size=epoch_winlen, stride=srate*10, 
										   drop_end=len_data-cut_loc_ii, seq_first=True)[:]
					y_arr = np.array(y).astype(np.int64)


					if multimodal_mode:
						curr_clin_vars  = get_clinvars_Xy(df_S, clin_var_cols, neg_pos_ratio, y_state, epoch_winlen, y_arr, drop_end=len_data-cut_loc_ii)
						clin_vars_X.extend(curr_clin_vars)
						clin_vars_y.extend(y_arr.tolist())

					dset = TSDatasets(X, y_arr)
					datasets.append(dset)

					neg_samples += np.sum(y_arr==0)
					pos_samples += np.sum(y_arr==1)
					assert np.sum(y_arr==1) == 0

				cut_loc_pi = ii_pi_edge[0] - epoch_winlen

				# Check if preictal period is long enough
				if len_data-cut_loc_pi < epoch_winlen:
					print('subject with short PI period:', s, len_data, ii_pi_edge[0], cut_loc_pi)
					continue

				# Preictal
				preictal_stride = 20
				X, y = TSUnwindowedDataset(X=df_S[data_cols].values, y=df_S['preictal'].values, y_func=y_state,
									   window_size=epoch_winlen, stride=preictal_stride, 
									   drop_start=cut_loc_pi, seq_first=True)[:]
				y_arr = np.array(y).astype(np.int64)


				if multimodal_mode:
					curr_clin_vars  = get_clinvars_Xy(df_S, clin_var_cols, preictal_stride, y_state, epoch_winlen, y_arr, drop_start=cut_loc_pi)
					clin_vars_X.extend(curr_clin_vars)
					clin_vars_y.extend(y_arr.tolist())


				dset = TSDatasets(X, y_arr)
				datasets.append(dset)
				pos_samples += np.sum(y_arr==1)

				# assert np.array_equal(np.unique(y_arr), [0, 1])

	pos_wt = neg_samples / (pos_samples + 1e-5)
	pos_wt = torch.as_tensor(pos_wt, dtype=torch.float)


	valid_len = 0

	df_val = get_fold_data(kfolds_dir, kfold, "valid", resume_from_secStats, secondary_stats_mode)
	print("\tdf_val shape:", df_val.shape)
	df_val = df_val.sort_values(['subject_id', 'abs_time_idx'], ascending=[True, True])

	if debug_mode:
		print("# Debug mode")
		huh_subjs = [s for s in df_val.subject_id if s<100]
		sel_subjs = random.sample(huh_subjs, 3)
		df_val = df_val[df_val['subject_id'].isin(sel_subjs)]
	print("# Total {} valid subjects.".format(len(df_val.subject_id.unique())))
	print("# Valid subjs:", df_val.subject_id.unique())


	# Add secondary statistics
	if secondary_stats_mode:
		temp_data_cols = [ f for f in df_val.columns if f not in ['subject_id', 'abs_time_idx', 'preictal', 'augmentation', 'snd_grp'] ]
		df_val = add_secondary_stats(df_val, temp_data_cols)

	df_val["unq_subj_id"] = df_val["subject_id"].astype(str) + '_' + df_val["augmentation"] + '_' + df_val["snd_grp"].astype(str)

	clin_var_cols = None

	if time_mode:
		df_val['time_vec'] = 0 
		for ns, s in enumerate(df_val.subject_id.unique()):
			df_val.loc[df_val.subject_id == s,['time_vec']] = np.arange(np.sum(df_val.subject_id == s))		

	if scaler:
		df_val.replace(np.nan, 0, inplace=True) # temp fix
		df_val[raw_data_cols] = scaler.transform(df_val[raw_data_cols])

	val_stride= srate * 5
	neg_samples = 0
	pos_samples = 0
	for s in df_val.unq_subj_id.unique():
		if JK_paper_mode:
			if not AFE_mode:
				s_root = s.split('_')[0]
				if int(s_root)>100 and int(s_root)<9000:
					if len(np.unique(df_S['preictal'].values)) == 1:
						print(f"Excluding UCSF subject with no seizures:{s_root}")
						continue

		df_S = df_val[df_val.unq_subj_id == s]
		if len(df_S[data_cols].values) >= epoch_winlen:
			X, y = TSUnwindowedDataset(X=df_S[data_cols].values, y=df_S['preictal'].values, y_func=y_state,
									   window_size=epoch_winlen, stride= int(val_stride), seq_first=True)[:]
			dset = TSDatasets(X, np.array(y).astype(np.int64))
			datasets.append(dset)
			valid_len += len(y)
			y_arr = np.array(y).astype(np.int64)
			neg_samples += np.sum(y_arr==0)
			pos_samples += np.sum(y_arr==1)
			if multimodal_mode:
				curr_clin_vars  = get_clinvars_Xy(df_S, clin_var_cols, val_stride, y_state, epoch_winlen, y_arr)
				clin_vars_X.extend(curr_clin_vars)
				clin_vars_y.extend(y_arr.tolist())

	pos_wt_v = neg_samples / (pos_samples + 1e-5)
	pos_wt_v = torch.as_tensor(pos_wt_v, dtype=torch.float)

	trainval_metadataset = TSMetaDataset(datasets)
	splits = TSSplitter(valid_size=valid_len, show_plot=False)(trainval_metadataset)

	metadatasets = TSMetaDatasets(trainval_metadataset, splits=splits)
	dls = TSDataLoaders.from_dsets(metadatasets.train, metadatasets.valid, bs=[bs, bs], 
		# batch_tfms=[TSClipOutliers(-3, 3, verbose=True), TSCutOut(0.05), TSTimeStepOut(0.05), TSStandardize()]) #, TSVarOut(0.03), TSCutOut(0.01), TSMaskOut(0.03)
		# batch_tfms=[TSCutOut(0.02), TSTimeStepOut(0.02), TSStandardize()]) #, TSVarOut(0.03), TSCutOut(0.01), TSMaskOut(0.03)
		batch_tfms=[TSTranslateX(1/epoch_winlen), TSTimeStepOut(1/epoch_winlen)]) #, TSVarOut(0.03), TSCutOut(0.01), TSMaskOut(0.03)

	# Code here to make pandas df of clinVars
	if multimodal_mode:
		clin_vars_X = np.vstack(clin_vars_X)
		print(clin_vars_X.shape)
		clin_vars_df = pd.DataFrame(np.array(clin_vars_X), columns=clin_var_cols)
		clin_vars_df['preictal'] = clin_vars_y
		tab_dls = get_tabular_dls(clin_vars_df, cat_names=[], procs=[], cont_names=clin_var_cols, y_names=['preictal'], splits=splits)
	else:
		tab_dls = None

	return dls, tab_dls, scaler, raw_data_cols, data_cols, clin_var_cols, pos_wt, pos_wt_v



def get_test_dl(kfold, epoch_winlen, kfolds_dir, raw_data_cols, data_cols, clin_var_cols, scaler, bs, multimodal_mode, test_stride=1, 
					secondary_stats_mode=True, time_mode=False, add_clinvars_mode=False, include_HUH_HIE_data=True, 
					resume_from_secStats=True, JK_paper_mode=False, label_ictal_as_interictal=False):

	df_test = get_fold_data(kfolds_dir, kfold, "test", resume_from_secStats, secondary_stats_mode)
	df_test['ictal'] = df_test['preictal']
	df_test['ictal'] = df_test['ictal'].replace({1:0})
	label_ictal_as_interictal=True
	if label_ictal_as_interictal:
		df_test['preictal'] = df_test['preictal'].replace({2:0})
	else:
		# label_ictal_as_preictal
		df_test['preictal'] = df_test['preictal'].replace({2:1})

	# Add clinical variables
	if add_clinvars_mode:
		df_test, clin_var_cols = add_clin_vars(df_test)

	df_test["unq_subj_id"] = df_test["subject_id"].astype(str) + '_' + df_test["augmentation"] + '_' + df_test["snd_grp"].astype(str)

	print("Test subjs:", df_test.subject_id.unique())

	# Estimate class imbalance
	neg_samples = 0
	pos_samples = 0
	for s in df_test.unq_subj_id.unique():
		y = df_test[df_test.unq_subj_id == s].preictal.values
		y_arr = np.array(y).astype(np.int64)
		neg_samples += np.sum(y_arr==0)
		pos_samples += np.sum(y_arr==1)
	neg_pos_ratio = int((neg_samples // pos_samples))
	print("Neg to Pos ratio (test):", neg_pos_ratio)

	if time_mode:
		df_test['time_vec'] = 0 
		for ns, s in enumerate(df_test.subject_id.unique()):
			df_test.loc[df_test.subject_id == s,['time_vec']] = np.arange(np.sum(df_test.subject_id == s))		

	if scaler:
		df_test.replace(np.nan, 0, inplace=True) # temp fix
		df_test[raw_data_cols] = scaler.transform(df_test[raw_data_cols])

	X_test, y_test, X_test_ictal, y_test_ictal, clin_vars_dfs, subj_keys = [], [], [], [], [], []

	AFE_mode = False
	if AFE_mode:
		test_stride = 20*20
	else:
		test_stride = 1
	

	for ns, s in enumerate(df_test.subject_id.unique()):

		df_S = df_test[df_test.subject_id == s]


		curr_X_test, curr_y_test = TSUnwindowedDataset(X=df_S[data_cols].values, y=df_S['preictal'].values, y_func=y_state,
						   window_size=epoch_winlen, stride=int(test_stride), seq_first=True)[:]


		curr_X_test_ictal, curr_y_test_ictal = TSUnwindowedDataset(X=np.arange(len(df_S['ictal'])), y=df_S['ictal'].values, y_func=y_state,
						   window_size=epoch_winlen, stride=int(test_stride), seq_first=True)[:]


		if multimodal_mode:
			curr_clin_vars  = get_clinvars_Xy(df_S, clin_var_cols, test_stride, y_state, epoch_winlen, curr_y_test)
			clin_vars_X = curr_clin_vars
			clin_vars_y = curr_y_test

			# Code here to make pandas df of clinVars
			clin_vars_df = pd.DataFrame(np.array(clin_vars_X), columns=clin_var_cols)
			clin_vars_df['preictal'] = clin_vars_y

			clin_vars_dfs.append(clin_vars_df)
		else:
			clin_vars_dfs.append(None)


		X_test.append(curr_X_test)
		y_test.append(curr_y_test)

		X_test_ictal.append(curr_X_test_ictal)
		y_test_ictal.append(curr_y_test_ictal)

		unq_subjs = np.unique(df_S.subject_id.values)
		assert len(unq_subjs) == 1
		subj_keys.extend(unq_subjs.tolist())

	return X_test, y_test, X_test_ictal, y_test_ictal, clin_vars_dfs, subj_keys




def generator(dls):
	iter_ = iter(dls)
	while True:
		for step in range(len(dls)):
			X, y = next(iter_)
			X = X.cpu().detach().numpy()
			y = y.cpu().detach().numpy()
			yield np.average(X, axis=2), y

def total_params(model):
	return sum(p.numel() for p in model.parameters())


