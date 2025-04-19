# Current version -- v3

import numpy as np
import pandas as pd
import sys
import preproc_utils as utils
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

def valid_feat_name(c):
	v = c not in ["index", "preictal", "trust", "seizing", "time_idx", "group", "GA", "y", "Unnamed: 0", \
				  "fname", "idx", 'subsegment_x', 'subsegment_y', 'subsegment', 'snd_grp', 'pna_szf', 'pna']
	v = v and c.split("_")[0] != "valid"
	return v


def get_huh_cork_raw_data_df():
	load_from_save = True
	if load_from_save:
		df_huh_train = pd.read_csv("data/df_SWEDE_CORK_20hz_jkRev.csv")

	# clean up dataframe

	# replace labels of buffer periods with label 3 so that loss fn will ignore those rows
	df_huh_train["preictal"] = df_huh_train["preictal"].replace(-1, 0) 
	print('Len df_huh_train.preictal:', len(df_huh_train))
	# drop rows w/ seizure as well as rows where features have nans
	# df_huh_train = df_huh_train[df_huh_train.trust]
	df_huh_train = df_huh_train.reset_index(level=0, drop=True)
	print('Len df_huh_train.preictal:', len(df_huh_train))
	df_huh_train["seizing"] = df_huh_train["seizing"].astype(str)

	curr_cols = df_huh_train.columns

	df_huh_train["subject_id"] = df_huh_train["subject_id"].astype(str)
	df_huh_train['abs_time_idx'] = df_huh_train['time_idx'].astype(int)

	# HUH_subjs = [s for s in df_huh_train.subject_id.unique() if len(s)==3]

	# df_huh_train = df_huh_train[df_huh_train.subject_id.isin(HUH_subjs)]

	return df_huh_train

def get_huh_training_df(load_from_save=True):
	if load_from_save:
		# df_huh_train = pd.read_pickle("data/df_all_subjs_Aug2022_processed.p")
		df_huh_train = pd.read_csv('data/huh_subjects.csv', dtype={'subject_id': str})

	# replace labels of buffer periods with label 3 so that loss fn will ignore those rows
	df_huh_train["preictal"] = df_huh_train["preictal"].replace(-1, 0) 
	print('Len df_huh_train.preictal:', len(df_huh_train))
	# drop rows w/ seizure as well as rows where features have nans
	# df_huh_train = df_huh_train[df_huh_train.trust]
	df_huh_train = df_huh_train.reset_index(level=0, drop=True)
	print('Len df_huh_train.preictal:', len(df_huh_train))
	df_huh_train["seizing"] = df_huh_train["seizing"].astype(str)

	# Make sure only HUH subjs are included
	HUH_subjs = [s for s in df_huh_train.subject_id.unique() if len(s)==3]

	df_huh_train = df_huh_train[df_huh_train.subject_id.isin(HUH_subjs)]

	return df_huh_train

# def get_df_ucsf_nosz():
# 	time_to_EEG_df = pd.read_csv('./data/Time_to_EEG_20230813_ucsfFinal.csv')
# 	time_to_EEG_df['Subject'] = time_to_EEG_df['Subject'].fillna(value=-1)
# 	time_to_EEG_df['Subject'] = time_to_EEG_df['Subject'].astype(int)

# 	# For loading UCSF Data without seizures
# 	load_from_save = True
# 	if load_from_save:
# 		df_nosz = pd.read_csv("data/df_master_allHIEnoSz_noTD.csv")

# 	# repl/ce labels of buffer periods with label 3 so that loss fn will ignore those rows
# 	df_nosz["preictal"] = df_nosz["preictal"].replace(-1, 0) 
# 	print('Len df_nosz:', len(df_nosz))
# 	# drop rows w/ seizure as well as rows where features have nans
# 	# df_master_train = df_master_train[df_master_train.trust]
# 	df_nosz = df_nosz.reset_index(level=0, drop=True)
# 	print('Len df_nosz:', len(df_nosz))
# 	df_nosz["seizing"] = df_nosz["seizing"].astype(str)
# 	df_nosz["subject_id"] = df_nosz["group"].astype(str)

# 	print(df_nosz.subject_id.unique())
# 	## Apply new SND label mapping
# 	df_nosz['snd_grp'] = np.zeros((len(df_nosz),))

# 	# Add abs_time_idx
# 	df_nosz['snd_grp'] = np.zeros((len(df_nosz),))
# 	df_nosz['abs_time_idx'] = df_nosz['time_idx'].astype(int)

# 	n = 0 
# 	for grp in df_nosz["subject_id"].unique():
# 		df_nosz.loc[df_nosz['subject_id'] == grp,'snd_grp'] = int(9900000 + n)
# 		n+=1 

# 	df_nosz["subject_id"] = df_nosz["subject_id"].str.slice(0,4)


# 	# Add postnatal_time_without_sz_lbls

# 	postnatal_age_without_sz_lbls = {}
# 	print(time_to_EEG_df.Subject)
# 	for name, grp in df_nosz.groupby(by='subject_id'):
# 		age_corr = time_to_EEG_df[time_to_EEG_df.Subject==int(name)].Time_to_EEG.values*3 # In minutes so convert to epoch time
# 		sz = np.array([1 if n == 'True' else 0 for n in grp['seizing'].values])

# 		if np.sum(sz) == 0:
# 			postnatal_time_without_sz = np.arange(len(sz)) + age_corr
# 		else:
# 			index_of_first_sz = np.where(sz == 1)[0][0]
# 			postnatal_time_without_sz = np.zeros(len(sz))
# 			postnatal_time_without_sz[:index_of_first_sz] = np.arange(index_of_first_sz) + age_corr

# 		postnatal_age_without_sz_lbls[name] = postnatal_time_without_sz

# 	print(df_nosz.shape)
# 	postnatal_age_lbls = {}
# 	for name, grp in df_nosz.groupby(by='subject_id'):
# 		len_time = len(grp)
# 		age_corr = time_to_EEG_df[time_to_EEG_df.Subject==int(name)].Time_to_EEG.values*3 # In minutes so convert to epoch time
# 		postnatal_age = np.arange(len_time) + age_corr
# 		postnatal_age_lbls[name] = postnatal_age

# 	## Apply pna_szf label mapping
# 	df_nosz['pna_szf'] = np.zeros((len(df_nosz),))
# 	for grp, lbls in postnatal_age_without_sz_lbls.items():
# 		df_nosz.loc[df_nosz['subject_id'] == grp,'pna_szf'] = lbls

# 	df_nosz['pna'] = np.zeros((len(df_nosz),))
# 	for grp, lbls in postnatal_age_lbls.items():
# 		df_nosz.loc[df_nosz['subject_id'] == grp,'pna'] = lbls

# 	return df_nosz


def get_df_cork():
	# For loading cork data without seizures
	load_from_save = True
	if load_from_save:
		df_cork = pd.read_csv("data/df_master_cork_allHIEnoSz_noTD.csv")

	# repl/ce labels of buffer periods with label 3 so that loss fn will ignore those rows
	df_cork["preictal"] = df_cork["preictal"].replace(-1, 0) 
	print('Len df_cork:', len(df_cork))
	# drop rows w/ seizure as well as rows where features have nans
	# df_master_train = df_master_train[df_master_train.trust]
	df_cork = df_cork.reset_index(level=0, drop=True)
	print('Len df_cork:', len(df_cork))
	df_cork["seizing"] = df_cork["seizing"].astype(str)
	df_cork["subject_id"] = df_cork["group"].astype(str)


	## Apply new SND label mapping
	df_cork['snd_grp'] = np.zeros((len(df_cork),))

	# Add abs_time_idx
	df_cork['snd_grp'] = np.zeros((len(df_cork),))
	df_cork['abs_time_idx'] = df_cork['time_idx'].astype(int)

	# 9900000 signifies UCSF No Seizure
	# 9700000 signifies CORK
	n = 0 
	for grp in df_cork["subject_id"].unique():
		df_cork.loc[df_cork['subject_id'] == grp,'snd_grp'] = int(9700000 + n)
		n+=1 

	df_cork["subject_id"] = df_cork["subject_id"].str.slice(0,4)
	df_cork['subject_id'] = df_cork['subject_id'].str.replace('ID','97')

	# Add postnatal_time_without_sz_lbls
	# postnatal_time_without_sz_lbls = {}
	# for name, grp in df_cork.groupby(by='subject_id'):
	# 	sz = np.array([1 if n == 'True' else 0 for n in grp['seizing'].values])

	# 	unknown_time_without_sz = True
	# 	if unknown_time_without_sz:
	# 		postnatal_time_without_sz = np.full(shape=len(sz), fill_value=np.nan)
	# 	else:
	# 		if np.sum(sz) == 0:
	# 			postnatal_time_without_sz = np.arange(len(sz))
	# 		else:
	# 			index_of_first_sz = np.where(sz == 1)[0][0]
	# 			postnatal_time_without_sz = np.zeros(len(sz))
	# 			postnatal_time_without_sz[:index_of_first_sz] = np.arange(index_of_first_sz) + np.random.randint(0, 3*60*12)

	# 	postnatal_time_without_sz_lbls[name] = postnatal_time_without_sz

	# ## Apply pna_szf label mapping
	# df_cork['pna_szf'] = np.zeros((len(df_cork),))
	# for grp, lbls in postnatal_time_without_sz_lbls.items():
	# 	df_cork.loc[df_cork['subject_id'] == grp,'pna_szf'] = lbls

	return df_cork


def get_huh_subjs(subj_ids):
	huh_subjects = [s for s in np.unique(subj_ids) if len(s)==3]
	huh_subjects.remove("003") # burst suppression
	huh_subjects.remove("005") # these subjects don't stop seizing long enough to have sz-free data with which to predict
	huh_subjects.remove("014")
	huh_subjects.remove("041")
	huh_subjects = np.array(huh_subjects)
	return huh_subjects


def gen_kf_splits_and_arr(df_master_train, include_huh_in_training, num_kfolds, num_train_samps, add_nosz_subjs_to_train):

	# Parse subjects; huh_hie was for manually including HUH HIE Subjects in tuning array
	huh_y_lbls = []
	huh_hie_subjs = df_master_train.subject_id.unique()
	for subj in huh_hie_subjs:
		huh_pi_lbls = df_master_train[(df_master_train.subject_id==subj)].preictal.unique()
		if set(huh_pi_lbls) == set([0]):
			huh_y_lbls.append(0)
		else:
			huh_y_lbls.append(1)
	huh_y_lbls = np.array(huh_y_lbls)

	df_cork = get_df_cork()
	cork_subjs = df_cork.subject_id.unique()

	y_cork = np.zeros((len(cork_subjs),))

	sz_state_arr = np.hstack([y_cork, huh_y_lbls])
	tuning_subjects_arr = np.hstack([np.array(cork_subjs), np.array(huh_hie_subjs)])

	s_kf = StratifiedKFold(n_splits=num_kfolds, shuffle=True, random_state=42)
	tuning_splits = [_ for _ in s_kf.split(tuning_subjects_arr, sz_state_arr)]


	return tuning_splits, tuning_subjects_arr, huh_hie_subjs


def gen_kf_splits_and_arr_rawMode(df_master_train, include_huh_in_training, num_kfolds, num_train_samps):
	# drops = []
	# tuning_subjects_samp = [s for s in df_master_train.subject_id.unique() if s not in drops]
	tuning_subjects_samp = df_master_train.subject_id.unique()

	print('len subjs:', len(tuning_subjects_samp))
	# Parse subjects; huh_hie was for manually including HUH HIE Subjects in tuning array
	# huh_y_lbls = []
	# if include_huh_in_training:
	# 	huh_hie_subjs = df_master_train.subject_id.unique()
	# 	for subj in huh_hie_subjs:
	# 		huh_pi_lbls = df_master_train[(df_master_train.subject_id==subj)].preictal.unique()
	# 		if set(huh_pi_lbls) == set([0]):
	# 			huh_y_lbls.append(0)
	# 		else:
	# 			huh_y_lbls.append(1)
	# 	huh_y_lbls = np.array(huh_y_lbls)
	# else:
	huh_hie_subjs = []

	s_kf = KFold(n_splits=num_kfolds, shuffle=True, random_state=42)
	tuning_subjects_arr = np.array(tuning_subjects_samp)
	tuning_splits = [_ for _ in s_kf.split(tuning_subjects_arr)]


	return tuning_splits, tuning_subjects_arr, huh_hie_subjs


def preproc_df_noTD_rawMode(allowed_augments, kfolds_dir, num_train_samps=-1, num_kfolds=5, 
				regen_kf_data=False, one_seizure_mode=False, include_huh_in_training=True, add_nosz_subjs_to_train=True,
				tSMOTE_mode=False):


	df_master_train = get_huh_cork_raw_data_df()

	# List of values to be excluded
	exclude_values = ["003", "005", "014", "041"]  # Add more values as needed
	df_master_train = df_master_train[~df_master_train.subject_id.isin(exclude_values)]

	feature_column_names = [c for c in df_master_train.columns if valid_feat_name(c)]
	all_safe_cols = feature_column_names + ['preictal'] + ['snd_grp'] + ['pna_szf'] + ['pna']

	assert bool(set(['seizing', 'time_idx', 'group']) & set(all_safe_cols)) == 0

	# Generate KF Splits
	tuning_splits, tuning_subjects_arr, huh_hie_subjs = gen_kf_splits_and_arr_rawMode(df_master_train, include_huh_in_training, num_kfolds, num_train_samps)

	if regen_kf_data:
		# df_train contains HUH
		# df_tuning contains UCSF

		# Merge training and tuning datasets
		df_master_train["preictal"] = df_master_train["preictal"].astype(str)
		df_master_train["seizing"] = df_master_train["seizing"].astype(str)

		# Remove bad subjects
		# huh_subjects = get_huh_subjs(df_master_train.subject_id.values)

		# Generate selected labels and safe subjects
		df_master_train['abs_time_idx'] = df_master_train['time_idx']*400
		df_master_train['abs_time_idx'] = df_master_train['abs_time_idx'].astype(int)
		df_master_train = df_master_train.sort_values(['subject_id', 'abs_time_idx'])

		gradient_labeling = False
		if gradient_labeling:
			grp_lbls, snd_lbls, newPI_lbls, ii_safe_subjs, postnatal_age_lbls, postnatal_time_without_sz_lbls = utils.get_lbls(df_master_train)
		else:
			# this segments by seizure epochs
			grp_lbls, snd_lbls, newPI_lbls, ii_safe_subjs, postnatal_age_lbls, postnatal_time_without_sz_lbls = utils.get_lbls_sz_epochs(df_master_train, one_seizure=one_seizure_mode)

		## Apply new label mapping
		df_master_train['ii_safe'] = np.zeros((len(df_master_train),))
		for grp, lbls in grp_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'ii_safe'] = lbls

		## Apply new SND label mapping
		df_master_train['snd_grp'] = np.zeros((len(df_master_train),))
		for grp, lbls in snd_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'snd_grp'] = lbls

		## Apply new PI_Lbls label mapping
		for grp, lbls in newPI_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'preictal'] = lbls

		## Apply pna_szf label mapping
		df_master_train['pna_szf'] = np.zeros((len(df_master_train),))
		for grp, lbls in postnatal_time_without_sz_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'pna_szf'] = lbls

		## Apply pna label mapping
		df_master_train['pna'] = np.zeros((len(df_master_train),))
		for grp, lbls in postnatal_age_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'pna'] = lbls


		# tSMOTE
		tSMOTE_subjs = []
		if tSMOTE_mode:
			df_master_tuning_smote = utils.get_smote(df_master_train, feature_column_names, one_seizure=one_seizure_mode)
			tSMOTE_subjs = df_master_tuning_smote.subject_id.unique()
			df_master_train = df_master_train.append(df_master_tuning_smote, ignore_index=True)


		# if add_nosz_subjs_to_train:
		# 	df_nosz = get_df_nosz()
		# 	df_cork = get_df_cork()
		# 	numel_df_master_train = len(df_master_train)
		# 	df_master_train = df_master_train.append(df_nosz, ignore_index=True)
		# 	df_master_train = df_master_train.append(df_cork, ignore_index=True)

		# 	# FOR II_SAFE BUG (ii_safe was created to identify good ii training segments on patients with seizures; for patients with no seizures, this hsould default to 1)
		# 	df_master_train.loc[numel_df_master_train:, ['ii_safe']] = 1

		# Check labeling:
		print('sanity check:', np.sum(df_master_train.ii_safe[df_master_train.preictal == 2]))

		# Final data preprocessing (merge HUH and UCSF)
		ii_safe_subjs_huh = [s for s in ii_safe_subjs if (len(s)==3 and s not in huh_hie_subjs)]
		print(ii_safe_subjs_huh)


		tuning_td_subjs = []
		gen_kf_dfs(df_master_train, ii_safe_subjs_huh, tSMOTE_subjs, tuning_splits, tuning_subjects_arr, tuning_td_subjs, allowed_augments, all_safe_cols, kfolds_dir)


	return tuning_splits, feature_column_names


def preproc_df_noTD(allowed_augments, kfolds_dir, num_train_samps=-1, num_kfolds=5,  
				regen_kf_data=False, one_seizure_mode=False, include_huh_in_training=True, add_nosz_subjs_to_train=True,
				tSMOTE_mode=False):

	# Needs to be rebuilt
	# Load tuning dataset


	# df_master_tuning_td = get_tuning_td_df()
	df_huh_train = get_huh_training_df()

	feature_column_names = [c for c in df_huh_train.columns if valid_feat_name(c)]
	all_safe_cols = feature_column_names + ['preictal'] + ['snd_grp'] + ['pna_szf'] + ['pna']

	assert bool(set(['seizing', 'time_idx', 'group']) & set(all_safe_cols)) == 0

	# Generate KF Splits
	tuning_splits, tuning_subjects_arr, huh_hie_subjs = gen_kf_splits_and_arr(df_huh_train, include_huh_in_training, num_kfolds, num_train_samps, add_nosz_subjs_to_train)

	print('tuning_subjects_arr:', tuning_subjects_arr)

	if regen_kf_data:
		# df_train contains HUH
		# df_tuning contains UCSF

		# Merge training and tuning datasets
		df_huh_train["preictal"] = df_huh_train["preictal"].astype(str)
		df_huh_train["seizing"] = df_huh_train["seizing"].astype(str)

		df_master_train = df_huh_train.copy()

		# Remove bad subjects
		huh_subjects = get_huh_subjs(df_master_train.subject_id.values)

		# Generate selected labels and safe subjects
		df_master_train = df_master_train.sort_values(['subject_id', 'abs_time_idx'])

		gradient_labeling = False
		if gradient_labeling:
			grp_lbls, snd_lbls, newPI_lbls, ii_safe_subjs, postnatal_age_lbls, postnatal_time_without_sz_lbls = utils.get_lbls(df_master_train)
		else:
			# this segments by seizure epochs
			grp_lbls, snd_lbls, newPI_lbls, ii_safe_subjs, postnatal_age_lbls, postnatal_time_without_sz_lbls = utils.get_lbls_sz_epochs(df_master_train, one_seizure=one_seizure_mode)

		## Apply new label mapping
		df_master_train['ii_safe'] = np.zeros((len(df_master_train),))
		for grp, lbls in grp_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'ii_safe'] = lbls


		## Apply new SND label mapping
		df_master_train['snd_grp'] = np.zeros((len(df_master_train),))
		for grp, lbls in snd_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'snd_grp'] = lbls

		## Apply new PI_Lbls label mapping
		for grp, lbls in newPI_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'preictal'] = lbls

		## Apply pna_szf label mapping
		df_master_train['pna_szf'] = np.zeros((len(df_master_train),))
		for grp, lbls in postnatal_time_without_sz_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'pna_szf'] = lbls

		## Apply pna label mapping
		df_master_train['pna'] = np.zeros((len(df_master_train),))
		for grp, lbls in postnatal_age_lbls.items():
			df_master_train.loc[df_master_train['subject_id'] == grp,'pna'] = lbls

		# tSMOTE
		tSMOTE_subjs = []
		if tSMOTE_mode:
			df_master_tuning_smote = utils.get_smote(df_master_train, feature_column_names, one_seizure=one_seizure_mode)
			tSMOTE_subjs = df_master_tuning_smote.subject_id.unique()
			df_master_train = df_master_train.append(df_master_tuning_smote, ignore_index=True)


		if add_nosz_subjs_to_train:
			# df_nosz = get_df_nosz()
			df_cork = get_df_cork()
			numel_df_master_train = len(df_master_train)
			# df_master_train = df_master_train.append(df_nosz, ignore_index=True)
			# df_master_train = df_master_train.append(df_cork, ignore_index=True)
			df_master_train = pd.concat([df_master_train, df_cork], ignore_index=True)

			# FOR II_SAFE BUG (ii_safe was created to identify good ii training segments on patients with seizures; for patients with no seizures, this hsould default to 1)
			df_master_train.loc[numel_df_master_train:, ['ii_safe']] = 1


		print('all subjects:', df_master_train['subject_id'].unique())
		print('num subjects:', len(df_master_train['subject_id'].unique()))

		# Check labeling:
		print('sanity check:', np.sum(df_master_train.ii_safe[df_master_train.preictal == 2]))

		# Final data preprocessing (merge HUH and UCSF)
		ii_safe_subjs_huh = [s for s in ii_safe_subjs if (len(s)==3 and s not in huh_hie_subjs)]
		print(ii_safe_subjs_huh)


		tuning_td_subjs = []
		gen_kf_dfs(df_master_train, ii_safe_subjs_huh, tSMOTE_subjs, tuning_splits, tuning_subjects_arr, tuning_td_subjs, allowed_augments, all_safe_cols, kfolds_dir)


	return tuning_splits, feature_column_names



def get_kf_splits(tuning_splits, manual_splits=False):

	num_splits = len(tuning_splits)
	new_splits = []

	if not manual_splits:
		for kfold in range(num_splits):
			tune_and_val_dx, test_dx = tuning_splits[kfold]
			train_dx, val_dx = train_test_split(tune_and_val_dx, 
												train_size=0.875,
												random_state=42)
			new_splits.append((train_dx, val_dx, test_dx))
	else:
		# This was manual for pilot data to ensure separate valid and test (for small kfold size ~ 5)
		for kfold,val_idx in zip(range(0,num_splits), [0, 1, 3, 2, 4]):
			tune_and_val_dx, test_dx = tuning_splits[kfold]
			del_idx =np.where(tune_and_val_dx==val_idx)[0]
			train_dx = np.delete(tune_and_val_dx, del_idx )
			val_dx = np.array([val_idx])
			# print((train_dx, val_dx, test_dx))
			new_splits.append((train_dx, val_dx, test_dx))

	return new_splits
	


def gen_kf_dfs(df_master_train, ii_safe_subjs_huh, pre_tSMOTE_subjs, tuning_splits, tuning_subjects_arr, tuning_td_subjs, 
				allowed_augments, all_safe_cols, kfolds_dir, add_td_subjs=False):
	
	assert 'snd_grp' in all_safe_cols
	assert 'pna_szf' in all_safe_cols
	assert 'pna' in all_safe_cols

	new_splits = get_kf_splits(tuning_splits)
	for kfold in range(0,len(new_splits)):
		train_dx, val_dx, test_dx = new_splits[kfold]

		train_td_subjs=[]
		if add_td_subjs:
			for tuning_s in tuning_subjects_arr[train_dx]:
				train_td_subjs.extend(list(filter(lambda x: tuning_s in x, tuning_td_subjs)))
		
		final_tSMOTE_subjs = []
		if 0:
			for subj in pre_tSMOTE_subjs:
				if (subj[1:] not in tuning_subjects_arr[val_dx]) and (subj[1:] not in tuning_subjects_arr[test_dx]) :
					final_tSMOTE_subjs.append(subj)

		all_train_subjs = np.hstack([tuning_subjects_arr[train_dx], train_td_subjs, ii_safe_subjs_huh, final_tSMOTE_subjs])

		print("Train subjs:", all_train_subjs)
		print("Valid:", tuning_subjects_arr[val_dx])
		print("Test:", tuning_subjects_arr[test_dx])

		# Set up train and valid data
		df_train = df_master_train[df_master_train.subject_id.apply(lambda x: x in all_train_subjs)]
		df_train = df_train[df_train.augmentation.apply(lambda x: x in allowed_augments)]
		# df_train = df_train[df_train.ii_safe.apply(lambda x: x == 1)] # this is for HIE analysis
		df_train = df_train[all_safe_cols]

		print("\tTrain subjs:", df_train['subject_id'].unique())

		df_train['preictal'] = df_train['preictal'].astype(int)
		df_train['subject_id'] = df_train['subject_id'].astype(int)

		# df_t_fn = kfolds_dir + 'kfolds_all_noTD/train_kf' + str(kfold) + '.hdf'
		df_t_fn = f"{kfolds_dir}/train_kf{kfold}.hdf"
		df_train.to_hdf(df_t_fn, 'data', mode='w')

		df_val = df_master_train[df_master_train.subject_id.apply(lambda x: x in tuning_subjects_arr[val_dx])]
		df_val = df_val[df_val.augmentation == "none"] 
		# df_val = df_val[df_val.ii_safe.apply(lambda x: x == 1)]
		df_val = df_val[all_safe_cols]

		df_val['preictal'] = df_val['preictal'].astype(int)
		df_val['subject_id'] = df_val['subject_id'].astype(int)

		# df_v_fn = 'kfolds_all_noTD/valid_kf' + str(kfold) + '.hdf'
		df_v_fn = f"{kfolds_dir}/valid_kf{kfold}.hdf"
		df_val.to_hdf(df_v_fn, 'data', mode='w')


		df_test = df_master_train[df_master_train.subject_id.apply(lambda x: x in tuning_subjects_arr[test_dx])]
		df_test = df_test[df_test.augmentation == "none"] 

		# Inject nosz subjs to test only (not train)
		# add_nosz_to_test = False
		# if add_nosz_to_test:
		# 	print("Adding nosz subjs")
		# 	print("Original df_test len:", len(df_test))
		# 	print("Preictal percent:", np.sum(df_test.preictal)/len(df_test))

		# 	df_test = df_test.append(get_df_nosz(), ignore_index=True)
		# 	print("Test subjs:", df_test.subject_id.unique())
		# 	print("New df_test len:", len(df_test))
		# 	print("Preictal values:", df_test.preictal.unique())
		# 	print("Preictal percent:", np.sum(df_test.preictal)/len(df_test))


		# df_test = df_test[df_test.ii_safe.apply(lambda x: x == 1)]
		df_test = df_test[all_safe_cols]
		df_test['preictal'] = df_test['preictal'].astype(int)
		df_test['subject_id'] = df_test['subject_id'].astype(int)

		# df_test_fn = 'kfolds_all_noTD_cork_huh_ucsf/test_kf' + str(kfold) + '.hdf'
		df_test_fn = f"{kfolds_dir}/test_kf{kfold}.hdf"

		df_test.to_hdf(df_test_fn, 'data', mode='w')


