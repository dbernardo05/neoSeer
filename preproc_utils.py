
import collections
import itertools
import numpy as np
import pandas as pd
import sys

def _zero_runs(a):
	iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
	return ranges


def concatenate_subsegments(df_unfiltered, allowed_augments, debug_reslice=False):
	# Iterate through subjects, and generate absolute time.
	# Note this needs to be done per augmentation
	compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

	df_unfiltered.sort_values(['group', 'augmentation', 'subsegment', 'time_idx'], ascending=[True, True, True, True], inplace=True)
	df = df_unfiltered[df_unfiltered['augmentation'].isin(allowed_augments)].copy()
	del df_unfiltered

	df['abs_time_idx'] = np.nan
	for curr_subj in sorted(df.subject_id.unique().tolist()):
		if debug_reslice:
			print('#####:', curr_subj)

		curr_preictal = df[df.subject_id == curr_subj].preictal.unique()
		assert compare(curr_preictal, [0, 1, 2])

		aug_idx = 0
		abs_st = 0
		abs_nd = 0
		curr_aug_t_start = 0


		for aug in sorted(allowed_augments):
			abs_time_idx = []
			subseg_idx = 0 

			subseg_grps = df[(df.subject_id == curr_subj) & (df['augmentation'] == aug)].groupby('subsegment')
			if debug_reslice:
				print('\taug:', aug_idx, aug)
			for subseg, grp in subseg_grps:
				if debug_reslice:
					print('\t\tsubseg:', subseg)
				rel_st,rel_nd = grp.time_idx.min(),grp.time_idx.max()

				if aug_idx == 0:
					abs_st += rel_st
					abs_nd += rel_nd + 1
					c_abs_st = abs_st
					c_abs_nd = abs_nd
					abs_time_idx.extend(range(abs_st,abs_nd))
					abs_st = abs_nd
				else:
					num_idcs = rel_nd-rel_st +1
					abs_nd = abs_st + num_idcs
					c_abs_st = abs_st
					c_abs_nd = abs_nd
					abs_time_idx.extend(range(abs_st,abs_nd))
					abs_st = abs_nd

				# Debug
				if debug_reslice:
					print(subseg_idx, '-', subseg, ': ', rel_st, rel_nd,' | ', c_abs_st, c_abs_nd)
					print('\t\t', curr_aug_t_start, abs_nd)

				# Sanity Check
				assert len(abs_time_idx) == len(np.unique(abs_time_idx))
				assert np.unique(np.diff(abs_time_idx))[0] == 1

				if subseg_idx == len(subseg_grps) - 1:
					curr_aug_t_start = abs_nd
				if debug_reslice:
					print(subseg_idx, len(subseg_grps), curr_aug_t_start)
				subseg_idx+=1

			aug_idx+=1
			df.loc[(df.subject_id == curr_subj)&
						  (df.augmentation == aug),'abs_time_idx'] = np.array(abs_time_idx)

	if debug_reslice:
		nan_locations = df.isna()
		stacked_nans = nan_locations.stack()
		nan_coords = stacked_nans[stacked_nans].index.tolist()

		unique_subject_ids = set()
		unique_subject_augs = set()
		for coord in nan_coords:
			unique_subject_ids.add(df.loc[coord[0], 'subject_id'])
			unique_subject_augs.add(df.loc[coord[0], 'augmentation'])

		print('subjs w/NaN',unique_subject_ids)
		print('augmentation w/NaN',unique_subject_augs)

	df['abs_time_idx'] = df['abs_time_idx'].astype(int)
	return df


def get_lbls(df_master_train, wins_per_min=3, plot_ii_safe=True, one_seizure=False):
	MIN_ENCODER_LEN = 10
	ictal_buffer = 10
	ii_safe_min_time = 30 # minutes
	ii_periods = []
	pi_sums = []

	# Check labels
	grp_lbls = {}
	snd_lbls = {}
	postnatal_time_without_sz_lbls = {}
	i=0

	ii_safe_subjs = []

	huh_poss_hie = []

	for name, grp in df_master_train.groupby(by='subject_id'):
		print(name)

		sz = np.array([1 if n == 'True' else 0 for n in grp['seizing'].values])
		pi = np.array([int(n) for n in grp['preictal'].values])
		ii_safe = np.zeros(pi.shape)
		ii_prePI = np.zeros(pi.shape)

		# For blocking postictal periods
		ii_block = np.zeros(pi.shape)

		if np.sum(sz) == 0:
			postnatal_time_without_sz = np.arange(len(sz))
		else:
			index_of_first_sz = np.where(sz == 1)[0][0]
			postnatal_time_without_sz = np.zeros(len(sz))
			postnatal_time_without_sz[:index_of_first_sz] = np.arange(index_of_first_sz)
		postnatal_time_without_sz_lbls[name] = postnatal_time_without_sz

		if not 'True' in grp.seizing.unique():
			grp_lbls[name] = ii_safe = np.ones(pi.shape)
			ii_safe_subjs.append(name)
			continue
			
		ii_runs = _zero_runs(pi)
		ii_sums = [r[1]-r[0] for r in ii_runs]

		# if patient is in continuous seizure (>90% seizure, then disregard) 
		if np.sum(ii_sums)/len(pi) < 0.1:
			grp_lbls[name] = ii_safe
			continue
		
		# Add preictal segments
		ii_safe[pi==1] = 1
		ii_block[pi==2] = 1

		if plot_ii_safe:
			plt.figure(figsize=(15,3))

		# Plot PI
		pi_runs = _zero_runs(1 - ii_safe)
		for idx, p in enumerate(pi_runs):

			# These are ii periods that will lead to PI label prediction
			if p[1]- p[0] >= 90:
				ii_prePI[p[0] - 90: p[1]-90] = 1
			if plot_ii_safe:
				# These are PI regions based on PI period of 30 minutes
				plt.plot([p[0], p[1]],[3,3], c='purple', alpha=1)

				plt.plot([p[0] - 90, p[1] - 90],[.5,.5], c='orange', alpha=1)
			if one_seizure:
				break

		# Plot SZ and SZ Postictal states to block
		sz_runs = _zero_runs(1-sz)
		for idx, p in enumerate(sz_runs):
			# These are ii periods that will lead to PI label prediction
			ii_block[p[0]:p[1]+90] = 1
			if plot_ii_safe:
				# These are PI regions based on PI period of 30 minutes
				plt.plot([p[0], p[1]],[4,4], c='red', alpha=1)
				plt.plot([p[1], p[1] + 90],[.5,.5], c='red', alpha=1)
			
		buff_ = wins_per_min*ictal_buffer
		postictal = wins_per_min

		safe_for_huh_hue_tuning = False
		
		slice_and_dice = []
		for idx, i in enumerate(ii_runs):
			if i[1]-i[0] > wins_per_min*ii_safe_min_time:
				if idx >= len(ii_runs)-2:
					# Final segment, finalize plot
					if plot_ii_safe:
						plt.plot(np.arange(len(pi)), pi)
						plt.axvline(len(ii_safe)/7)
						plt.title(name)
					
				# Label easy interictal periods, ignore the rest
				if i[1] - i[0] > ii_safe_min_time*wins_per_min:
					convergence_point = np.minimum((i[0] + i[1]) // 2, i[1]-wins_per_min*int(ii_safe_min_time*(2/3)))
					ii_safe[i[0] + postictal: convergence_point] = 1
		
				if name in ['034', '036', '062', '064']:
					# These start with seizure and no subsequent seizures, therefore no true preictal, just use interictal
					if plot_ii_safe:
						plt.plot([i[0] + postictal, convergence_point],[3,3], c='g', alpha=1)
					print('\tskipping because this subject starts with seizure.')
					continue
				
				if name in ['013', '023', '050', '068', '076', '077']:
					print('\tskipping because subjects have status epilepticus (so no preictal or interictal); or flat')
					continue
					
				# Label easy preictal periods
				# This is probably end of file (pre-augmentation) for file with seizure at beginning
				# so skip
				if idx == len(ii_runs) and pi[0] == 1:
					print('\tskipping end of file.')
					continue

				# Don't Label last epoch, because likely just tail interictal
				if idx == len(ii_runs) - 1:
					continue

				# Have determined these subject has at least one segment safe for tuning set
				safe_for_huh_hue_tuning = True
				
				slice_and_dice.append(i)

				if plot_ii_safe:
					# These are regions in middle of interictal region that are hard to label.
					plt.plot([i[0] + postictal, convergence_point],[3,3], c='g', alpha=1)
					# These are regions in middle of beginning and end of ii region that are hard to label.
	#                 plt.plot([i[1] - 15*wins_per_min - 5*wins_per_min, i[1]],[3,3], c='r', alpha=1)

			if one_seizure:
				break

		# Final adjustments
		ii_safe[ii_safe==0] += ii_prePI[ii_safe==0]
		ii_safe[ii_safe==1] -= ii_block[ii_safe==1]

		if plot_ii_safe:
			plt.plot(ii_safe*0.5-.6)
			plt.savefig('subj_tracings/{}.png'.format(name))

		grp_lbls[name] = ii_safe
		# assert (0 in np.unique(ii_safe)) and (1 in np.unique(ii_safe))
		ii_safe_subjs.append(name)
		
		if len(name) == 3:
			if '0' not in grp.preictal.unique():
				# These are subjects with no interictal, just preictal
				continue
			if '1' not in grp.preictal.unique():
				# These are subjects with seizure at beginning but no preictal ('034')
				continue
			if safe_for_huh_hue_tuning:
				huh_poss_hie.append(name)
		
		# New SND Group Code
		snd_runs = _zero_runs(1-ii_safe)
		snd = np.zeros(ii_safe.shape,dtype='int')
		snd_grp = 0
		for idx, p in enumerate(snd_runs):
			# These are ii periods that will lead to PI label prediction
			snd_id = int(int(name)*1E6 + snd_grp)
			snd[p[0]:p[1]] = snd_id
			snd_grp+=1
			# print('\t', p, snd_id)
		snd_lbls[name] = snd


	return grp_lbls, snd_lbls, ii_safe_subjs, postnatal_time_without_sz_lbls



def get_lbls_sz_epochs(df_master_train, wins_per_min=3, ii_safe_min_duration=20, plot_ii_safe=False, debug_mode=False, one_seizure=False):

	ii_periods = []
	pi_sums = []

	# Check labels
	grp_lbls = {}
	snd_lbls = {}
	newPI_lbls = {}
	postnatal_time_without_sz_lbls = {}
	postnatal_age_lbls = {}
	i=0

	ii_safe_subjs = []

	huh_poss_hie = []

	print('# def get_lbls_sz_epochs():')
	for name, grp in df_master_train.groupby(by='subject_id'):
		if one_seizure and len(name) == 3:
			# Ignore HUH subjs for one seizure mode
			continue
		print('\t\t:', name)
		if debug_mode:
			print(name, grp.abs_time_idx.iloc[0], grp.abs_time_idx.iloc[-1])
			if len(name) == 4:
				print('\t', np.unique(np.diff(grp[grp.augmentation=='none'].abs_time_idx)))

		num_augments = len(grp.augmentation.unique())
		sz = np.array([1 if n == 'True' else 0 for n in grp['seizing'].values])

		if int(name) > 100 and int(name) < 9000: 
			age_corr = time_to_EEG_df[time_to_EEG_df.Subject==int(name)].Time_to_EEG.values*3 # In minutes so convert to epoch time
			print(age_corr)
			if len(age_corr) > 0:
				if np.sum(sz) == 0:
					postnatal_time_without_sz = np.arange(len(sz)) + age_corr
				else:
					index_of_first_sz = np.where(sz == 1)[0][0]
					postnatal_time_without_sz = np.zeros(len(sz))
					postnatal_time_without_sz[:index_of_first_sz] = np.arange(index_of_first_sz) + age_corr
				postnatal_time_without_sz_lbls[name] = postnatal_time_without_sz

				postnatal_age = np.arange(len(sz)) + age_corr
				postnatal_age_lbls[name] = postnatal_age
			else:
				postnatal_age_lbls[name] = np.full((len(sz),), np.nan)
				postnatal_time_without_sz_lbls[name] = np.full((len(sz),), np.nan)	
		else:
			postnatal_age_lbls[name] = np.full((len(sz),), np.nan)
			postnatal_time_without_sz_lbls[name] = np.full((len(sz),), np.nan)

		pi = np.array([int(n) for n in grp['preictal'].values])
		new_pi = np.zeros(pi.shape)

		# label seizures in new_pi
		new_pi[pi==2] = 2

		ii_safe = np.zeros(pi.shape)

		ii_runs = _zero_runs(sz)
		ii_sums = [r[1]-r[0] for r in ii_runs]

		# Fix 20221127 -- for end interictal > beginning interictal joining
		deaug_sz = np.copy(sz) # Divide sz arr into augmented segs
		orig_sz_len = len(sz)/num_augments
		assert orig_sz_len.is_integer()
		orig_sz_len = int(orig_sz_len)
		deaug_sz = deaug_sz.reshape((num_augments, orig_sz_len))

		deaug_ii_runs = []
		for n, row in enumerate(deaug_sz):
			deaug_ii_run = _zero_runs(row) + int(n*orig_sz_len)
			deaug_ii_runs.append(deaug_ii_run)
		deaug_ii_runs = np.vstack(deaug_ii_runs)

		# Handle various cases
		if len(name) == 3:
			if '0' not in grp.preictal.unique():
				# These are subjects with no interictal, just ictal/preictal
				continue
			if '1' not in grp.preictal.unique():
				# These are subjects with seizure at beginning but no preictal ('034')
				continue
		elif name in ['034', '036', '062', '064']:
			# These start with seizure and no subsequent seizures, therefore no true preictal 
			print('\tskipping because this subject starts with seizure.')
		elif name in ['013', '023', '050', '068', '076', '077']:
			print('\tskipping because said so.')
			continue
		elif not 'True' in grp.seizing.unique():
			# All interictal, OK to keep
			grp_lbls[name] = ii_safe = np.ones(pi.shape)
			ii_safe_subjs.append(name)
			continue
		elif np.sum(ii_sums)/len(pi) < 0.1:
			# if patient is mostly seizing (90% of time, then disregard) 
			grp_lbls[name] = ii_safe
			continue

		if plot_ii_safe:
			plt.figure(figsize=(15,3))

		one_seizure_dur = None # Flag for first seizure as well
		safe_for_huh_hue_tuning = False
		for idx, i in enumerate(deaug_ii_runs):

			curr_sz_dur = i[1] - i[0]

			# Label easy interictal periods, ignore periods too close to seizure (short ii_safe duration)
			if curr_sz_dur >= ii_safe_min_duration*wins_per_min:

				if not one_seizure:
					pass
				elif not one_seizure_dur:
					pass
				elif one_seizure_dur != curr_sz_dur:
					continue
				elif one_seizure_dur == curr_sz_dur:		
					# If a repeat (augmentation) of first seizure found then continue
					pass
				else:
					sys.exit("Error at first seizure handling")

				ii_safe[i[0]: i[1]] = 1

				# Save duration
				one_seizure_dur = i[1] - i[0]

				# Have determined these subject has at least one segment safe for tuning set
				safe_for_huh_hue_tuning = True
				if plot_ii_safe:
					# These are PI regions based on PI period of 30 minutes
					plt.plot([i[0], i[1]],[1,1], c='green', alpha=1)
			# else:
			# 	print('\tskipping because safe ii period is less than 30 minutes')


		if safe_for_huh_hue_tuning:
			huh_poss_hie.append(name)
	
		# Add end of augmentation marker to ii_safe
		aug_l = []
		for augment in grp.augmentation.unique():
			aug_l.append(len(grp[grp.augmentation == augment]))

		# Make sure all elements are the same
		assert aug_l.count(aug_l[0]) == len(aug_l)
		# print(len(grp), aug_l, int(len(grp)/4))
		assert aug_l[0] == int(len(grp)/len(grp.augmentation.unique()))
		aug_seg_l = aug_l[0]
		for n in range(1, len(aug_l)):
			ii_safe[aug_seg_l*n] = -2

		grp_lbls[name] = ii_safe
		# assert (0 in np.unique(ii_safe)) and (1 in np.unique(ii_safe))
		ii_safe_subjs.append(name)
		
		# New SND Group Code
		# Segments into II and PI segments 
		snd_runs = _zero_runs(1-ii_safe)
		snd = np.zeros(ii_safe.shape,dtype='int')
		snd_grp = 0

		curr_epoch_st = 0

		for idx, p in enumerate(snd_runs):
			# These are ii periods that will lead to PI label prediction
			snd_id = int(int(name)*1E6 + snd_grp)
			print('\t', snd_id)
			print('\t\t', p[0], p[1])
			snd[p[0]:p[1]] = snd_id
			snd_grp+=1
			# print('\t', p, snd_id)

			pi_len = 45

			# Code to deal with augmentation segment transitions
			# When iterating through snd_grp's, at some point, augmentation EOF is reached
			# There is error when snd_grp following augmentation EOF looks back 'too far'
			# into prior augmentation file
			ii_safe_masked = np.copy(ii_safe)
			ii_safe_masked[:p[0]-1] = 0
			ii_safe_masked[p[1]:] = 0
			aug_transitions = np.where(ii_safe_masked==-2)[0]

			for aug_eos in aug_transitions:
				if (aug_eos >= p[0]-1) and (aug_eos < p[1]):
					curr_epoch_st = aug_eos
					# print("\t\tnew aug_eos", aug_eos)
					break

			pi_st = np.maximum(p[1] - pi_len, curr_epoch_st)

			# print('######')
			# print('\t', pi[p[0]:p[1]])

			if np.sum(pi[p[0]:p[1]])>2:
				snd_type = 'pi'
				snd_yval = [4,4]
				new_pi[pi_st:p[1]] = 1
			else:
				snd_type = 'ii'
				snd_yval = [3,3]
			


			if plot_ii_safe:				
				plt.plot([p[0], p[1]],snd_yval, alpha=1)
			# print('\tsnd {} - pi sum:'.format(snd_id), np.sum(pi[p[0]:p[1]]))
			# print('\t\t', pi[p[0]:p[1]])

		# label seizures in new_pi
		new_pi[pi==2] = 2

		snd_lbls[name] = snd
		newPI_lbls[name] = new_pi

		if plot_ii_safe:
			plt.plot(ii_safe*0.5-.6)
			plt.plot(np.arange(len(pi)), pi-2, color='red')
			plt.plot(np.arange(len(pi)), new_pi-4, color='purple')

			plt.axvline(len(ii_safe)/len(grp.augmentation.unique()))
			plt.title(name)
		
			plt.savefig('subj_tracings/{}.png'.format(name))

	return grp_lbls, snd_lbls, newPI_lbls, ii_safe_subjs, postnatal_age_lbls, postnatal_time_without_sz_lbls

