#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import time
import logging

import fastai
import numpy as np
import pandas as pd
import torch

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# tsai/fastai imports
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.progress import CSVLogger
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat, LabelSmoothingCrossEntropyFlat

from tsai.all import *
from tsai.basics import *
from tsai.data.all import *
from tsai.models.utils import *
from tsai.models.InceptionTimePlus import *
from tsai.models.TabModel import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

# utils and patch imports
import preproc
from mixed_patch import *
from run_tsai_utils import *
from custom_models.TransformerModel_modified import *

MODEL_REGISTRY = {
	"TCN":             TCN,
	"ConvLSTM":        LSTMPlus,
	"Transformer":     TransformerModel,
	# "ConvTransformer": TransformerPlus,
	"OmniScaleCNN":    OmniScaleCNN,
	"TST":             TST,
	"InceptionTime":   InceptionTime,
	"ResNet":          ResNet,
}

@hydra.main(
	version_base="1.2",      
	config_path="conf",
	config_name="config"
)
def run(cfg: DictConfig):
	# ——— setup logging ———
	os.makedirs("results", exist_ok=True)
	logging.basicConfig(
		filename=cfg.logging.file,
		level=getattr(logging, cfg.logging.level),
		format="%(asctime)s %(levelname)s %(message)s"
	)
	log = logging.getLogger(__name__)

	orig_cwd = get_original_cwd()
	kfolds_path = os.path.join(orig_cwd, cfg.kfolds_dir)

	device = torch.device(
		"cuda" if torch.cuda.is_available() else
		"mps" if torch.backends.mps.is_available() else
		"cpu"
	)

	results = pd.DataFrame(columns=[
		"featclass","epoch_win_len","learning_rate","fold","arch","hyperparams","total params",
		"train loss","valid loss","val_acc","val_prec","val_recall","val_f1",
		"test_prAUC","test_base_prAUC","test_f1","test_rocauc","time","epochs"
	])

	# loop over features & window sizes
	for featclass in cfg.feature_classes:
		for epoch_winlen in [cfg.preictal_duration]:

			# get dataloaders
			dls, tab_dls, scaler, raw_cols, data_cols, clin_cols, pos_wt, pos_wt_v = \
				get_metadataloader(
					cfg.kfold, epoch_winlen, cfg.batch_size, kfolds_path,
					featclass,
					multimodal_mode=cfg.multimodal_mode,
					add_clinvars_mode=cfg.multimodal_mode,
					secondary_stats_mode=cfg.secondary_stats_mode,
					time_mode=cfg.time_mode,
					reuse_training_objs=cfg.reuse_training_objs,
					JK_paper_mode=cfg.JK_paper_mode
				)

			X_test, y_test_actual, _, _, clin_dfs, subj_keys = get_test_dl(
					cfg.kfold, epoch_winlen, kfolds_path,
					raw_cols, data_cols, clin_cols,
					scaler, cfg.batch_size,
					multimodal_mode=cfg.multimodal_mode,
					add_clinvars_mode=cfg.multimodal_mode,
					secondary_stats_mode=cfg.secondary_stats_mode,
					time_mode=cfg.time_mode,
					JK_paper_mode=cfg.JK_paper_mode
				)

			trainval_dls = dls if not cfg.multimodal_mode else get_mixed_dls(dls, tab_dls)

			# class weights & loss
			weights = [cfg.ii_weight / pos_wt_v, 1.0]
			class_weights = torch.FloatTensor(weights).to(device)

			metrics = [accuracy,
					   Precision(average="macro"),
					   Recall(average="macro"),
					   F1Score(average="macro")]

			# loop through architectures
			for arch_cfg in cfg.archs:
				ArchClass   = MODEL_REGISTRY[arch_cfg.name]
				arch_kwargs = dict(arch_cfg.kwargs)
				lr          = float(arch_cfg["lr"])
				patience    = int(arch_cfg["patience"])
				model = build_ts_model(ArchClass, dls=dls, **arch_kwargs)
				arch_name = model.__class__.__name__
				print("\tarch:", arch_name)
				loss_func = LabelSmoothingCrossEntropyFlat(weight=class_weights, eps=0.07)
				learn = Learner(
					trainval_dls,
					model,
					loss_func=loss_func,
					metrics=metrics,
					cbs=[
						EarlyStoppingCallback(monitor="valid_loss", min_delta=0.01, patience=patience),
						CSVLogger(append=True),
						SaveModelCallback(monitor="f1_score", comp=np.greater, min_delta=0.001),
					]
				)

				# train
				t0 = time.time()
				learn.fit_one_cycle(cfg.max_epochs, lr)
				elapsed = time.time() - t0
				vals = learn.recorder.values[-1]
				epochs_ran = len(learn.recorder.values)

				# get test preds, +/- uncertain quantififcation 
				all_preds, all_test, all_y_test_actuals = [], [], []
				for i in range(cfg.num_UQMC_runs):
					y_preds, y_test, y_test_actuals, win_keys  = [], [], [], []
					print("Predicting on test set...")

					# Testing test dataloader
					for curr_X_test, curr_y_test_actual, subj_key in zip(X_test, y_test_actual, subj_keys):
						# print("\t", subj_key, len(curr_X_test))
						test_ds = TSDatasets(curr_X_test, curr_y_test_actual)
						test_dl = DataLoader(test_ds, device=default_device(), bs=8)
						y_probas, y_test_curr = learn.get_preds(dl=test_dl, with_input=False, with_decoded=False, cbs=[])
						y_preds_curr = y_probas.numpy()[:,1]
						y_preds.append(y_preds_curr)
						y_test.append(y_test_curr.detach().numpy())
						y_test_actuals.append(curr_y_test_actual)
						win_keys.extend([subj_key]*len(y_test_curr))

					# UQ not on, then no need to stack
					if len(y_preds) > 1:
						y_preds=np.hstack(y_preds)
					if len(y_test) > 1:
						y_test=np.hstack(y_test)
					if len(y_test_actuals) > 1:
						y_test_actuals=np.hstack(y_test_actuals)

					all_preds.append(y_preds)
					all_test.append(y_test)
					all_y_test_actuals.append(y_test_actuals)

				# get mean y_pred
				if cfg.num_UQMC_runs > 1:
					mean_y_preds = np.mean(np.vstack(all_preds), axis=0)
				else:
					mean_y_preds = np.hstack(all_preds)
				# print('y_preds:', mean_y_preds[:10], mean_y_preds.shape)
				# print('y_test:', y_test[:10], len(y_test))

				# predict & eval
				pr_baseline_test = np.sum(y_test==1)/len(y_test)
				test_precision, test_recall, thresholds = precision_recall_curve(y_test, mean_y_preds)
				test_precision[np.isnan(test_precision)] = 0  # Handle nans
				test_recall[np.isnan(test_recall)] = 0  # Handle nans
				auc_pr_test = auc(test_recall, test_precision)
				fscore = (2 * test_precision * test_recall) / (test_precision + test_recall)
				ix = np.argmax(fscore)
				f1_test = fscore[ix]
				test_rocauc = roc_auc_score(y_test, mean_y_preds)

				# save to results DataFrame
				results.loc[len(results)] = [
					featclass, epoch_winlen, lr, cfg.kfold, arch_name, arch_kwargs,
					total_params(model),
					*vals,  # train loss, valid loss, acc, prec, recall, f1
					auc_pr_test, pr_baseline_test, f1_test, test_rocauc,
					int(elapsed), epochs_ran
				]

			# write out per‑fold results
			out_base = f"results/{arch_name}_kf{cfg.kfold}_{featclass}_win{epoch_winlen}"
			results.to_csv(out_base + ".csv", index=False)
			with open(out_base + "_y_preds.pkl", "wb") as f: pickle.dump(mean_y_preds, f)
			with open(out_base + "_y_test.pkl",  "wb") as f: pickle.dump(y_test,     f)
			with open(out_base + "_winkeys.pkl","wb") as f: pickle.dump(win_keys,   f)


if __name__ == "__main__":
	run()   # invoke Hydra wrapper

