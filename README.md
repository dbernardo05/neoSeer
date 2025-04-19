# Short‑Horizon Neonatal Seizure Prediction (neoSeer)

Deep‑learning code, configs, and utilities for our paper&nbsp;⬇️

> **Short‑Horizon Neonatal Seizure Prediction Using EEG‑Based Deep Learning**  
> Jonathan Kim MD, Edilberto Amorim MD, Vikram R. Rao MD PhD, Hannah C. Glass MD MAS, Danilo Bernardo MD\*  
> (Submitted 2025)

---

## Significance
Neonatal seizures carry high morbidity. Existing seizure prediction approaches provide *static* risk scores spanning hours to days.  
We tackle *static,* *minute‑scale* (“short‑horizon”) seizure prediction using quantitative EEG and DL.
We achieve AUROC ≈ 0.80 at a 3 min SPH / 7 min SOP with modest false‑alarm rates.

---

## Approach and architecture at a glance
Experiments are orchestrated with Hydra YAML configs, trained via fastai’s learner loop atop PyTorch, and draw models from tsai’s time‑series zoo (or custom). Pre‑processing, multimodal EEG + clinical fusion, and uncertainty estimation utilities sit alongside the main script. CPU, CUDA, or Apple‑silicon compatible.

---

## Repo highlights

| Folder / file | Contents |
|---------------|------------------|
| `conf/` | Hydra YAML configs (data paths, model zoo, hyper‑params) |
| `run_tsai.py` | **Main training / eval script** |
| `preproc.py` | Data‑wrangling & feature‑extraction helpers |
| `mixed_patch.py` | Light wrapper for multimodal (EEG + clinical) fusion |
| `results/` | Outputted metrics, predictions, logs |
| `custom_models/` | Custom models |
| `notebooks/` | Optional exploratory notebooks and calibration curves |

> **Model zoo**  
> Temporal convolutional network (TCN), InceptionTime, ResNet, OmniScaleCNN, TST, ConvLSTM, and lightweight Transformer. Add more prebuilt time-series AI (tsai) models (https://timeseriesai.github.io/tsai/) by adding models to `MODEL_REGISTRY` in `run_tsai.py`, or add your own custom models to `custom_models/` folder.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/<user>/neoSeer.git
cd neoSeer

# 2. Create env (Tested on Python 3.10, PyTorch ≥ 2.2, fastai v2, tsai v0.3, Hydra 1.2)
pip install -r requirements.txt

# 3. Point configs to your QEEG dataset root (default is ./data)

# 4. Generate train/valid/test splits
python run_preproc.py

# 5. Example usage-train a single epoch with ConvLSTM on a single k-fold
python run_tsai.py kfold=0 archs=convLSTM preictal_duration=5 max_epochs=1

# 6. For complete run:
python run_tsai.py -m max_epochs=10

```

Results land in `multirun/{date}/{time}/results/`, including performance metrics plus prediction pickles for downstream analyses.

---

## Config (Hydra)

All exp knobs live in `conf/`:

```yaml
# conf/config.yaml   ← umbrella file
...
kfold: 0
batch_size: 8
max_epochs: 10
preictal_duration: 5        # minutes
feature_classes: ['all']
...
```

Override params from the CLI:

```bash
python run_tsai.py max_epochs=20 
```

## Data

We rely on **two open neonatal EEG seizure corpora** from Helsinki University Hospital and Cork University Maternity Hospital (DOI in manuscript). 
Raw EDFs **not included**—please download from the source (links below) and run QEEGfeats code to generate QEEG features (https://github.com/dbernardo05/qeegfeats)
* HUH - https://zenodo.org/records/4940267
* Cork - https://zenodo.org/records/7477575
---

## Extensible

* **Add a model** – drop your PyTorch backbone in `models/` and register it. The `build_ts_model` helper automatically matches input channels & output classes.  
* **Multimodal fusion** – enable `multimodal_mode=true` to concatenate clinical tabular variables through `mixed_patch.py`.  
* **Uncertainty quantification** – set `num_UQMC_runs>1` for MC‑dropout ensembling.  

---

## Acknowledgements

We thank the Helsinki & Cork teams for open‑sourcing their neonatal EEG datasets.

---

## 📧 Contact

**Dan Bernardo** – dbernardoj (at) gmail.com
