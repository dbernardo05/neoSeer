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
> We register classic TCN, InceptionTime, ResNet, OmniScaleCNN, TST, a custom ConvLSTM, and a lightweight transformer. Add your own by editing `MODEL_REGISTRY` in `run_tsai.py`.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/<user>/neoSeer.git
cd neoSeer

# 2. Create env (Tested on Python 3.10, PyTorch ≥ 2.2, fastai v2, tsai v0.4, Hydra 1.2)
pip install -r requirements.txt

# 3. Point configs to your QEEG dataset root (default is ./data)

# 4. Generate train/valid/test splits
python run_preproc.py

# 5. To test, train a single 5‑minute‑preictal-window ConvLSTM on fold 0
python run_tsai.py \
  kfold=0                                \
  archs=conv_lstm                        \
  preictal_duration=3                    \
  max_epochs=1

# 6. For complete run:
python run_tsai.py \         
  -m \
  max_epochs=10

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
