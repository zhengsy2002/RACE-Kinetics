# RACE-Kinetics

Reaction-associated pretraining with frozen ESM-C and MolFormer encoders, followed by ExtraTrees regression of `log10(kcat)`.

This release reads supplied development/test assignments. It never regenerates a split from `kcat.csv`'s `fold` column. Existing projects and results are not modified.

## Evaluation protocols

| Protocol | Split file | Development | Fixed test | Inner validation |
|---|---|---:|---:|---|
| Random records | `data/splits/split_random.csv` | 25,215 | 2,802 | 10 folds |
| Homology groups | `data/splits/split.csv` | 25,215 | 2,802 | 10 component-disjoint folds |

The dataset contains 28,017 records. Assignments are matched by `sample_id`, checked against `row_index`, and audited for observation-group leakage. Homology assignments additionally require sequence SHA-256 agreement and component isolation, including between inner folds. This audit verifies the supplied grouping; it does not rerun sequence similarity search or establish an absolute pairwise identity bound.

Validation produces OOF predictions for development records only. Final fitting uses the entire development partition. Testing loads that final model and checks its training receipt before evaluating the fixed test partition. Test records are never used for fitting, early stopping or hyperparameter selection. The runner uses configured parameters; it does not perform an automatic parameter search.

These records previously contributed to model development. The new evaluation is a retrospective holdout evaluation, not previously unseen external validation. Freeze the configuration before running the test stage and report all specified training seeds.

## Model and experiments

The model uses the following architecture and feature layout:

- Frozen ESM-C 600M sequence features, followed by a 256-dimensional residual protein projection.
- Frozen MolFormer molecular features and a shared molecular adapter; the reaction projector consumes `[reactant, product, product - reactant]`.
- ER and RP bidirectional multi-positive contrastive losses, plus conflict-masked DIR binary cross-entropy. Default weights are **1:1:1**, with no learned task weights or gradient projection.
- 768 learned features, 2,048 Morgan fingerprint bits and seven assay/status fields. Elementwise cross-modal interactions are not included. Missing products are filled with the corresponding reactant side. The reserved assay column remains zero by default; `features.product_missing=true` explicitly replaces it with original product missingness, without changing the dimension.
- ExtraTrees: 200 trees, `max_features=0.3`, `min_samples_leaf=2`.

| Name | Change from full |
|---|---|
| `full` | No change |
| `no_er` | ER weight set to zero |
| `no_rp` | RP weight set to zero |
| `no_dir` | DIR weight set to zero |
| `no_environment` | Remove pH, temperature and their two missingness indicators |
| `learned_only` | Retain only the 768 learned features |

`--experiments all` selects these six experiments. Feature-only ablations reuse the full model's pretrained representation. Loss ablations train their own representation. Each representation is fixed across downstream seeds and both protocols. The default downstream seeds are 42, 43 and 44; the pretraining seed is independently specified by `training.seed` (default 42). These are three regressor seeds, not three independent pretraining runs.

The loss configuration contains only `enzyme_reaction`, `reactant_product` and `direction`. DIR filtering is controlled by `supervision.direction` (`conflict_masked_bce` by default, or `all_bce`). The `study`, `pretraining` and `loss_weighting` sections from research prototypes are not accepted.

This release removes the inactive conditional head and its forward pass. ER/RP/DIR formulas are unchanged, but removing unused random initialization and dropout operations changes the random-number stream. Retrain all comparison arms with this release; numerical reproduction of older seeded runs is not promised.

## Installation and inputs

Python 3.10 or newer is required. Install the appropriate PyTorch build for the server before installing the package:

```bash
python -m pip install -e ".[extraction]"
```

All data, split files, caches, encoder weights and outputs must remain inside this project. Paths or symlinks into other projects are rejected. Large resources are not bundled:

```text
data/
  kcat.csv
  splits/split_random.csv
  splits/split.csv
  pretrain/reactzyme/source/proteins.tsv
  pretrain/reactzyme/source/reactions.tsv
  models/esmc/
  models/molformer/
```

`proteins.tsv` requires `Entry`, `Sequence`, and `Rhea ID`. `reactions.tsv` requires `Rhea ID`, `substrate`, and `product`. Place copies of both source tables in the paths above. The input dataset and split files also belong to this project.

When feature extraction needs an encoder, it checks the local model and downloads it only if incomplete. Complete feature caches skip the encoder entirely; complete local encoders are reused without contacting the Hub. There is no separate download entry point. Downloads use pinned revisions:

| Encoder | Hugging Face repository | Revision | Local directory |
|---|---|---|---|
| ESM-C 600M HF | `biohub/ESMC-600M-hf` | `0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c` | `data/models/esmc/` |
| MolFormer | `ibm-research/MoLFormer-XL-both-10pct` | `361063d0ad524ef77cf39b08469f6be770dc550f` | `data/models/molformer/` |

Downloads need network access. `HF_ENDPOINT` can select an accessible Hub endpoint, and `HF_TOKEN` can supply credentials if needed. `downloads.auto_download=false` disables automatic downloads. Feature extraction itself loads only local files. No source-project setup or external feature cache is used.

Necessary features and their identity metadata remain in `data/cache/` and `data/features/`; encoder weights remain in `data/models/`. Temporary writes, Hugging Face downloads/modules, Torch and compilation caches share `data/tmp/run-<pid>-<unique>/`. The run directory is created on demand and library subdirectories are created only when used. Normal completion, Python exceptions, Ctrl+C and handled SIGTERM clean the current run's temporary directory; other runs and permanent features are untouched. SIGKILL, power loss or files held open by another process can leave temporary files. Interrupted download scratch files are not retained after cleanup.

Fresh preparation excludes the current target sequences and known complete reactions from ReactZyme. Missing products cannot support complete-reaction exclusion.

## Run

Inspect both supplied partitions and resource availability:

```bash
python run.py --config configs/default.json --stage check
```

Run both protocols for the full model, including preparation, extraction, pretraining, inner validation, final fitting and testing:

```bash
bash scripts/run.sh --config configs/default.json --experiments full --stage all
```

Run all six experiments:

```bash
bash scripts/run.sh --config configs/default.json --experiments all --stage all
```

For development before exposing fixed-test metrics, use separate stages:

```bash
python run.py --config configs/default.json --stage prepare
python run.py --config configs/default.json --stage extract
python run.py --config configs/default.json --experiments all --stage pretrain
python run.py --config configs/default.json --experiments all --stage features
python run.py --config configs/default.json --experiments all --stage validate
python run.py --config configs/default.json --experiments all --stage fit
python run.py --config configs/default.json --experiments all --stage test
```

Use `--protocol random` or `--protocol homology` to run one protocol. `--seeds 42` limits downstream training to one seed. Run all intended seeds before comparing seed means. `CUDA_VISIBLE_DEVICES=1` selects physical GPU 1 on a Linux server; ExtraTrees runs on the CPU.

Completed final regressors carry their training identity inside the joblib file and can be reused after input/configuration checks. Pretraining keeps a resume checkpoint while incomplete; after successful completion, its completion metadata is stored in the best weights and the resume checkpoint is removed. Development validation and test predictions can be recomputed on later invocations. Changed inputs, parameters or implementation require a new result directory (`--results-dir results/new_run`). Runs sharing the same result root are serialized by a temporary PID lock.

## Outputs

```text
data/cache/
data/features/
data/features/handcrafted/handcrafted_features.npz
data/features/handcrafted/feature_manifest.json
data/pretrain/reactzyme/prepared/
results/pretraining/<experiment>/weights/
results/<protocol>/<experiment>/seed_<seed>/
  weights/regressor.joblib
  predictions/development_oof.csv
  predictions/test.csv
  evaluation/inner_fold_metrics.csv
  evaluation/development.json
  evaluation/test.json
results/evaluation/test_metrics.csv
results/evaluation/test_summary.csv
```

Audit results and pretraining progress are printed to the terminal, without new `data/audits/` or `data/runs/` reports. Execution records stay in memory and are cleared when the run ends. Required configuration/provenance is embedded in weights. Cache metadata remains alongside feature arrays to prevent mismatched reuse. The flat handcrafted directory holds one dataset/feature-layout version at a time; incompatible contents are rejected. Only checkpoints created by this release are supported. Older run manifests, checkpoint directories and PR-named fields are not read.

All feature arrays are stored under `data/`. Inner-fold regressors are not retained; the final development-trained regressor is retained. Mean and standard deviation in `test_summary.csv` describe variation across the specified downstream seeds (`ddof=1`), not confidence intervals or inner-fold variability. Partial seed sets are marked incomplete and have no aggregate mean.

Only load trusted PyTorch/joblib checkpoints; these formats can deserialize Python objects.

## Verification and release

```bash
python -m unittest discover -s tests -v
```

Tests cover partition integrity, sequence identity, development-only fitting, test-label perturbation, prediction coverage, modified-model detection, ER/RP/DIR gradients, conflict masks, and exact CPU checkpoint resume. A CPU integration test trains full and loss-ablation models with synthetic backbone vectors, constructs all 2,823 features, checks feature-only ablations, and completes inner validation and held-out regression. These tests do not establish scientific performance or validate the pretrained backbone extraction itself.

The dataset, feature caches, model files, local configuration, and results are ignored by Git. The supplied split files are retained for reproducibility. Distribute the dataset through an appropriate repository with its hash and access conditions. Add the authors' chosen license, repository URL and manuscript citation before publication; none is invented here. Runtime package versions and source hashes are saved with each fitted experiment.
