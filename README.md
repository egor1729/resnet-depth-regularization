# Depth sweeps of L2-regularised ResNets

Code for the experiments of the master's thesis *Validity of the Neural-ODE Limit in Deep
L2-Regularized ResNets* (ETH Zurich, 2026), supervised by Dr. Gil Kur and Prof. Andreas Krause. It reproduces

1. the main experiment, depth sweeps of two-layer residual networks trained on the MAGIC
   gamma telescope data with the penalised objective of the thesis, and
2. the analysis of pretrained open-weight transformers (forward passes and weights only).

Everything runs on a CPU.

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # depth sweeps
pip install -r requirements-open-weights.txt    # additionally, open-weight models
```

A CPU-only build of PyTorch is enough, for example
`pip install torch --index-url https://download.pytorch.org/whl/cpu` before the line
above. The MAGIC data (1.5 MB) are downloaded from the UCI repository into `data/` on
first use and checked against a checksum.

Quick check (under a minute):

```bash
pytest -q
python scripts/run_sweep.py configs/smoke.yaml
```

## 1. Depth sweeps on MAGIC

### Setting

The network is a stack of `L` two-layer blocks on the input space (`d = 10`), followed by a
linear read-out,

```
z_{l+1} = z_l + V_l phi(U_l z_l + a_l) + b_l,      phi(z) = z * sigmoid(z),
```

with hidden width 6 and no normalisation or branch multiplier. The objective is

```
BCE + lambda_res * sum_l c_l + lambda_out * 0.5 * (|W_out|^2 + |b_out|^2),
c_l = 0.5 * (|U_l|_F^2 + |a_l|^2 + |V_l|_F^2 + |b_l|^2),
```

with `lambda_res` in {2e-3, 5e-3}, the same for every depth, and `lambda_out = 1e-6`.
Training is full batch: 3000 Adam steps (learning rate 1e-3, 100 warm-up steps, cosine
decay to 5e-5, no optimiser weight decay), then up to 2000 L-BFGS iterations (history 100,
strong Wolfe line search). Depths are `L = 4, 8, 16, 32, 64` and seeds 0, 1, 2. The seed
sets both the random 80/20 train/validation split and the initialisation. The three
initialisations (zero biases) are

| config suffix | U, V | read-out |
|---|---|---|
| `small` | N(0, 0.3^2 / fan_in) | N(0, 0.3^2 / fan_in) |
| `depth_scaled` | N(0, 1 / (fan_in L)) | N(0, 1 / fan_in) |
| `near_zero` | N(0, 0.03^2 / fan_in) | N(0, 0.03^2 / fan_in) |

The six configurations are in `configs/magic_lam{2e-3,5e-3}_{small,depth_scaled,near_zero}.yaml`.

### Running

```bash
bash scripts/reproduce_magic.sh
```

runs all six sweeps (90 networks) one after the other and then writes the figures. This
takes several hours on a laptop; the deep networks dominate. A sweep can be split over
processes, since each run is stored separately and finished runs are skipped:

```bash
python scripts/run_sweep.py configs/magic_lam2e-3_small.yaml --depths 64 --seeds 0 &
python scripts/run_sweep.py configs/magic_lam2e-3_small.yaml --depths 64 --seeds 1 &
...
python scripts/run_sweep.py configs/magic_lam2e-3_small.yaml   # collects the tables
python scripts/make_figures.py --out outputs/figures_lam2e-3 \
    small=outputs/magic_lam2e-3_small depth_scaled=outputs/magic_lam2e-3_depth_scaled \
    near_zero=outputs/magic_lam2e-3_near_zero
```

`python scripts/rediagnose.py <config>` recomputes all tables of a finished sweep from the
saved models, for example after changing a diagnostic setting.

### Output

`outputs/<config name>/runs/L<L>_seed<s>/` holds `model.pt`, `layers.csv`,
`sections.csv`, `summary.json` and `train_log.csv`. The sweep directory collects them in
`run_summary.csv` (one row per network), `layer_metrics.csv` (one row per block) and
`section_metrics.csv` (accepted affine sections). `make_figures.py` writes
`layer_cost_profiles`, `depth_scaling`, `accuracy` (PNG and PDF), `depth_summary.csv`
(mean and standard deviation over seeds) and `run_summary_all.csv`.

Main columns:

| column | meaning |
|---|---|
| `c_l`, `L_c_l` | block cost and depth-scaled block cost `L c_l` |
| `cost_U`, `cost_V` | the two halves of `c_l` (balancing: they end up equal) |
| `total_residual_cost`, `max_layer_cost` | `sum_l c_l` and `max_l c_l` |
| `top1_cost_share` | `max_l c_l / sum_l c_l` |
| `effective_num_layers` | `N_eff = (sum_l c_l)^2 / sum_l c_l^2` |
| `argmax_layer_p` | position `l / L` of the heaviest block |
| `num_dead_blocks` | blocks with `c_l < 1e-3` |
| `best_affine_section_length` | longest accepted affine section, in blocks |
| `disp_mean`, `disp_max` | displacement `|z_{l+1} - z_l|` of the tracked training points |
| `jump_score` | robust outlier score of a block within its network |

### Diagnostics

All per-block quantities use the trajectories of 128 training and 128 validation points.
The affine-section scan fits an affine map `z_s -> A z_s + q` to the training points of a
section of blocks `s, ..., e-1` by least squares (ridge 1e-8) and measures the relative
error `|Z_e - A Z_s - q|_F / |Z_e - mean(Z_e)|_F`. It also measures the error on probes,
4 noisy copies of each training point with noise of standard deviation 0.01, 0.03 and
0.1, pushed through the actual blocks. A section is accepted if the error is at most 3%
on training and validation points and at most 5% on the probes. Sections with a probe
error between 5% and 10% are also accepted and labelled `weak_affine_section`. The
minimal section length is `max(3, round(0.1 L))`. The scan first splits the network at
blocks whose jump score is at least 2.5 and which are not affine on their own, then
repeatedly takes the longest accepted section of the longest remaining segment. The jump
score is the sum of robust z-scores (median and median absolute deviation over the
blocks of one network) of `log c_l`, `log max displacement`,
`log |V_l|_op |U_l|_op` and the largest change of log pairwise distances.

## 2. Open-weight models

```bash
bash scripts/reproduce_open_weights.sh
```

analyses PleIAs/Baguettotron (80 layers), PleIAs/Monad (64), HuggingFaceTB/SmolLM2-135M
(30, with intermediate checkpoints at steps 240k, 960k, 1.68M) and EleutherAI/pythia-160m
(12, checkpoints at steps 0, 1000, 16000, 64000), each also with a fresh initialisation of
its configuration (`@init`, default Hugging Face initialisation with seed 0). It downloads
about 3 GB of weights and the WikiText-103 data set. Single models:

```bash
python scripts/run_open_weights.py EleutherAI/pythia-160m EleutherAI/pythia-160m@step0
python scripts/open_weights_table.py outputs/open_weights
```

`name@revision` is a branch or tag of the Hugging Face repository. No training is done.

What is measured (`resnet_theory/open_weights.py`, `resnet_theory/anatomy.py`):

- The residual stream `h_0, ..., h_L` (inputs of the blocks and the output of the last
  block, before the final norm) on windows of 256 tokens from the WikiText-103 validation
  set. Each token is one trajectory. The first position and tokens whose stream norm
  exceeds 4 times the layer median at some layer (massive activations) are excluded.
- Affine fits of every section on 48 windows, evaluated on 16 held-out windows. The error
  is normalised by the spread of the increment `h_e - h_s`, not of `h_e`, so 0 means
  affine and 1 means no better than a constant. The scan of part 1 is run with thresholds
  of 30% (train and held-out) and 50% (probes that perturb the stream and rerun the model).
- Norm escape on 32 windows: median stream norms, absolute step sizes and path length,
  in units of the median stream norm `|h_1|` after the first block.
- Contraction and expansion of the fitted one-block maps (sums of negative and positive log
  singular values) after dividing out the growth of the median stream norm.
- Weights: MLP costs with the norm gains folded into the following matrices, raw MLP
  costs, gains, Jacobians of the MLP branch `h -> mlp(norm(h))` at 32 tokens, and the MLP
  cost minimised over the two rescaling symmetries of a SwiGLU block.

`<tag>__summary.csv` collects the quantities quoted in the thesis. The bulk is all
blocks except the first and the last.

| column | meaning |
|---|---|
| `affine_error_bulk` | median increment-normalised error of one-block fits, held out |
| `norm_growth_bulk` | `|h_{L-1}| / |h_1|` (median norms) |
| `step_bulk`, `path_bulk` | median step and path length over the bulk, in units of `|h_1|` |
| `last_block_contraction_over_bulk` | contraction of the last block over the bulk median |
| `mlp_jacobian_bulk_sum` | sum over the bulk of the median Jacobian norm of the MLP branch |
| `removable_share_bulk_median` | share of the MLP cost removed by rebalancing (SwiGLU) |
| `mlp_gain_bulk_median` | median norm gain before the MLP |
| `raw_mlp_cost_bulk_median` | MLP cost per block before folding the gains |

## Reproducibility

Given the same library versions and hardware, a run is deterministic. Across PyTorch
versions or CPUs the results differ in the later digits, and the long L-BFGS phase can
amplify such differences, so individual runs may differ slightly from the numbers in the
thesis while the averages over seeds agree.

## Layout

```
resnet_theory/
  model.py         residual network, block costs
  data.py          MAGIC loading and split
  train.py         objective, Adam + L-BFGS
  diagnostics.py   per-block metrics, jump score, affine-section scan
  sweep.py         depth sweeps and their output tables
  figures.py       combined figures and depth tables
  open_weights.py  residual-stream analysis of pretrained transformers
  anatomy.py       gains, Jacobians and rebalancing of pretrained blocks
scripts/           command-line entry points
configs/           the six sweep configurations and a smoke test
tests/             fast checks
```

## Data

MAGIC Gamma Telescope, UCI Machine Learning Repository (R. K. Bock, 2004,
https://doi.org/10.24432/C52C8B), CC BY 4.0, downloaded by `resnet_theory/data.py`.
WikiText-103 (Merity et al., 2017) is
loaded through the `datasets` package.

## License

The code is released under the MIT license.
