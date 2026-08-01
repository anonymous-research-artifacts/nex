# NEX core implementation

This anonymous repository contains the four core components of NEX:

1. first-positive neuron novelty;
2. sticky-HMM temporal segmentation;
3. Explore-to-Exploit reuse credit; and
4. Good-Mass scoring.

## Environment

The package has been tested with:

- Python 3.12.9
- NumPy 2.2.6
- scikit-learn 1.9.0
- hmmlearn 0.3.3

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

Run the included sparse-cache example:

```bash
./run_example.sh
```

The included weights were learned on the complete calibration cache used for
the archived run. The small cache contains one response for checking the
scoring path; a model-level NEX score is the mean Good-Mass Fraction over the
complete mini set, as defined in Eq. 11.

For another cache, run the complete candidate-specific self-calibration and
scoring pipeline with its path:

```bash
./run.sh /path/to/cache ./outputs
```

This writes:

```text
outputs/
  weights.npz
  response_scores.npy
  summary.json
```

No dataset or machine path is embedded in the implementation. The Python
entry point also exposes separate training and scoring commands:

```bash
python run_nex.py train \
  --cache-dir /path/to/calibration-cache \
  --weights-out ./outputs/weights.npz

python run_nex.py score \
  --cache-dir /path/to/candidate-cache \
  --weights ./outputs/weights.npz \
  --output-dir ./candidate-score
```

## Cache schema

`learn_nex_weights(cache_dir)` expects the sparse row cache used by the method:

```text
cache_dir/
  rows/
    sample_row_ptr.int64
    row_ptr.int64
    token_row_ptr.int64
    keys.uint32
    w_sum.float16
  base/
    row_ptr.int64
    keys.uint32
    w_sum.float16
```

A packed neuron key is `(layer << 16) | unit`. The reserved layer value 255 is
ignored. The released implementation uses the paper defaults `rho=0.95` and
`min_run=2`.

Cache construction is inference-backend-specific. This release consumes the
sparse interface above.

## Python API

```python
from pathlib import Path

from nex_core import learn_nex_weights, score_cache

cache_dir = Path("/path/to/an/anonymized/cache")
weights = learn_nex_weights(cache_dir)
nex_score = score_cache(cache_dir, weights)
print(nex_score)
```

The score is the mean of the per-response Good-Mass Fractions. Weight learning
and scoring may use the same small unlabeled cache for candidate-specific
self-calibration.

## Repository contents

```text
nex_core/novelty.py       first-positive neuron novelty
nex_core/segmentation.py  sticky-HMM E/X segmentation
nex_core/reuse_credit.py  Explore-to-Exploit reuse credit
nex_core/scoring.py       cache orchestration and Good-Mass scoring
example_cache/            small anonymized sparse-cache example
example_weights.npz       weights learned on the complete calibration cache
run_nex.py                parameterized Python command-line interface
run.sh                    one-command launcher
run_example.sh            scoring example using the archived weights
```

## License

This reference implementation is released under the MIT License. See
`LICENSE` for details.
