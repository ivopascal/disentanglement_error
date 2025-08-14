# Disentanglement Error

Implementation of the **Disentanglement Error** metric introduced in the paper:

[**"Measuring Uncertainty Disentanglement Error in Classification"**](https://arxiv.org/abs/2408.12175)
by Ivo Pascal de Jong, Andreea Ioana Sburlea, Matthia Sabatelli & Matias Valdenegro-Toro

This repository provides:
- Core Python implementation of the **Disentanglement Error** metric.
- Example usage and experiments via Jupyter notebooks.

The experiments from the paper are not included in this repository. For the experiments please refer to [github.com/ivopascal/uq_disentanglement_comparison](https://github.com/ivopascal/uq_disentanglement_comparison)

---

## Installation

Install the latest version from PyPI:

```bash
pip install disentanglement-error
```

Or install directly from source:
```bash
git clone https://github.com/ivopascal/disentanglement_error.git
cd disentanglement_error
pip install -e .
```

## Quick start
Here’s a minimal example of how to compute the Disentanglement Error for a model:
```python
from disentanglement_error.disentangling_model import DisentanglingModel
from disentanglement_error.error_metric import calculate_disentanglement_error

class MyModel(DisentanglingModel):
    def __init__(self):
        super().__init__()
        # TODO: Your model initialization logic here.

    def fit(self, X, y):
        # TODO: How your model is trained goes here
        # Keep in mind that fit() will be called
        # multiple times for multiple runs.

    def predict_disentangling(self, X):
        # TODO: Implement an inference pass that
        # returns predictions and uncertainties for a batch.
        predictions = ...
        aleatoric_uncertainties = ...
        epistemic_uncertainties = ...
        
        return predictions, aleatoric_uncertainties, epistemic_uncertainties

X, y = collect_my_dataset()
disentanglement_error = calculate_disentanglement_error(X, y, MyModel(), return_json=False)
```
---
## Inspection and Parameter setting
To gain further insights into the experiment, you can return `json` results 
which can be transformed into a Pandas DataFrame for easy handling. 
```python
from disentanglement_error.util import json_results_to_df
disentanglement_error, result_json, config_json = calculate_disentanglement_error(X, y, MyModel(), return_json=True)
df = json_results_to_df(result_json, config_json)
df.drop("Run_Index", axis=1).groupby(["Experiment", "Percentage"]).mean().groupby(['Experiment']).plot() # Simple plotting
```
From this inspection you can check whether the experiments worked properly. You should see:

1. Score increases with dataset size logarithmically.
2. Score decreases with label noise mostly linear.
3. These effects are much greater than noise.

Based on these graphs (or computational constraints) you can modify the parameters of the experiment:
```python
from disentanglement_error.util import json_results_to_df
kw_config = {    
    "dataset_sizes": [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0],
    "label_noises": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "n_runs": 5
}
disentanglement_error, _, _= calculate_disentanglement_error(X, y, MyModel(), kw_config=kw_config)
```
---
## Examples

Explore the Jupyter notebooks for hands-on examples:

- `examples/CIFAR10_it_demo.ipynb` – Demo of Information Theoretic disentangling on the CIFAR10 dataset. 
- `examples/tabular_it_demo.ipynb` - A demo on tabular data. This is ideal for testing because it is relatively quick to train.
- `examples/regression_example.ipynb` - A demo on a regression dataset. The experiments generalise to regression.

---

## Citation

If you use this implementation in your work, please cite the original paper:
```text
@article{de2024disentangled,
  title={Measuring Uncertainty Disentanglement Error in Classification},
  author={de Jong, Ivo Pascal and Sburlea, Andreea Ioana, Sabatelli, Matthia and Valdenegro-Toro, Matias},
  journal={arXiv preprint arXiv:2408.12175},
  year={2024}
}
```
---
## Contact

If you have any questions, please contact Ivo Pascal de Jong at ivo.de.jong@rug.nl, or open an issue on Github. 

--- 
### Enjoy disentangling your uncertainties!
