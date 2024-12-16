# Reproduction Instruction

This document describes how to reproduce the results of [Yoshizawa *et al*. *ChemRxiv*, 2024.](https://doi.org/10.26434/chemrxiv-2024-dh681)
The reproduction methods are described below for each section of the paper.


## Results in Section 2.2: Multi-objective optimization avoiding reward hacking in drug design.

Please use `config/setting_dyramo.yaml` as the configuration file for `run.py`, as follows.

```bash
python run.py -c config/setting_dyramo.yaml
```

To reproduce the results in the paper, perform Bayesian optimization calculations 5 times with different random seeds.
The values of random seeds are as follows: `2273, 6992, 7886, 9766, 9793`. (This applies to subsequent sections as well.)
Set the random seed in the configuration file as shown below.

```yaml
BO:
  seed: 2273
```

In the paper, random exploration is conducted as a comparison to exploration by Bayesian optimization.
For random exploration, set the total number of search iterations in `num_random_search` and set `num_bayes_search` to 0 as shown below.

```yaml
BO:
  num_random_search: 40
  num_bayes_search: 0
```

Furthermore, the paper also compares DyRAMO with molecular generation without considering the applicability domains (ADs).
To achieve this, add the following options to the `reward_function` option in the configuration file.
Setting the option `reward_function.AD.[property].weight` to 0 will ignore the AD for that `[property]`.

```yaml
reward_function:
  AD:
    EGFR:
      weight: 0
    Metabolic_stability:
      weight: 0
    Permeability:
      weight: 0
```


## Results in Section 2.3: Prioritizing properties in reliability adjustment.

Within `config/setting_dyramo.yaml`, there are settings related to prioritization as shown below.

```yaml
DSS:
  EGFR:
    priority: high
  Metabolic_stability:
    priority: low
  Permeability:
    priority: low
```

The paper discusses four prioritization patterns: no priority pattern (Pattern 1), inhibitory activity prioritized pattern (Pattern 2), metabolic stability prioritized pattern (Pattern 3), and permeability prioritized pattern (Pattern 4).
The priority settings for each pattern are shown in the table below.

|  Pattern  |  EGFR  | Metabolic stability | Permeability |
| --------- | ------ | ------------------- | ------------ |
| Pattern 1 | `low`  | `low`               | `low`        |
| Pattern 2 | `high` | `low`               | `low`        |
| Pattern 3 | `low`  | `high`              | `low`        |
| Pattern 4 | `low`  | `low`               | `high`       |

Please set the priority for each property according to the table and run calculations as follows:

```bash
python run.py -c config/setting_dyramo.yaml
```


## Results in Section 2.4: Molecular design in a situation where the overlap of ADs is not expected.

Please modify lines 10 and 11 in `reward/DyRAMO_reward.py` as follows:

```py
LGB_MODELS_PATH = 'data/lgb_models_drop_overlap.pkl'
FEATURE_PATH = 'data/fps_drop_overlap.pkl'
```

The above refers to the prediction models trained with partially removed data and the fingerprints of the training data.
After making this change, run calculations as follows:

```bash
python run.py -c config/setting_dyramo.yaml
```


## Results in Section 3: Molecular design when the number of properties to be optimized is 13.

Please use `config/setting_dyramo_kinase.yaml` as the configuration file for `run.py`, and execute the following:

```bash
python run.py -c config/setting_dyramo_kinase.yaml
```

This configuration file refers `reward/DyRAMO_kinase_reward.py` to compute a reward function.

## Results in Supplementary Information: Molecular design using Tanimoto combo as a criterion for defining ADs.

In this script, shapescreen is used to calculate similarity. Please install CDPKit according to the following documentation: https://cdpkit.org/v1.1.1/installation.html

Then please prepare `config/setting_dyramo_shapescreen.yaml` as the configuration file for `run.py`, and execute the following:

```bash
python run.py -c config/setting_dyramo_shapescreen.yaml
```

This configuration file refers `reward/DyRAMO_shapescreen_reward.py` to compute a reward function.

For more details about shapescreen, please refer to the official documentation: https://cdpkit.org/v1.1.1/applications/shapescreen.html

## Additional Informations

For a more accurate reproduction of the results (excluding results using the Tanimoto combo), please checkout to the following commit: [b6c19e7](https://github.com/ycu-iil/DyRAMO/tree/b6c19e72d4351c8e26b2decfd36ddf3a862e0d3f).
Follow the README there to set up the environment with python=3.7 and chemtsv2=0.9.11, and then run the calculations.
While the overall results will not be significantly different, this environment can make the generated molecules exactly match those in the paper.
