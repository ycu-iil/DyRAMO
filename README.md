# DyRAMO

DyRAMO (Dynamic Reliability Adjustment for Multi-objective Optimization) is a framework to perform multi-objective optimization while maintaining the reliability of multiple prediction models.

## How to set up

### OS Requirements

DyRAMO is supported for Linux operation systems.
DyRAMO has been tested on AlmaLinux release 9.3. 

### Python Dependencies

- python: 3.11
- chemtsv2: 1.0.3
- physbo: 2.0.0
- (optional) lightgbm: 3.2.1 (for property prediction)

#### Installation (example)

```bash
cd YOUR_WORKSPACE
python3.11 -m venv .venv
source .venv/bin/activate
pip install chemtsv2==1.0.3 physbo==2.0.0 lightgbm==3.2.1
```
This installation via pip normally finishes in about 40 seconds.

## How to run DyRAMO

### 1. Clone this repository and move into it

```bash
git clone git@github.com:ycu-iil/DyRAMO.git
cd DyRAMO
```

### 2. Prepare a reward file for molecule generation

DyRAMO employs ChemTSv2 as a molecule generator.
Here, please prepare a reward file for ChemTSv2 according to instructions on [how to define reward function in ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2/blob/c61abbc702b914a76e076d87d416cdc67d3fd517/reward/README.md).
An example of a reward file for DyRAMO can be found in [`reward/DyRAMO_reward.py`](https://github.com/ycu-iil/DyRAMO/blob/main/reward/DyRAMO_reward.py).

### 3. Prepare a setting file

Please prepare a yaml file containing the settings for both DyRAMO and ChemTSv2.
An example of a setting file can be found in [`config/setting_dyramo.yaml`](https://github.com/ycu-iil/DyRAMO/blob/main/config/setting_dyramo.yaml).
Details of the settings are described in [Setting to run DyRAMO](#settings-to-run-dyramo) section.

### 4. Run DyRAMO

Please execute `run.py` with the yaml file as an argument.
```bash
python run.py -c config/setting_dyramo.yaml
```

Expected outputs
- `run.log`: An execution log file for DyRAMO.
- `serach_history.csv`: A csv file containing information on the explored input variables and the corresponding objective variables.
- `search_result.npz`: A binary file containing information on the search results. Please refer [PHYSBO documentation](https://issp-center-dev.github.io/PHYSBO/manual/master/en/notebook/tutorial_basic.html#Serializing-the-results) for details.
- `result/`: A directory containing the results of molecule generation with ChemTSv2.

Expected run time
- Generating 10,000 molecules with a C-value of 0.01 is generally assumed to take about 10 minutes.
- In this case, this generation is set to be repeated 40 times, requiring a total of approximately 7 hours.

## Settings to run DyRAMO

The settings for DYRAMO and ChemTSv2 are described in a single yaml file.
The settings for ChemTSv2 are partially quoted here.
More details can be found in the following [link](https://github.com/molecule-generator-collection/ChemTSv2/blob/c61abbc702b914a76e076d87d416cdc67d3fd517/README.md#support-optionfunction-pushpin).
(The description of ChemTSv2 settings written here is taken from the above link.)

<table>
    <tr>
        <th>Option</th>
        <th>Suboption</th>
        <th>Descriotion</th>
    </tr>
    <tr>
        <td rowspan="1"><code>c_val</code></td>
        <td>-</td>
        <td>An exploration parameter to balance the trade-off between exploration and exploitation. A larger value (e.g., 1.0) prioritizes exploration, and a smaller value (e.g., 0.1) prioritizes exploitation.</td>
    </tr>
    <tr>
        <td rowspan="1"><code>threshold_type</code></td>
        <td>-</td>
        <td>Threshold type to select how long (<code>hours</code>) or how many (<code>generation_num</code>) molecule generation to perform per run.</td>
    </tr>
    <tr>
        <td rowspan="1"><code>hours</code></td>
        <td>-</td>
        <td>Time for molecule generation in hours per run.</td>
    </tr>
    <tr>
        <td rowspan="1"><code>generation_num</code></td>
        <td>-</td>
        <td>Number of molecules to be generated per run.</td>
    </tr>
    <tr>
        <td rowspan="1"><code>reward_function</code></td>
        <td><code>property</code></td>
        <td>Settings for calculating reward function, Dscore. Datails for setting of the Dscore parameters can be found in the following <a href="https://github.com/molecule-generator-collection/ChemTSv2/blob/c61abbc702b914a76e076d87d416cdc67d3fd517/doc/multiobjective_optimization_using_dscore.md#how-to-adjust-dscore-paramaters" >link</a>.</td>
    </tr>
    <tr>
        <td rowspan="4"><code>search_range</code></td>
        <td></td>
        <td>Search range of reliability levels for each property. Search ranges are defined by upper and lower limits and their intervals. For example, if the upper limit, lower limit, interval  are set to <code>0.9</code>, <code>0.1</code>, and <code>0.2</code>, respectively, the search range is defined as follows: <code>[0.1, 0.3, 0.5, 0.7, 0.9]</code>.</td>
    </tr>
    <tr>
        <td><code>[prop].max</code></td>
        <td>Upper limit of search range.</td>
    </tr>
    <tr>
        <td><code>[prop].min</code></td>
        <td>Lower limit of search range.</td>
    </tr>
    <tr>
        <td><code>[prop].step</code></td>
        <td>Interval of search points.</td>
    </tr>
    <tr>
        <td rowspan="3"><code>DSS</code></td>
        <td></td>
        <td>Settings for defining DSS score, an objective function in Bayesian optimization processes.</td>
    </tr>
    <tr>
        <td><code>reward.ratio</code></td>
        <td>Proportion of molecules to be evaluated in DSS. The average of the top <code>ratio</code> of rewards from the generated molecules is evaluated .</td>
    </tr>
    <tr>
        <td><code>[prop].priority</code></td>
        <td>Priority for properties in adjusting reliabilirty levels. Select one from <code>high</code>, <code>middle</code>,  and <code>low</code> for each property.</td>
    </tr>
    <tr>
        <td rowspan="4"><code>BO</code></td>
        <td></td>
        <td>Settings for Bayesian optimization with PHYSBO.</td>
    </tr>
    <tr>
        <td><code>num_random_search</code></td>
        <td>Number of random search iterations for initialization.</td>
    </tr>
    <tr>
        <td><code>num_bayes_search</code></td>
        <td>Number of search iterations by Bayesian optimization.</td>
    </tr>
    <tr>
        <td><code>score</code></td>
        <td>The type of aquision funciton. <code>TS</code> (Thompson Sampling), <code>EI</code> (Expected Improvement) and <code>PI</code> (Probability of Improvement) are available.</td>
    </tr>
    <tr>
        <td rowspan="1"><code>num_generation</code></td>
        <td>-</td>
        <td>Number of running of molecule generation at each search point.</td>
    </tr>
</table>

> [!NOTE]
> The `[prop]` and `.` represent the name of property to be optimized and nesting structure, respectively.
> For example, a description of the `search_range` parameter in yaml should be as follows.
> ```yaml
> search_range:
>   EGFR:
>     min: 0.1
>     max: 0.9
>     step: 0.01
>  ```


## How to Cite

```bibtex
@article{Yoshizawa2024,
  title = {Avoiding Reward Hacking in Multi-Objective Molecular Design: A Data-Driven Generative Strategy with a Reliable Design Framework},
  url = {https://doi.org/10.26434/chemrxiv-2024-dh681},
  DOI = {10.26434/chemrxiv-2024-dh681},
  journal = {ChemRxiv},
  author = {Yoshizawa, Tatsuya and Ishida, Shoichi and Sato, Tomohiro and Ohta, Masateru and Honma, Teruki and Terayama, Kei},
  year = {2024},
  month = jun
}
```
> [!NOTE]
> If you would like to reproduce the results of the above article, please refer to [doc/reproduction_instruction.md](https://github.com/ycu-iil/DyRAMO/blob/main/doc/reproduction_instruction.md).


## License

This package is distributed under the MIT License.

## Contact

- Tatsuya Yoshizawa (tatsuya.yoshizawa@riken.jp)
- Kei Terayama (terayama@yokohama-cu.ac.jp)
