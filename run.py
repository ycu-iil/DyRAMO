import argparse
from copy import deepcopy
import csv
from glob import glob
import itertools
import logging
import os
from pprint import pformat
import subprocess
import sys
import tempfile

from chemtsv2.misc.scaler import max_gauss, min_gauss, minmax, rectangular
import numpy as np
import pandas as pd
import physbo
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from misc.downloader import download_rnn_model, download_filters, download_policy


def get_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        usage=f'ex.) python {os.path.basename(__file__)} -c setting.yaml'
    )
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='path to a config file'
        )
    return parser.parse_args()


def load_config(conf_path):
    """Load configuration from a YAML file.

    Args:
        conf_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)
    return conf


def setup_logger(savedir):
    """Set up the logger for the script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=os.path.join(savedir, 'run.log'), mode='w')
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s ')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def download_required_files(conf, logger):
    """Download required files for molecule generation.

    Args:
        conf (dict): Configuration dictionary.
        logger (logging.Logger): Logger.
    """
    logger.info('Download required files if they do not exist...')
    download_rnn_model(conf, logger)
    download_filters(conf, logger)
    download_policy(conf, logger)


def set_config_default(conf):
    """Set default values for configuration.

    Args:
        conf (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary.
    """
    # configuration for reward function
    objectives = conf['reward_function']['property'].keys()
    conf['reward_function'].setdefault('AD', {})
    for obj in objectives:
        conf['reward_function']['AD'].setdefault(obj, {})
        conf['reward_function']['AD'][obj].setdefault('num', 1)
        conf['reward_function']['AD'][obj].setdefault('threshold', None)
        conf['reward_function']['AD'][obj].setdefault('scaler', 'step')
        conf['reward_function']['AD'][obj].setdefault('weight', 1)
    # configuration for DSS score
    conf.setdefault('DSS', {})
    conf['DSS'].setdefault('reward', {})
    conf['DSS']['reward'].setdefault('ratio', 0.1)
    conf['DSS']['reward'].setdefault('weight', len(objectives))
    for obj in objectives:
        conf['DSS'].setdefault(obj, {})
        conf['DSS'][obj].setdefault('scaler', 'max_gauss')
        conf['DSS'][obj].setdefault('priority', 'middle')
        conf['DSS'][obj].setdefault('weight', 1)
    # configuration for Bayesian optimization
    conf.setdefault('BO', {})
    conf['BO'].setdefault('num_random_search', 10)
    conf['BO'].setdefault('num_bayes_search', 30)
    conf['BO'].setdefault('score', 'EI')
    conf['BO'].setdefault('interval', 10)
    conf['BO'].setdefault('num_rand_basis', 500)
    conf['BO'].setdefault('seed', 1234)
    conf.setdefault('num_generation', 1)
    return conf


def set_scaler_params(conf):
    """Set parameters for sigmoidal scalars, mu and sigma, from the priority for each objective.

    Args:
        conf (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary with scaler parameters.
    """
    search_range_params = conf['search_range']
    fx_params = conf['DSS']
    sigma_priority = {'high': 0.15, 'middle': 0.25, 'low': 0.35}
    for k in search_range_params.keys():
        mu = search_range_params[k]['max']
        priority = fx_params[k]['priority']
        if priority not in sigma_priority.keys():
            raise ValueError(f"Set the priority from one of {list(sigma_priority.keys())}")
        sigma = sigma_priority[fx_params[k]['priority']]
        conf['DSS'][k]['mu'] = mu
        conf['DSS'][k]['sigma'] = sigma
    return conf


def define_search_space(conf):
    """Define the search space for the optimization.

    Args:
        conf (dict): Configuration dictionary.

    Returns:
        np.ndarray: Search space array.
    """
    search_range_params = conf['search_range']
    objectives = search_range_params.keys()
    search_value_list = []
    for obj in objectives:
        range_min = search_range_params[obj]['min']
        range_max = search_range_params[obj]['max']
        range_step = search_range_params[obj]['step']
        search_value = np.arange(range_min, range_max+range_step, range_step)
        search_value_list.append(search_value)
    search_space = np.array(list(itertools.product(*search_value_list)))
    return search_space


def edit_config(conf, selected_action_idx, selected_action):
    """Edit configuration for a specific action.

    Args:
        conf (dict): Configuration dictionary.
        selected_action_idx (int): Index of the selected combination of reliability levels.
        selected_action (np.ndarray): Selected combination of reliability levels.

    Returns:
        str: Path to the new configuration file.
    """
    new_conf = deepcopy(conf)
    output_dir = os.path.join(new_conf['output_dir'], 'result', f'action{selected_action_idx}')
    new_conf['output_dir'] = output_dir
    objectives = [obj for obj in new_conf['DSS'].keys() if obj != 'reward']
    for i, obj in enumerate(objectives):
        # new_conf['reward_function'][obj]['AD']['threshold'] = float(selected_action[i])
        new_conf['reward_function']['AD'][obj]['threshold'] = float(selected_action[i])

    path_to_new_conf = os.path.join(output_dir, f'action{selected_action_idx}.yaml')
    os.makedirs(output_dir, exist_ok=True)
    with open(path_to_new_conf, "w") as f:
        yaml.safe_dump(new_conf, f, sort_keys=False)
    return path_to_new_conf


def generate_molecules(path_to_conf):
    """Generate molecules based on the provided configuration.

    Args:
        path_to_conf (str): Path to the configuration file.
    """
    # load config
    with open(path_to_conf, 'r') as f:
        conf = yaml.safe_load(f)
    # set random seed for molecule generation
    rng = np.random.default_rng(1234)
    molgen_seed_list = rng.integers(low=0, high=10000, size=conf['num_generation'])
    molgen_seed_list = [int(s) for s in molgen_seed_list]
    # run molecule generation
    processes = []
    for seed in molgen_seed_list:
        tmp_conf = deepcopy(conf)
        tmp_conf['output_dir'] = f"{tmp_conf['output_dir']}/seed{seed}"
        tmp_conf['random_seed'] = seed
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            yaml.safe_dump(tmp_conf, tmp)
            path_to_tmp_conf = tmp.name
        cmd = ['chemtsv2', '-c', path_to_tmp_conf]
        process = subprocess.Popen(cmd)
        processes.append((process, path_to_tmp_conf))
    for process, path_to_tmp_conf in processes:
        process.wait()
        os.remove(path_to_tmp_conf)


def scale_objective_value(params, value):
    """Scale the objective value based on the specified scaling function.

    Args:
        params (dict): Parameters for scaling.
        value (float): Value to be scaled.

    Returns:
        float: Scaled value.
    """
    scaling = params['scaler']
    if scaling == 'max_gauss':
        return max_gauss(value, 1.0, params['mu'], params['sigma'])
    elif scaling == 'min_gauss':
        return min_gauss(value, 1.0, params['mu'], params['sigma'])
    elif scaling == 'minmax':
        return minmax(value, params['min'], params['max'])
    elif scaling == 'rectangular':
        return rectangular(value, params['min'], params['max'])
    elif scaling == 'identity':
        return value
    else:
        raise ValueError("Set the scaling function from one of 'max_gauss', 'min_gauss', 'minimax', rectangular, or 'identity'")


class Simulator:
    """Simulator class to evaluate actions."""

    def __init__(self, input_conf, search_space):
        """Initialize the simulator.

        Args:
            input_conf (dict): Configuration dictionary.
            search_space (np.ndarray): Search space array.
        """
        self.input_conf = input_conf
        self.search_space = search_space


    def __call__(self, action):
        """Evaluate the action.

        Args:
            action (list): Action (combination of reliability levels) to be evaluated.

        Returns:
            float: Evaluation result (DSS score).
        """
        # input
        selected_action_idx = action[0]

        # edit config
        path_to_new_conf = edit_config(self.input_conf, selected_action_idx, self.search_space[selected_action_idx])

        # generate molecules
        generate_molecules(path_to_new_conf)

        # load generation result
        generation_results = []
        dirname = os.path.dirname(path_to_new_conf)
        path_generation_result = glob(f'{dirname}/seed*/result_C*.csv')
        for path in path_generation_result:
            df_temp = pd.read_csv(path)
            generation_results.append(df_temp)

        # evaluate generation results
        reward_score_list = []
        for df in generation_results:
            df_unique = df.drop_duplicates(subset=['smiles'])
            fx_param_reward = self.input_conf['DSS']['reward']
            top_ratio = fx_param_reward['ratio']
            top_num = int(len(df)*top_ratio)
            top_reward_values = df_unique['reward'].sort_values(ascending=False).iloc[:top_num]
            ave_top_reward_values = top_reward_values.mean()
            reward_score_list.append(ave_top_reward_values)
        max_reward_score = np.max(reward_score_list)

        # evaluate the selected reliability level for each objective
        des_rel_levels_list = []
        params_for_dss = {k: v for k, v in self.input_conf['DSS'].items() if k != 'reward'}
        objectives = self.input_conf['search_range'].keys()
        for i, obj in enumerate(objectives):
            des_rel_levels_list.append(scale_objective_value(params_for_dss[obj], self.search_space[selected_action_idx][i]))

        # calculate DSS score
        components_to_calc_dss = np.array([max_reward_score, *des_rel_levels_list])
        weights = [self.input_conf['DSS'][k]['weight'] for k in self.input_conf['DSS'].keys()]
        weights = np.array(weights)
        dss_score = np.prod(components_to_calc_dss**weights) ** (1/weights.sum())

        # update search history
        with open(os.path.join(self.input_conf['output_dir'], 'search_history.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([selected_action_idx, *self.search_space[selected_action_idx], dss_score])

        return dss_score


def run_search(input_conf, search_space):
    """Run the Bayesian optimization search.

    Args:
        input_conf (dict): Configuration dictionary.
        search_space (np.ndarray): Search space array.

    Returns:
        physbo.search.discrete.results.history: Search history.
    """

    # configure setting for the search
    policy = physbo.search.discrete.policy(test_X=search_space)
    simulator = Simulator(input_conf, search_space)

    bo_setting = input_conf['BO']
    policy.set_seed(bo_setting['seed'])
    # The meaning of the following parameters is described here: https://issp-center-dev.github.io/PHYSBO/manual/master/en/api/physbo.search.discrete.policy.html#module-physbo.search.discrete.policy
    num_random_search = bo_setting['num_random_search']
    num_bayes_search = bo_setting['num_bayes_search']
    score = bo_setting['score']
    interval = bo_setting['interval']
    num_rand_basis = bo_setting['num_rand_basis']

    # run the search
    if num_random_search > 0:
        search_history = policy.random_search(
            max_num_probes=num_random_search,
            simulator=simulator
            )
    if num_bayes_search > 0:
        search_history = policy.bayes_search(
            max_num_probes=num_bayes_search,
            simulator=simulator,
            score=score,
            interval=interval,
            num_rand_basis=num_rand_basis
            )

    return search_history


def main():
    """Main function."""

    # parse a command line argument
    args = get_parser()
    path_to_input_conf = args.config
    input_conf = load_config(path_to_input_conf)

    # create a directory to store the search result
    if os.path.exists(input_conf['output_dir']):
        raise FileExistsError(f"{input_conf['output_dir']} already exists. Please change `output_dir` in {path_to_input_conf}.")
    os.makedirs(input_conf['output_dir'], exist_ok=True)

    # set up logger
    logger = setup_logger(input_conf['output_dir'])

    # initialize configuration
    input_conf = set_config_default(input_conf)
    input_conf = set_scaler_params(input_conf)

    # download required files for molecule generation
    download_required_files(input_conf, logger)

    # create a csv file to store search history
    objectives = [str(obj) for obj in input_conf['DSS'].keys() if obj != 'reward']
    df_search_history = pd.DataFrame(columns=['selected_action_idx', *objectives, 'dss_score'])
    df_search_history.to_csv(os.path.join(input_conf['output_dir'], 'search_history.csv'), index=False)

    # log configuration
    logger.info('========== Configuration for DyRAMO ==========')
    logger.info(f'Config file: {path_to_input_conf}')
    logger.info(f'reward_function (property):\n{pformat(input_conf["reward_function"]["property"])}')
    logger.info(f'reward_function (AD):\n{pformat(input_conf["reward_function"]["AD"])}')
    logger.info(f'search_range:\n{pformat(input_conf["search_range"])}')
    logger.info(f'DSS:\n{pformat(input_conf["DSS"])}')
    logger.info(f'BO:\n{pformat(input_conf["BO"])}')
    logger.info('=============================================')

    # run the search
    search_space = define_search_space(input_conf)
    logger.info('Search is running...')
    search_history = run_search(input_conf, search_space)
    logger.info('Search was finished.')
    best_fx, best_actions = search_history.export_all_sequence_best_fx()
    for i, (fx, action) in enumerate(zip(best_fx, best_actions)):
        logger.info(f'Best action {i+1} - {search_space[action]}, DSS score: {fx}')

    # save the search result
    output_path = os.path.join(input_conf['output_dir'], 'search_result.npz')
    search_history.save(output_path)
    logger.info(f'Search result was saved at {output_path}')

    logger.info('Done!')


if __name__ == '__main__':
    main()