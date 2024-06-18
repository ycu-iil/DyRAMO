import os
import requests


def download_rnn_model(conf, logger):
    path_model_json = conf['model_setting']['model_json']
    dirname = os.path.dirname(path_model_json)
    os.makedirs(dirname, exist_ok=True)
    if not os.path.exists(path_model_json):
        url = f'https://raw.githubusercontent.com/molecule-generator-collection/ChemTSv2/master/{path_model_json}'
        with open(path_model_json, 'w') as f:
            f.write(requests.get(url).text)
        logger.info(f"Downloaded: {path_model_json}")

    path_model_weight = conf['model_setting']['model_weight']
    if not os.path.exists(path_model_weight):
        dirname = os.path.dirname(path_model_weight)
        os.makedirs(dirname, exist_ok=True)
        url = f'https://raw.githubusercontent.com/molecule-generator-collection/ChemTSv2/4980a850bc2411fcdebe2adaab87609c2d75972e/{path_model_weight}'
        with open(path_model_weight, 'wb') as f:
            f.write(requests.get(url).content)
        logger.info(f"Downloaded: {path_model_weight}")

    path_token = conf['token']
    if not os.path.exists(path_token):
        dirname = os.path.dirname(path_token)
        os.makedirs(dirname, exist_ok=True)
        url = f'https://raw.githubusercontent.com/molecule-generator-collection/ChemTSv2/4980a850bc2411fcdebe2adaab87609c2d75972e/{path_token}'
        with open(path_token, 'wb') as f:
            f.write(requests.get(url).content)
        logger.info(f"Downloaded: {path_token}")


def download_filters(conf, logger):
    filter_keys = [key for key in conf.keys() if 'filter' in key]
    for key in filter_keys:
        if type(conf[key]) is bool:
            continue
        filter_name = conf[key]['module'].split('.')[-1]
        if not os.path.exists(f'filter/{filter_name}.py'):
            url = f'https://raw.githubusercontent.com/molecule-generator-collection/ChemTSv2/4980a850bc2411fcdebe2adaab87609c2d75972e/filter/{filter_name}.py'
            with open(f'filter/{filter_name}.py', 'w') as f:
                f.write(requests.get(url).text)
            logger.info(f"Downloaded: filter/{filter_name}.py")


def download_policy(conf, logger):
    os.makedirs('policy', exist_ok=True)
    if not os.path.exists('policy/policy.py'):
        url = f'https://raw.githubusercontent.com/molecule-generator-collection/ChemTSv2/4980a850bc2411fcdebe2adaab87609c2d75972e/policy/policy.py'
        with open('policy/policy.py', 'w') as f:
            f.write(requests.get(url).text)
        logger.info(f"Downloaded: policy/policy.py")
    policy_module = conf['policy_setting']['policy_module'].split('.')[-1]
    if not os.path.exists(f'policy/{policy_module}.py'):
        url = f'https://raw.githubusercontent.com/molecule-generator-collection/ChemTSv2/4980a850bc2411fcdebe2adaab87609c2d75972e/policy/{policy_module}.py'
        with open(f'policy/{policy_module}.py', 'w') as f:
            f.write(requests.get(url).text)
        logger.info(f"Downloaded: policy/{policy_module}.py")