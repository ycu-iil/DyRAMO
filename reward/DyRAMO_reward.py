import pickle

from chemtsv2.misc.scaler import max_gauss, min_gauss, rectangular
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem

from reward.reward import Reward


LGB_MODELS_PATH = 'data/lgb_models.pkl'
FEATURE_PATH = 'data/fps.pkl'
with open(LGB_MODELS_PATH, mode='rb') as l, \
    open(FEATURE_PATH, mode='rb') as f:
    lgb_models = pickle.load(l)
    feature_dict = pickle.load(f)


def step(x, threshold):
    if x >= threshold:
        return 1
    else:
        return 0


def scale_objective_value(params, value):
    scaling = params['scaler']
    if scaling == 'max_gauss':
        return max_gauss(value, 1.0, params['mu'], params['sigma'])
    elif scaling == 'min_gauss':
        return min_gauss(value, 1.0, params['mu'], params['sigma'])
    elif scaling == 'step':
        return step(value, params['threshold'])
    elif scaling == "rectangular":
        return rectangular(value, params["min"], params["max"])
    elif scaling == 'identity':
        return value
    else:
        raise ValueError("Set the scaling function from one of 'max_gauss', 'min_gauss', 'rectangular', 'inverted_step' or 'identity'")


def calc_tanimoto_similarity(feat_generated, feat_train):
    similarity = DataStructs.BulkTanimotoSimilarity(feat_generated, feat_train, returnDistance=False)
    similarity_sorted = sorted(similarity, reverse=True) # Sort by similarity in descending order
    return similarity_sorted


class DyRAMO_reward(Reward):
    def get_objective_functions(conf):

        def EGFR(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['EGFR'].predict(fp)[0]

        def Stab(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['Stab'].predict(fp)[0] #Stab

        def Perm(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['Perm'].predict(fp)[0]

        def EGFR_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['EGFR'])
            num = conf['reward_function']['AD']['EGFR']['num']
            return np.mean(similarity[:num])

        def Stab_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['Stab'])
            num = conf['reward_function']['AD']['Metabolic_stability']['num']
            return np.mean(similarity[:num])

        def Perm_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['Perm'])
            num = conf['reward_function']['AD']['Permeability']['num']
            return np.mean(similarity[:num])

        return [EGFR, Stab, Perm, EGFR_sim, Stab_sim, Perm_sim]


    def calc_reward_from_objective_values(values, conf):
        if None in values:
            return -1

        egfr, stab, perm, egfr_sim, stab_sim, perm_sim = values
        params = conf['reward_function']

        # AD filter
        is_in_AD = [] # 0: out of AD, 1: in AD
        is_in_AD.append(scale_objective_value(params['AD']['EGFR'], egfr_sim))
        is_in_AD.append(scale_objective_value(params['AD']['Metabolic_stability'], stab_sim))
        is_in_AD.append(scale_objective_value(params['AD']['Permeability'], perm_sim))
        weights_AD = []
        weights_AD.append(params['AD']['EGFR']['weight'])
        weights_AD.append(params['AD']['Metabolic_stability']['weight'])
        weights_AD.append(params['AD']['Permeability']['weight'])
        for i, w in zip(is_in_AD, weights_AD):
            if w == 0:
                continue
            if i == 0:
                return 0

        # Property
        scaled_values = []
        scaled_values.append(scale_objective_value(params['property']['EGFR'], egfr))
        scaled_values.append(scale_objective_value(params['property']['Metabolic_stability'], stab))
        scaled_values.append(scale_objective_value(params['property']['Permeability'], perm))
        weights = []
        weights.append(params['property']['EGFR']['weight'])
        weights.append(params['property']['Metabolic_stability']['weight'])
        weights.append(params['property']['Permeability']['weight'])

        multiplication_value = 1
        for v, w in zip(scaled_values, weights):
            multiplication_value *= v**w
        dscore = multiplication_value ** (1/sum(weights))

        return dscore
