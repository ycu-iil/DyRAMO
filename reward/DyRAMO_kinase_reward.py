import pickle

from chemtsv2.misc.scaler import max_gauss, min_gauss, rectangular
from chemtsv2.reward import Reward
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem


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

        def ERBB2(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['ERBB2'].predict(fp)[0]

        def ABL(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['ABL'].predict(fp)[0]

        def SRC(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['SRC'].predict(fp)[0]

        def LCK(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['LCK'].predict(fp)[0]

        def PDGFRbeta(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['PDGFRbeta'].predict(fp)[0]

        def VEGFR2(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['VEGFR2'].predict(fp)[0]

        def FGFR1(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['FGFR1'].predict(fp)[0]

        def EPHB4(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['EPHB4'].predict(fp)[0]

        def Stab(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['Stab'].predict(fp)[0]

        def Perm(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['Perm'].predict(fp)[0]

        def Sol(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['Sol'].predict(fp)[0]

        def hERG(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['hERG'].predict(fp)[0]

        def EGFR_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['EGFR'])
            num = conf['reward_function']['AD']['EGFR']['num']
            return np.mean(similarity[:num])

        def ERBB2_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['ERBB2'])
            num = conf['reward_function']['AD']['ERBB2']['num']
            return np.mean(similarity[:num])

        def ABL_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['ABL'])
            num = conf['reward_function']['AD']['ABL']['num']
            return np.mean(similarity[:num])

        def SRC_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['SRC'])
            num = conf['reward_function']['AD']['SRC']['num']
            return np.mean(similarity[:num])

        def LCK_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['LCK'])
            num = conf['reward_function']['AD']['LCK']['num']
            return np.mean(similarity[:num])

        def PDGFRbeta_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['PDGFRbeta'])
            num = conf['reward_function']['AD']['PDGFRbeta']['num']
            return np.mean(similarity[:num])

        def VEGFR2_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['VEGFR2'])
            num = conf['reward_function']['AD']['VEGFR2']['num']
            return np.mean(similarity[:num])

        def FGFR1_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp,feature_dict['FGFR1'])
            num = conf['reward_function']['AD']['FGFR1']['num']
            return np.mean(similarity[:num])

        def EPHB4_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(
                fp, feature_dict['EPHB4'])
            num = conf['reward_function']['AD']['EPHB4']['num']
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

        def Sol_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['Sol'])
            num = conf['reward_function']['AD']['Solubility']['num']
            return np.mean(similarity[:num])

        def hERG_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['hERG'])
            num = conf['reward_function']['AD']['hERG']['num']
            return np.mean(similarity[:num])

        return [EGFR, ERBB2, ABL, SRC, LCK, PDGFRbeta, VEGFR2, FGFR1, EPHB4, Stab, Perm, Sol, hERG,
                EGFR_sim, ERBB2_sim, ABL_sim, SRC_sim, LCK_sim, PDGFRbeta_sim, VEGFR2_sim, FGFR1_sim,
                EPHB4_sim, Stab_sim, Perm_sim, Sol_sim, hERG_sim]


    def calc_reward_from_objective_values(values, conf):
        if None in values:
            return -1

        egfr, erbb2, abl, src, lck, pdgfrbeta, vegfr2, fgfr1, ephb4, stab, perm, sol, hERG, \
        egfr_sim, erbb2_sim, abl_sim, src_sim, lck_sim, pdgfrbeta_sim, vegfr2_sim, fgfr1_sim, \
        ephb4_sim, stab_sim, perm_sim, sol_sim, herg_sim = values
        params = conf['reward_function']

        # AD filter
        is_in_AD = [] # 0: out of AD, 1: in AD
        is_in_AD.append(scale_objective_value(params['AD']['EGFR'], egfr_sim))
        is_in_AD.append(scale_objective_value(params['AD']['ERBB2'], erbb2_sim))
        is_in_AD.append(scale_objective_value(params['AD']['ABL'], abl_sim))
        is_in_AD.append(scale_objective_value(params['AD']['SRC'], src_sim))
        is_in_AD.append(scale_objective_value(params['AD']['LCK'], lck_sim))
        is_in_AD.append(scale_objective_value(params['AD']['PDGFRbeta'], pdgfrbeta_sim))
        is_in_AD.append(scale_objective_value(params['AD']['VEGFR2'], vegfr2_sim))
        is_in_AD.append(scale_objective_value(params['AD']['FGFR1'], fgfr1_sim))
        is_in_AD.append(scale_objective_value(params['AD']['EPHB4'], ephb4_sim))
        is_in_AD.append(scale_objective_value(params['AD']['Metabolic_stability'], stab_sim))
        is_in_AD.append(scale_objective_value(params['AD']['Permeability'], perm_sim))
        is_in_AD.append(scale_objective_value(params['AD']['Solubility'], sol_sim))
        is_in_AD.append(scale_objective_value(params['AD']['hERG'], herg_sim))
        weights_AD = []
        weights_AD.append(params['AD']['EGFR']['weight'])
        weights_AD.append(params['AD']['ERBB2']['weight'])
        weights_AD.append(params['AD']['ABL']['weight'])
        weights_AD.append(params['AD']['SRC']['weight'])
        weights_AD.append(params['AD']['LCK']['weight'])
        weights_AD.append(params['AD']['PDGFRbeta']['weight'])
        weights_AD.append(params['AD']['VEGFR2']['weight'])
        weights_AD.append(params['AD']['FGFR1']['weight'])
        weights_AD.append(params['AD']['EPHB4']['weight'])
        weights_AD.append(params['AD']['Metabolic_stability']['weight'])
        weights_AD.append(params['AD']['Permeability']['weight'])
        weights_AD.append(params['AD']['Solubility']['weight'])
        weights_AD.append(params['AD']['hERG']['weight'])
        for i, w in zip(is_in_AD, weights_AD):
            if w == 0:
                continue
            if i == 0:
                return 0

        # Property
        scaled_values = []
        scaled_values.append(scale_objective_value(params['property']['EGFR'], egfr))
        scaled_values.append(scale_objective_value(params['property']['ERBB2'], erbb2))
        scaled_values.append(scale_objective_value(params['property']['ABL'], abl))
        scaled_values.append(scale_objective_value(params['property']['SRC'], src))
        scaled_values.append(scale_objective_value(params['property']['LCK'], lck))
        scaled_values.append(scale_objective_value(params['property']['PDGFRbeta'], pdgfrbeta))
        scaled_values.append(scale_objective_value(params['property']['VEGFR2'], vegfr2))
        scaled_values.append(scale_objective_value(params['property']['FGFR1'], fgfr1))
        scaled_values.append(scale_objective_value(params['property']['EPHB4'], ephb4))
        scaled_values.append(scale_objective_value(params['property']['Metabolic_stability'], stab))
        scaled_values.append(scale_objective_value(params['property']['Permeability'], perm))
        scaled_values.append(scale_objective_value(params['property']['Solubility'], sol))
        scaled_values.append(scale_objective_value(params['property']['hERG'], hERG))
        weights = []
        weights.append(params['property']['EGFR']['weight'])
        weights.append(params['property']['ERBB2']['weight'])
        weights.append(params['property']['ABL']['weight'])
        weights.append(params['property']['SRC']['weight'])
        weights.append(params['property']['LCK']['weight'])
        weights.append(params['property']['PDGFRbeta']['weight'])
        weights.append(params['property']['VEGFR2']['weight'])
        weights.append(params['property']['FGFR1']['weight'])
        weights.append(params['property']['EPHB4']['weight'])
        weights.append(params['property']['Metabolic_stability']['weight'])
        weights.append(params['property']['Permeability']['weight'])
        weights.append(params['property']['Solubility']['weight'])
        weights.append(params['property']['hERG']['weight'])

        multiplication_value = 1
        for v, w in zip(scaled_values, weights):
            multiplication_value *= v**w
        dscore = multiplication_value ** (1/sum(weights))

        return dscore
