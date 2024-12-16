import os
import glob
import pickle
import subprocess

from chemtsv2.misc.scaler import max_gauss, min_gauss, rectangular
from chemtsv2.reward import Reward
import numpy as np
from rdkit import Chem
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


def run_shapescreen(mol, conf, key):

    moldir = os.path.join(conf['output_dir'], '3D_mol')
    os.makedirs(moldir, exist_ok=True)
    gen_mol_3d_path = os.path.join(moldir, f"mol{conf['gid']}.sdf")

    # Generate 3D conformer for generated molecule
    if not os.path.exists(gen_mol_3d_path):
        mol_h = Chem.AddHs(mol)
        embedding_params = AllChem.ETKDGv3()
        embedding_params.randomSeed = 0xf00d
        try:
            AllChem.EmbedMolecule(mol_h, embedding_params)
        except:
            return None
        writer = Chem.SDWriter(gen_mol_3d_path)
        writer.write(mol_h)
        writer.close()

    # Run Shapescreen
    shapescreen_output = os.path.join(moldir, f"mol{conf['gid']}_ref_{key}.sdf")
    command = [
        conf['shapescreen_bin_path'],
        '--query', gen_mol_3d_path,
        '--database', conf['shapescreen_reference_path'][key],
        '--output', shapescreen_output,
        '--score', conf['shapescreen_score'],
        '--score-sd-tags', 'True',
        '--query-format', 'sdf',
        '--database-format', 'sdf',
        '--output-format', 'sdf',
        '--num-threads', str(conf['shapescreen_num_threads']),
        '--best-hits', str(conf['shapescreen_num_output_mols']),
        ]
    if 'COLOR' or 'COMBO' in conf['shapescreen_score']:
        command.extend([
            '--all-carbon', 'False'
            ])
    if conf['shapescreen_export_report']:
        command.extend([
            '--report', os.path.join(moldir, f"mol{conf['gid']}_ref_{key}_report.csv")
            ])
    try:
        print(f"Running Shapescreen with command: {' '.join(command)}")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Shapescreen completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error occurred while running Shapescreen.")
        print("Command output:", e.stdout)
        print("Command error:", e.stderr)
        return None
    except FileNotFoundError:
        raise FileNotFoundError("shapescreen executable not found. Please ensure it is installed and available.")

    # Extract the highest similarity score
    shapescreen_out_fname = glob.glob(os.path.join(moldir, f"mol{conf['gid']}_ref_{key}*.sdf"))[0]
    maxsim_mol = [m for m in Chem.SDMolSupplier(shapescreen_out_fname)][1]   # 1st mol is the most similar one. 0th mol is the query itself.
    prop_name = ' '.join([word.capitalize() for word in conf['shapescreen_score'].split('_')])
    max_sim = float(maxsim_mol.GetProp(prop_name))

    return max_sim


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
            return lgb_models['Stab'].predict(fp)[0]

        def Perm(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['Perm'].predict(fp)[0]

        def EGFR_sim(mol):
            if mol is None:
                return None
            max_sim = run_shapescreen(mol, conf, 'EGFR')
            return max_sim

        def Stab_sim(mol):
            if mol is None:
                return None
            max_sim = run_shapescreen(mol, conf, 'Stab')
            return max_sim

        def Perm_sim(mol):
            if mol is None:
                return None
            max_sim = run_shapescreen(mol, conf, 'Perm')
            return max_sim

        return [EGFR, Stab, Perm, EGFR_sim, Stab_sim, Perm_sim]


    def calc_reward_from_objective_values(values, conf):
        if None in values:
            return -1

        egfr, stab, perm, egfr_sim, stab_sim, perm_sim = values
        
        if 'COMBO' in conf['shapescreen_score']:
            egfr_sim = egfr_sim / 2
            stab_sim = stab_sim / 2
            perm_sim = perm_sim / 2
        
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
