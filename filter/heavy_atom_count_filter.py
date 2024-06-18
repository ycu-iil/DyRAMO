from filter.filter import Filter

class HeavyAtomCountFilter(Filter):
    def check(mol, conf):
        hac = mol.GetNumHeavyAtoms()
        if 'threshold' in conf['heavy_atom_count_filter'].keys():
            return conf['heavy_atom_count_filter']['threshold'] >= hac
        elif 'min_threshold' in conf['heavy_atom_count_filter'].keys() and 'max_threshold' in conf['heavy_atom_count_filter'].keys():
            return  hac >= conf['heavy_atom_count_filter']['min_threshold'] and hac <= conf['heavy_atom_count_filter']['max_threshold']