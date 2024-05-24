import re
import pandas as pd
import os

energy_pattern = re.compile(r'SCF Done:.*?(-?\d+\.\d+)')
freeenergy_pattern = re.compile(r'Sum of electronic and thermal Free Energies=\s*([-\d\.]+)')

def extract_last_values(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    with open(file_path, 'r') as file:
        content = file.read()
        energy_matches = energy_pattern.findall(content)
        free_energy_match = freeenergy_pattern.search(content)
        
        scf_energy = float(energy_matches[-1]) if energy_matches else None
        free_energy = float(free_energy_match.group(1)) if free_energy_match else None
        
        return scf_energy, free_energy

def energy_diff(energy1, energy2):
    return round((energy1 - energy2) * 627.50947, 2) # Convert Hartrees to kcal/mol

data = {
    'Reactant': [],
    'TS': [],
    'Product': [],
    'eR': [],
    'eTS': [],
    'eP': [],
    'GR': [],
    'GTS': [],
    'GP': [],
    'deTSR': [],
    'deTSP': [],
    'dePR': [],
    'dGTSR': [],
    'dGTSP' : [],
    'dGPR': []
}

with open('wrtprecomplex.txt', 'r') as file:
    prefixes = file.read().splitlines()

for prefix in prefixes:
    directory = f'/home/vikikrpd/VIKI_BACKUP/4TBHDD/CovInDB/CovInDB_calcs/dataset/presentation/wrtprecomplex'
    reactant_file = f'{directory}/{prefix}_Reactant.log'
    ts_file = f'{directory}/{prefix}.log'
    product_file = f'{directory}/{prefix}_Product.log'
    
    reactant_scf_energy, reactant_free_energy = extract_last_values(reactant_file)
    ts_scf_energy, ts_free_energy = extract_last_values(ts_file)
    product_scf_energy, product_free_energy = extract_last_values(product_file)
    
    if None in [reactant_scf_energy, ts_scf_energy, product_scf_energy, reactant_free_energy, ts_free_energy, product_free_energy]:
        continue
    
    ts_r_scf_energy = energy_diff(ts_scf_energy, reactant_scf_energy)
    ts_p_scf_energy = energy_diff(ts_scf_energy, product_scf_energy)
    p_r_scf_energy = energy_diff(product_scf_energy, reactant_scf_energy)
    ts_r_free_energy = energy_diff(ts_free_energy, reactant_free_energy)
    ts_p_free_energy = energy_diff(ts_free_energy, product_free_energy)
    p_r_free_energy = energy_diff(product_free_energy, reactant_free_energy)
    
    data['Reactant'].append(prefix+'_Reactant')
    data['TS'].append(prefix)
    data['Product'].append(prefix+'_Product')
    data['eR'].append(reactant_scf_energy)
    data['eTS'].append(ts_scf_energy)
    data['eP'].append(product_scf_energy)
    data['GR'].append(reactant_free_energy)
    data['GTS'].append(ts_free_energy)
    data['GP'].append(product_free_energy)
    data['deTSR'].append(ts_r_scf_energy)
    data['deTSP'].append(ts_p_scf_energy)
    data['dePR'].append(p_r_scf_energy)
    data['dGTSR'].append(ts_r_free_energy)
    data['dGTSP'].append(ts_p_free_energy)
    data['dGPR'].append(p_r_free_energy)

    print(f"Processed all files for {prefix}")

df = pd.DataFrame(data)
df.to_csv('wrtprecomplex.csv', index=False)
