import re
import pandas as pd
import os

energy_pattern = re.compile(r'SCF Done:.*?(-?\d+\.\d+)')
freeenergy_pattern = re.compile(r'Sum of electronic and thermal Free Energies=\s*([-\d\.]+)')
enthalpy_pattern = re.compile(r'Sum of electronic and thermal Enthalpies=\s*([-\d\.]+)')
freq_pattern = re.compile(r'Frequencies\s+--\s+(\S+)')
irinten_pattern = re.compile(r'IR Inten\s+--\s+(\S+)')
dipole_pattern = re.compile(r'Tot=\s+([-\d\.]+)')
homo_pattern = re.compile(r'Alpha  occ\. eigenvalues\s+--\s+(.+)')
lumo_pattern = re.compile(r'Alpha virt\. eigenvalues\s+--\s+(.+)')
qhirsh_pattern = re.compile(r'\s{7}Tot\s+([-+]?\d*\.?\d+)')

def extract_last_values(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    with open(file_path, 'r') as file:
        content = file.read()
        energy_matches = energy_pattern.findall(content)
        free_energy_match = freeenergy_pattern.search(content)
        enthalpy_match = enthalpy_pattern.search(content)
        freq_match = freq_pattern.search(content)
        irinten_match = irinten_pattern.search(content)
        dipole_match = dipole_pattern.search(content)
        homo_matches = homo_pattern.findall(content)
        lumo_match = lumo_pattern.search(content)
        qhirsh_matches = qhirsh_pattern.findall(content)
        
        scf_energy = float(energy_matches[-1]) if energy_matches else None
        free_energy = float(free_energy_match.group(1)) if free_energy_match else None
        enthalpy = float(enthalpy_match.group(1)) if enthalpy_match else None
        lowfreq = float(freq_match.group(1)) if freq_match else None
        lowirinten = float(irinten_match.group(1)) if irinten_match else None
        dipole = float(dipole_match.group(1)) if dipole_match else None
        homo = float(homo_matches[-1].split()[-1]) if homo_matches else None
        lumo = float(lumo_match.group(1).split()[0]) if lumo_match else None
        qhirsh = float(qhirsh_matches[-1].split()[0]) if qhirsh_matches else None
        
        return scf_energy, free_energy, enthalpy, lowfreq, lowirinten, dipole, homo, lumo, qhirsh

def energy_diff(energy1, energy2):
    return round((energy1 - energy2) * 627.50947, 2) # Convert Hartrees to kcal/mol

data = {
    'Reactant': [],
    'Datapoint': [],
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
    'dGPR': [],
    'HR': [],
    'HTS': [],
    'HP': [],
    'freqR': [],
    'freqTS': [],
    'freqP': [],
    'irR': [],
    'irTS': [],
    'irP': [],
    'dipoleR': [],
    'dipoleTS': [],
    'dipoleP': [],
    'homoR': [],
    'homoTS': [],
    'homoP': [],
    'lumoR': [],
    'lumoTS': [],
    'lumoP': [],
    'qhirshR': [],
    'qhirshTS': [],
    'qhirshP': []
}

with open('wrtprecomplex.txt', 'r') as file:
    prefixes = file.read().splitlines()

for prefix in prefixes:
    directory = f'/home/vikikrpd/VIKI_BACKUP/4TBHDD/CovInDB/CovInDB_calcs/dataset/presentation/wrtprecomplex_PM7'
    reactant_file = f'{directory}/{prefix}_Reactant.log'
    ts_file = f'{directory}/{prefix}.log'
    product_file = f'{directory}/{prefix}_Product.log'
    
    R_scf_energy, R_free_energy, R_enthalpy, R_freq, R_irinten, R_dipole, R_homo, R_lumo, R_qhirsh = extract_last_values(reactant_file)
    TS_scf_energy, TS_free_energy, TS_enthalpy, TS_freq, TS_irinten, TS_dipole, TS_homo, TS_lumo, TS_qhirsh = extract_last_values(ts_file)
    P_scf_energy, P_free_energy, P_enthalpy, P_freq, P_irinten, P_dipole, P_homo, P_lumo, P_qhirsh = extract_last_values(product_file)
    
    if None in [R_scf_energy, TS_scf_energy, P_scf_energy, R_free_energy, TS_free_energy, P_free_energy, R_enthalpy, TS_enthalpy, P_enthalpy, R_freq, TS_freq, P_freq, R_irinten, TS_irinten, P_irinten, R_dipole, TS_dipole, P_dipole, R_homo, TS_homo, P_homo, R_lumo, TS_lumo, P_lumo, R_qhirsh, TS_qhirsh, P_qhirsh]:
        continue
    
    TS_R_scf_energy = energy_diff(TS_scf_energy, R_scf_energy)
    TS_P_scf_energy = energy_diff(TS_scf_energy, P_scf_energy)
    P_R_scf_energy = energy_diff(P_scf_energy, R_scf_energy)
    TS_R_free_energy = energy_diff(TS_free_energy, R_free_energy)
    TS_P_free_energy = energy_diff(TS_free_energy, P_free_energy)
    P_R_free_energy = energy_diff(P_free_energy, R_free_energy)
    
    data['Reactant'].append(prefix+'_Reactant')
    data['Datapoint'].append(prefix)
    data['Product'].append(prefix+'_Product')
    data['eR'].append(R_scf_energy)
    data['eTS'].append(TS_scf_energy)
    data['eP'].append(P_scf_energy)
    data['GR'].append(R_free_energy)
    data['GTS'].append(TS_free_energy)
    data['GP'].append(P_free_energy)
    data['deTSR'].append(TS_R_scf_energy)
    data['deTSP'].append(TS_P_scf_energy)
    data['dePR'].append(P_R_scf_energy)
    data['dGTSR'].append(TS_R_free_energy)
    data['dGTSP'].append(TS_P_free_energy)
    data['dGPR'].append(P_R_free_energy)
    data['HR'].append(R_enthalpy)
    data['HTS'].append(TS_enthalpy)
    data['HP'].append(P_enthalpy)
    data['freqR'].append(R_freq)
    data['freqTS'].append(TS_freq)
    data['freqP'].append(P_freq)
    data['irR'].append(R_irinten)
    data['irTS'].append(TS_irinten)
    data['irP'].append(P_irinten)
    data['dipoleR'].append(R_dipole)
    data['dipoleTS'].append(TS_dipole)
    data['dipoleP'].append(P_dipole)
    data['homoR'].append(R_homo)
    data['homoTS'].append(TS_homo)
    data['homoP'].append(P_homo)
    data['lumoR'].append(R_lumo)
    data['lumoTS'].append(TS_lumo)
    data['lumoP'].append(P_lumo)
    data['qhirshR'].append(R_qhirsh)
    data['qhirshTS'].append(TS_qhirsh)
    data['qhirshP'].append(P_qhirsh)

    print(f"Processed all files for {prefix}")

df = pd.DataFrame(data)
df.to_csv('wrtprecomplex_pm7.csv', index=False)
