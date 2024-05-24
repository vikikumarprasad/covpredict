import os
import logging
import pandas as pd
import numpy as np
import datamol as dm
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import AllChem, rdMolDescriptors, EState
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from morfeus import read_xyz, SASA, BuriedVolume, Sterimol, Dispersion
#from morfeus import BiteAngle, ConeAngle, SolidAngle, LocalForce, XTB
from dscribe.descriptors import CoulombMatrix, SineMatrix, EwaldSumMatrix, ACSF, SOAP, MBTR, LMBTR, ValleOganov
from ase import Atoms

# Setup logging
logging.basicConfig(filename='featurize.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def compute_molfeat_descriptors(mol, filename):
    ''' Available kind options are desc3D,desc2D,mordred,cats2D,cats3D,
        pharm2D,pharm3D,scaffoldkeys,skeys,electroshape,usr,usrcat  '''
    if mol is None:
        return pd.DataFrame()
    try:
        #transform_mordred = FPVecTransformer(kind="mordred", ignore_3D=True, replace_nan=True, n_jobs=-1) # ignore 3D descriptors
        transform_mordred = FPVecTransformer(kind="mordred", ignore_3D=False, replace_nan=True, n_jobs=1) # don't ignore 3D descriptors
        features_mordred, index_mordred = transform_mordred(mol, ignore_errors=True)
        if features_mordred is not None:
            df_descriptors = pd.DataFrame(features_mordred, columns=transform_mordred.columns)
        else:
            df_descriptors = pd.DataFrame()
        return df_descriptors
    except Exception as e:
        logging.error(f"Descriptor calculation failed for {filename}: {str(e)}")
        return pd.DataFrame()

def compute_morfeus_descriptors(xyzname, filename):
    try:
        descriptors = {}
        elements, coordinates = read_xyz(xyzname)
        sasa = SASA(elements, coordinates)
        buriedvolume = BuriedVolume(elements, coordinates, 1)
        sterimol = Sterimol(elements, coordinates, 1, 2)
        dispersion = Dispersion(elements, coordinates)
        descriptors['SASA'] = [sasa.area]
        descriptors['SASV'] = [sasa.volume]
        descriptors['FracBuriedVol'] = [buriedvolume.fraction_buried_volume]
        descriptors['SterimolB1'] = [sterimol.B_1_value]
        descriptors['SterimolB5'] = [sterimol.B_5_value]
        descriptors['SterimolL'] = [sterimol.L_value]
        descriptors['Pint'] = [dispersion.p_int]
        df_descriptors = pd.DataFrame(descriptors)
        return df_descriptors
    except Exception as e:
        logging.error(f"Morfeus descriptor calculation failed for {filename}: {str(e)}")
        return pd.DataFrame()

def compute_rdkit_descriptors(mol, filename):
    if mol is None:
        return pd.DataFrame()
    try:
        descriptors = {
            "PEOE": [],
            "LogP": 0.0,
            "MR": 0.0,
            "EState": [],
            "TPSA": 0.0,
            "AmideBonds": 0
        }
        crippen_contribs = rdMolDescriptors.CalcCrippenDescriptors(mol)
        estate = EState.EStateIndices(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        amide_bonds = rdMolDescriptors.CalcNumAmideBonds(mol)
        descriptors["PEOE"] = [x[0] for x in peoe]
        descriptors["LogP"] = [crippen_contribs[0]]
        descriptors["MR"] = [crippen_contribs[1]]
        descriptors["EState"] = [estate]
        descriptors["TPSA"] = [tpsa]
        descriptors["AmideBonds"] = [amide_bonds]
        df_descriptors = pd.DataFrame(descriptors)
        return df_descriptors
    except Exception as e:
        logging.error(f"RDKit descriptor calculation failed for {filename}: {str(e)}")
        return pd.DataFrame()

def compute_electronic_descriptors(elements, coordinates, filename):
    return df_descriptors

def compute_molfeat_fingerprints(mol, filename):
    ''' Available options are maccs,avalon,ecfp,fcfp,topological,
        atompair,rdkit,pattern,layered,map4,secfp,erg,estate,avalon-count,
        rdkit-count,ecfp-count,fcfp-count,topological-count,atompair-count '''
    if mol is None:
        return pd.DataFrame()
    try:
        #transform_fp = FPVecTransformer(kind="rdkit", n_jobs=1)
        #transform_fp = FPVecTransformer(kind="ecfp:6", n_jobs=-1) #Special case of ecfp:6
        #transform_fp = PretrainedHFTransformer(kind='ChemBERTa-77M-MLM', notation='smiles') #Special case of ChemBERTa
        transform_fp = FPVecTransformer(kind="ecfp:6", n_jobs=1)
        columns_fp = transform_fp.columns
        features_fp, index_fp = transform_fp(mol, ignore_errors=True)
        if features_fp is not None:
            df_descriptors = pd.DataFrame(features_fp, columns=columns_fp)
        else:
            df_descriptors = pd.DataFrame()
        return df_descriptors
    except Exception as e:
        logging.error(f"Fingeprint calculation failed for {filename}: {str(e)}")
        return pd.DataFrame()

def compute_dscribe_fingerprints(mol, filename):
    features_fp = []
    transform_fp= CoulombMatrix(n_atoms_max=122) #For non-periodic systems only
    #transform_fp = SineMatrix(n_atoms_max=100,permutation="sorted_l2",sparse=False) #For periodic systems only
    #transform_fp = EwaldSumMatrix(n_atoms_max=100) #For periodic systems only
    #transform_fp = ValleOganov(species=['H','C','N','O','P','S','F','Cl','Br','I'],function='distance',sigma=10**(-0.5),n=100,r_cut=5)
    #transform_fp = SOAP(species=['H','C','N','O','P','S','F','Cl','Br','I'],r_cut=6.0,n_max=8,l_max=6,periodic=False)
    #transform_fp = ACSF(species=['H','C','N','O','P','S','F','Cl','Br','I'],r_cut=6.0,g2_params=[[1, 1],[1, 2],[1, 3]],g4_params=[[1, 1, 1],[1, 2, 1],[1, 1, -1],[1, 2, -1]])
    #transform_fp = MBTR(species=['H','C','N','O','P','S','F','Cl','Br','I'],k1={"geometry":{"function":"atomic_number"},"grid":{"min":1,"max":8,"sigma":0.1,"n": 100}},k2={"geometry":{"function":"inverse_distance"},"grid":{"min":0,"max":1,"sigma":0.1,"n":100}},k3={"geometry":{"function":"cosine"},"grid":{"min":-1,"max":1,"sigma":0.1,"n":100}},periodic=False)
    #transform_fp = LMBTR(species=['H','C','N','O','P','S','F','Cl','Br','I'],k2={"geometry":{"function":"inverse_distance"},"grid":{"min":0,"max":1,"sigma":0.1,"n":100}},k3={"geometry":{"function":"cosine"},"grid":{"min":-1,"max":1,"sigma":0.1,"n":100}},periodic=False)

    atoms = mol.GetAtoms()
    positions = mol.GetConformers()[0].GetPositions()
    symbols = [atom.GetSymbol() for atom in atoms]
    ase_structure = Atoms(symbols=symbols, positions=positions)
    descriptors = transform_fp.create(ase_structure, n_jobs=1) #CoulombMatrix,SineMatrix,EwaldSumMatrix,ValleOganov,ACSF,SOAP,MBTR,LMBTR
    #print(f"Shape of descriptors: {descriptors.shape}")
    features_fp.append(descriptors)

    ncolumns = transform_fp.get_number_of_features()
    columns_fp = [f"fp_{i}" for i in range(ncolumns)]
    #print(f"Length of columns_fp: {len(columns_fp)}")
    if features_fp is not None:
        df_descriptors = pd.DataFrame(features_fp, columns=columns_fp)
    else:
        df_descriptors = pd.DataFrame()
    return df_descriptors

def smiles_to_3d_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def sanitize_molecule(mol,filename):
    try:
        Chem.SanitizeMol(mol)
    except rdchem.AtomValenceException as e:
        logging.error(f" For {filename} valence error ignored: {str(e)}")
    except rdchem.KekulizeException as e:
        logging.error(f" For {filename} kekulization error ignored: {str(e)}")
    except rdchem.RDKitException as e:
        logging.error(f"For {filename} RDKit-specific error ignored: {str(e)}")
    except Exception as e:
        logging.error(f"For {filename} sanitization failed: {str(e)}")
    return mol

#####################################################################
#smiles_list = ["CCO", "CCC", "CCN"]
#Plist = [smiles_to_3d_mol(smiles) for smiles in smiles_list]

workdir = '/home/vikikrpd/Softwares/covpredict/data'
row_list = 'wrtprecomplex.txt'

row_list_path = os.path.join(workdir, row_list)
with open(row_list_path, 'r') as file:
    molfile_list = [line.strip() + "_Product.mol" for line in file if line.strip()]

with open(row_list_path, 'r') as file:
    xyzfile_list = [line.strip() + "_Product.xyz" for line in file if line.strip()]

Plist_mol = [os.path.join(workdir, filename) for filename in molfile_list]
Plist_xyz = [os.path.join(workdir, filename) for filename in xyzfile_list]

#mol = []
results = []
for filename, entry, xyzname in zip(molfile_list, Plist_mol, Plist_xyz):
    #obj = Chem.MolFromSmiles(entry)
    obj = Chem.MolFromMolFile(entry, removeHs=False, sanitize=False, strictParsing=True)
    #mol.append(obj)
    
    if obj is None:
        print(f"obj for {filename} is None")
        continue
    sanitized_obj = sanitize_molecule(obj,filename)
    if sanitized_obj is None:
        print(f"sanitized_obj for {filename} is None")
        continue
    
    #inpfeats = compute_molfeat_descriptors(sanitized_obj, filename)
    #inpfeats = compute_morfeus_descriptors(xyzname, filename)
    inpfeats = compute_rdkit_descriptors(sanitized_obj, filename)
    #inpfeats = compute_molfeat_fingerprints(sanitized_obj, filename)
    #inpfeats = compute_dscribe_fingerprints(sanitized_obj, filename)

    if inpfeats.empty:
        print(f"No descriptors generated for {filename}")
        continue

    filename_clean = filename.replace(".mol", "")
    inpfeats.insert(0, 'Product', filename_clean)
    results.append(inpfeats)

df = pd.concat(results, ignore_index=True)
print(df.head)
df.to_csv('inpfeats_rdkitdesc.csv', index=False)
