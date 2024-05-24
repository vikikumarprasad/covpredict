import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
import mpl_scatter_density
import shap
#import rdkit
#import GPy
#import GPflow
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, PassiveAggressiveRegressor, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_absolute_error

### Model interpretability based on SHAP values.
#preds_sh = xg_model.predict(X_train , output_margin = True)
#explainer = shap.TreeExplainer(xg_model)
#sh_values = explainer.shap_values(X_train)

#print(smi_df.index.values[np.random.randint(0, smi_df.shape[0] - 1)])
#print(smi_df.loc[11911])

#def explainReact(df_smi , sh_values , xtrain , ytrain , labels):
## place a text box in upper left in axes coords
#  idxs = xtrain.index[np.random.randint(0 , xtrain.shape[0] -1 )]
#  val_id = np.where(xtrain.index == idxs)[0][0]
#  mol_r = df_smi.loc[idxs]["Rsmiles"]
#  mol_p = df_smi.loc[idxs]["Psmiles"]
#  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#  fig, ax = plt.subplots()
#  mop =  "AE Mopac = " + np.format_float_positional(xtrain["AE_mopac"].values[val_id], precision=2) + "  $kcal/mol$ "
#  
#  dft =  "AE DFT = " + np.format_float_positional(ytrain.values[val_id] , precision = 2) + "  $kcal/mol$ " 
#  ax.text(0.05, 1.15, mop, transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#  ax.text(0.55, 1.15, dft, transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#  rxn = AllChem.ReactionFromSmarts(mol_r + ">>" + mol_p , useSmiles = True)
#  display(Chem.Draw.ReactionToImage(rxn))
#  shap.plots._waterfall.waterfall_legacy(explainer.expected_value ,  sh_values[val_id] , feature_names=labels.columns.values )
#  return

#from IPython import get_ipython
#from IPython.core.magic import register_cell_magic
#import PIL
#from base64 import b64decode
#from io import BytesIO
#@register_cell_magic
#def capture_png(line, cell):
#    get_ipython().run_cell_magic(
#        'capture',
#        ' --no-stderr --no-stdout result',
#        cell
#    )
#    out_paths = line.strip().split(' ')
#    for output in result.outputs:
#        data = output.data
#        if 'image/png' in data:
#            path = out_paths.pop(0)
#            if not path:
#                raise ValueError('Too few paths given!')
#            png_bytes = data['image/png']
#            if isinstance(png_bytes, str):
#                png_bytes = b64decode(png_bytes)
#            assert isinstance(png_bytes, bytes)
#            bytes_io = BytesIO(png_bytes)
#            image = PIL.Image.open(bytes_io)
#            image.save(path, 'png')
#informative_names = {"AE_mopac":r"$BH_{PM7}$" , "exp_mopac":r"$e^{-BH_{PM7}}$" , "Par_n_Pople":r"$\eta^{{TS}}$" , "Mul":r"$\alpha^{{TS}}$" , 
#                     "ch_f" : "+CH" , "DH_Mopac":r"$\Delta E_{{r}}$" , "lap_eig_1":r"$\lambda_{{1}}^{{TS}}$" , "Freq":r"$\omega_{{1}}^{{TS}}$" , 
#                     "ZPE_TS_P":r"$ZPE^{{TS}} - ZPE^{{P}}$" , "SMR_VSA9":r"$RDKIT_{{1}}$" , "LabuteASA":r"$RDKIT_{{2}}$" , "BalabanJ":r"$RDKIT_{{3}}$" , 
#                     "ZPE_TS_R":r"$ZPE^{{TS}} - ZPE^{{R}}$" , "ZPE_P_R":r"$ZPE^{{P}} - ZPE^{{R}}$" , "ch_b": "-CH" , "co_f":"+CO" , "MolLogP":r"$RDKIT_{{4}}$" , 
#                     "Chi0v": r"$RDKIT_{{5}}$" , "piS_P_R":r"$\pi^{{S}}_{{P}} - \pi^{{S}}_{{R}}$" , "VSA_EState2":r"$RDKIT_{{6}}$" , "hh_f"  : "+HH"}
#label_Xtrain = X_train.rename(columns = informative_names)
#get_ipython().run_cell_magic('capture_png', 'example_reac.png reac.png', 'explainReact(smi_df , sh_values , X_train , y_train_xg , label_Xtrain)\n')

#import sys
#from PIL import Image
#images = [Image.open(x) for x in ['example_reac.png' , 'reac.png']]
#widths, heights = zip(*(i.size for i in images))
#total_width = max(widths)
#max_height = sum(heights)
#new_im = Image.new('RGB', (total_width, max_height) , color=(255,255,255,0))
#x_offset = 0
#z_offset = 0
#for im in images:
#  new_im.paste(im, (z_offset , x_offset))
#  x_offset += im.size[1]
#  z_offset += int(im.size[1]/3)+10
#new_im.save('final_reac.png')
#display(new_im)

#from IPython.display import Image
#from pandas import Series
#display(Image('reac.png'))

### Explained predictions for two reactions:
##SMILES corresponding to reactant on reaction 1000
#mol_r = '[O:1]=[C:2]([N:3]([C:4]1([H:9])[C:5]([H:10])([H:11])[C:6]1([H:12])[H:13])[H:8])[H:7]'
##SMILES corresponding to product on reaction 1000
#mol_p = '[O:1](/[C:2](=[N:3]\\[H:8])[H:7])[C:6]([C:4](=[C:5]([H:10])[H:11])[H:9])([H:12])[H:13]'
#mols = [mol_r , mol_p]
#rxn = AllChem.ReactionFromSmarts(mol_r + ">>" + mol_p , useSmiles = True)
#display(Chem.Draw.ReactionToImage(rxn))
#label_Xtrain.columns
## E[f(X)] is the average of activation energies values in the training set, and f(x) is the prediction of the model. Every row contains a value for the contribution per descriptor, shifting the base value.
##SMILES corresponding to reactant on reaction 10237
#mol_r ='[C:1]([C@@:2]1([H:10])[N:3]([H:11])[C:4]([H:12])([H:13])[C:5]1=[O:6])([H:7])([H:8])[H:9]'
##SMILES corresponding to product on reaction 10237
#mol_p = '[C:1]([C+:2]1[N:3]([H:11])[C:4]([H:12])=[C:5]1[O-:6])([H:7])([H:8])[H:9].[H:10][H:13]'
#mols = [mol_r , mol_p]
#rxn = AllChem.ReactionFromSmarts(mol_r + ">>" + mol_p , useSmiles = True)
#display(Chem.Draw.ReactionToImage(rxn))
#print("DFT -> " , y_train_xg.values[6350])
#print("MOPAC -> " , X_train["AE_mopac"].values[6350])
#rxn = AllChem.ReactionFromSmarts(mol_r + ">>" + mol_p , useSmiles = True)
#display(Chem.Draw.ReactionToImage(rxn))
#shap.plots._waterfall.waterfall_legacy(explainer.expected_value ,  sh_values[6350] , feature_names=label_Xtrain.columns.values)
#shap.summary_plot(sh_values , X_train)
#duplicate_columns = label_Xtrain.columns[label_Xtrain.columns.duplicated()]
#print(duplicate_columns)
#plt.figure(figsize=(18,10))
#plt.subplot(1,2,1)
#shap.summary_plot(sh_values , label_Xtrain , plot_type = "bar" , show=False , feature_names=["" for m in ft_names] , plot_size=None)
#plt.subplot(1,2,2)
#shap.summary_plot(sh_values , label_Xtrain , show = False  , feature_names=ft_names , plot_size=None)
#plt.tight_layout()
#plt.savefig("shap_concat.pdf")
#ft_names = label_Xtrain.columns.values
#shap.summary_plot(sh_values , label_Xtrain , plot_type = "bar" , show=False)
#plt.savefig("shap_importance_abs.pdf")
