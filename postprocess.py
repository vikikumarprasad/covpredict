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

# Plots for prediction results
#def plot_df( dft , mopac , preds):
#     sns.set_style("white")
#     mae = mean_absolute_error(dft, preds)
#     mae_or = mean_absolute_error(dft , mopac)
#     fig , (ax1 , ax2)  = plt.subplots(2,1 , figsize= (10,12))
#     ax1.set_xlabel("PM7 Barrier Height $(kcal/mol)$" , fontsize = 14)
#     ax1.set_ylabel("DFT Barrier Height $(kcal/mol)$" , fontsize = 14)
#     ax2.set_xlabel("PM7+DL Barrier Height $(kcal/mol)$" , fontsize = 14)
#     ax2.set_ylabel("DFT Barrier Height $(kcal/mol)$" , fontsize = 14)
#     ax1.plot(mopac , dft , 'o' , color = "indianred" , markersize = 1 )
#     ax2.plot(preds , dft , 'o' , color = "indianred" , markersize = 1 )
#     ax1.text(125.5,20, "MAE= %.2f $(kcal/mol)$" %mae_or ,style='italic',bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
#     ax2.text(100.5,32.5, "MAE= %.2f $(kcal/mol)$" %mae ,style='italic',bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
#     ax1.set_ylim(np.min(mopac),np.max(mopac))
#     ax2.set_ylim(np.min(preds),np.max(preds))
#     lims1 = [np.min([ax1.get_xlim(), ax1.get_ylim()]),  np.max([ax1.get_xlim(), ax1.get_ylim()])]
#     my_suptitle = fig.suptitle("DFT vs PM7 comparison + DL correction" , fontsize = 22 , y = 1.05)
#     ax1.plot(lims1, lims1, 'k-', alpha=0.75, zorder=0 , linewidth = 2 , label = "$y=x$")
#     ax1.set_aspect('equal')
#     ax1.set_xlim(lims1)
#     ax1.set_ylim(lims1)
#     lims2 = [np.min([ax2.get_xlim(), ax2.get_ylim()]),  np.max([ax2.get_xlim(), ax2.get_ylim()])]
#     ax2.plot(lims2, lims2, 'k-', alpha=0.75, zorder=0 , linewidth = 2 , label = "$y=x$")
#     ax2.set_aspect('equal')
#     ax2.set_xlim(lims2)
#     ax2.set_ylim(lims2)
#     ax1.legend(fontsize=14)
#     ax2.legend(fontsize=14)
#     plt.tight_layout()
#     plt.savefig("xgb_preds_vert.pdf" ,  dpi=fig.dpi, bbox_inches='tight',bbox_extra_artists=[my_suptitle])
#     return
#plot_df(y_test["ae_diff"].values, X_test["AE_mopac"].values , preds_xg)

#def PlotCorr(ae_dft , ae_mopac , ae_corr):
#     sns.set_style("darkgrid")
#     fig , (ax1 , ax2)  = plt.subplots(1,2 , figsize= (8,6))
#     mae = mean_absolute_error(ae_dft, ae_corr)
#     mae_or = mean_absolute_error(ae_dft , ae_mopac)
#     fig.suptitle("DFT vs PM7 comparison + DL correction")
#     ax1.plot(ae_mopac , ae_dft , "o" , color = "deeppink" , alpha=0.7)
#     ax1.plot([np.min(ae_mopac),np.max(ae_mopac)] , [np.min(ae_dft) , np.max(ae_dft)] , "-k" , linewidth=4)
#     ax1.set_xlabel("$E_{a}^{PM7}$ $(kcal/mol)$")
#     ax1.set_ylabel("$E_{a}^{DFT}$ $(kcal/mol)$")
#     ax1.text(30,165, "MAE= %.2f $(kcal/mol)$" %mae_or ,style='italic',
#     bbox={'facecolor': 'gold', 'alpha': 0.5, 'pad': 10})
#     ax2.plot(ae_corr , ae_dft , "o" , color = "deepskyblue" , alpha=0.7)
#     ax2.plot([np.min(ae_corr),np.max(ae_corr)] , [np.min(ae_dft) , np.max(ae_dft)] , "-k" , linewidth=4)
#     ax2.set_xlabel("$E_{a}^{PM7+DL}$ $(kcal/mol)$")
#     ax2.text(40,165, "MAE= %.2f $(kcal/mol)$" %mae ,style='italic',
#     bbox={'facecolor': 'gold', 'alpha': 0.5, 'pad': 10})
#     plt.savefig("ea_preds_nn.pdf")
#     return
#PlotCorr(y_test["ae_diff"].values , X_test["AE_mopac"].values , y_preds[:,1])

#def DensPlot(ae_dft , ae_mopac , ae_corr):
#     fig = plt.figure(figsize = (8,6))
#     ax = fig.add_subplot(1, 1, 1, projection='scatter_density' )
#     ax.grid(False)
#     density = ax.scatter_density(ae_corr, ae_dft , cmap=plt.cm.viridis )
#     ax.set_xlabel("$E_{a}^{PM7+DL}$ $(kcal/mol)$")
#     ax.set_ylabel("$E_{a}^{DFT}$ $(kcal/mol)$")
#     fig.colorbar(density, label='Count')
#     plt.savefig("dens_nn.pdf")
#     return
#DensPlot(y_test["ae_diff"].values, X_test["AE_mopac"].values , y_preds[:,1])
