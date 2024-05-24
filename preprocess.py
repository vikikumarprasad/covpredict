import os
import warnings
import joblib
import pandas as pd
import numpy as np
import datamol as dm
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from sklearn.model_selection import train_test_split, ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
#from splito import ScaffoldSplit, PerimeterSplit, MaxDissimilaritySplit, MolecularWeightSplit, StratifiedDistributionSplit
#from splito.simpd import SIMPDSplitter

# Making the output less verbose
#warnings.simplefilter("ignore")
#os.environ["PYTHONWARNINGS"] = "ignore"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#dm.disable_rdkit_log()

def remove_zero_variance_features(data):
    selector = VarianceThreshold()
    selector.fit(data)
    retained_columns =  data.columns[selector.get_support()]
    dropped_columns =  data.columns[~selector.get_support()]
    print(f"{len(dropped_columns)} columns dropped because of zero variance features: {dropped_columns.tolist()}")
    return data[retained_columns]

def remove_collinear_features(data, threshold):
    corr_matrix = data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    print(f"{len(to_drop)} columns dropped because of correlation greater than {threshold}: {to_drop}")
    return data.drop(columns=to_drop, axis=1, inplace=False)

def scale_features_standard(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def scale_features_minmax(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def scale_targets_standard(data):
    scaler = StandardScaler()
    data_array = data.values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data_array)
    return pd.DataFrame(scaled_data, columns=[data.columns]), scaler

def scale_targets_minmax(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_array = data.values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data_array)
    return pd.DataFrame(scaled_data, columns=[data.columns]), scaler

#def scaffold_split(smiles):
    #scaffolds = [dm.to_smiles(dm.to_scaffold_murcko(dm.to_mol(smi))) for smi in smiles]
    #splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    #splitter = ScaffoldSplit(smiles=data["smiles"].tolist(), n_jobs=-1, test_size=0.2, random_state=111) #Initialize a splitter
    #train_idx, test_idx = next(splitter.split(X=data.smiles.values)) #Generate indices for training and test set
    #return next(splitter.split(smiles, groups=scaffolds))

df_X = pd.read_csv('inpfeats_morfeus.csv', index_col=False)

df_y = pd.read_csv('refDZ+PM7.csv', usecols=['Product', 'dGTSR_refDZ', 'dGTSR_PM7'], index_col=False)
df_merged = pd.merge(df_X, df_y, on='Product', how='inner')
X = df_merged.drop(['Product', 'dGTSR_refDZ', 'dGTSR_PM7'], axis=1)

y_ref = pd.DataFrame()
y_sqm = pd.DataFrame()
y_diff = pd.DataFrame()
y_ref['dGTSR_refDZ'] = df_merged['dGTSR_refDZ']
y_sqm['dGTSR_PM7'] = df_merged['dGTSR_PM7']
y_diff['dTSR_PM7-dGTSR_refDZ'] = df_merged['dGTSR_PM7'] - df_merged['dGTSR_refDZ']
y = y_ref

# Removing zero variance threshold features
X_numeric = X.select_dtypes(include=[np.number])  # Ensure only numeric data is processed
X = remove_zero_variance_features(X_numeric)

# Get descriptive data statistics 
print(X.describe())

################################################################################################################################
# Draw the heatmap with the mask and correct aspect ratio
#corr = X.corr() # Calculate pairwise correlation among descriptors
#f = plt.figure(figsize=(80, 76))
#plt.matshow(corr, fignum=f.number , cmap = plt.cm.seismic)
#plt.xticks(range(X.select_dtypes(['number']).shape[1]), X.select_dtypes(['number']).columns, fontsize=14, rotation=90)
#plt.yticks(range(X.select_dtypes(['number']).shape[1]), X.select_dtypes(['number']).columns, fontsize=14)
#plt.tick_params(axis='x', pad=15)
#cb = plt.colorbar()
#cb.ax.tick_params(labelsize=14)
#plt.title('Correlation Matrix', fontsize=28)
#plt.savefig("correlation_matrix.pdf")

# Plot probability density of target variable for the full dataset
#sns.kdeplot(shade = True , data = y)
#plt.title("KDE Density for target variable")
#plt.xlabel("$Y$ $(kcal/mol)$")
#plt.savefig("kdetarget.pdf")

# Plot pairwise relationships among target variable and descriptors
#sns.pairplot(df , y_vars = X.columns.values , x_vars = y.columns.values , plot_kws={"color":"xkcd:blue with a hint of purple"})
#plt.savefig("desc_vs_target.pdf")

# Plot distribution of the descriptors in for KDE
#k = len(X.columns)-1
#n = 2
#m = (k - 1) // n + 1
#fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
#for i, (name, col) in enumerate(X.iteritems()):
#  if i!=(k):
#    r, c = i // n, i % n
#    ax = axes[r, c]
#    col.hist(ax=ax , color = "darkviolet" , alpha = 0.5,bins=15)
#    ax2 = ax.twinx()
#    sns.kdeplot(ax=ax2, data = col  , color = "deepskyblue"  , shade = True, alpha = 0.3)
#    ax2.title.set_text(name)
#    ax2.set_ylim(0)
#fig.tight_layout()
#plt.savefig("descriptors_distribution.pdf")
##################################################################################################################################

X = remove_collinear_features(X, 0.99) # Removing collinear features
X = scale_features_standard(X)
#y, y_scaler = scale_targets_standard(y)
#X = scale_features_minmax(X)
#y, y_scaler = scale_targets_minmax(y)
#joblib.dump(y_scaler, 'y_scaler-mordred2.pkl') #Save the y_scaler for inverse transform predictions later

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# TO-DO: Apply various splits from splito
#train_idx, test_idx = scaffold_split(y)
#X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Save the processed data
X_train.to_csv('X_train-morfeus.csv', index=False)
X_test.to_csv('X_test-morfeus.csv', index=False)
y_train.to_csv('y_train-morfeus.csv', index=False)
y_test.to_csv('y_test-morfeus.csv', index=False)

# Output the first few rows of the processed training data for verification
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
