from numpy.core.shape_base import block
from numpy.lib.function_base import percentile, quantile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer,\
QuantileTransformer,PolynomialFeatures, power_transform, KBinsDiscretizer,OneHotEncoder
from sklearn.linear_model import RidgeCV, LassoCV,LogisticRegressionCV,Lasso
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import cross_validate,cross_val_predict,KFold,GridSearchCV
from sklearn.feature_selection import f_regression, SelectFromModel, SelectPercentile,mutual_info_regression,SelectKBest
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import NearestNeighbors
from itertools import combinations, permutations
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

def affiche_score(cr_va):
    '''affiche le train score et le test score (moyenne et std) d'un cross_validate
    Arguments : un cross_validate
    
    
    '''

    print('Train score : mean {} std {} \nTest score : mean {} std {} '.format(
        round(cr_va['train_score'].mean(),3),
        round(cr_va['train_score'].std(),3),
        round(cr_va['test_score'].mean(),3),
        round(cr_va['test_score'].std(),3)
    ))

fich=pd.read_csv('train.csv').set_index('id')
test=pd.read_csv('test.csv').set_index('id')

target=fich['loss']
df_train=fich.iloc[:,:-1]
df_viz=fich.copy()
col_init=df_train.columns
#####

rfr=RandomForestRegressor(n_estimators=50,random_state=0,verbose=3)
rfr.fit(df_train[col_init],target)
df_feat_imp=pd.DataFrame([col_init,rfr.feature_importances_], index=['variable','feature_importance']).T.set_index('variable')
df_feat_imp.sort_values('feature_importance')
df_feat_imp.to_csv('feature_importance.csv')
perm_imp=pd.read_csv('permutation_importance_rfr',index_col=0)
col_fil95=perm_imp.sort_values('feature_importances').index[5:]
col_fil90=perm_imp.sort_values('feature_importances').index[10:]

###################################################

pip4=make_pipeline(KBinsDiscretizer(n_bins=20, encode='ordinal',strategy='kmeans'),OneHotEncoder(handle_unknown='ignore'), RidgeCV(alphas=[1000,3160]))
pip5=make_pipeline(PCA(90),KBinsDiscretizer(n_bins=20, encode='ordinal',strategy='kmeans'),OneHotEncoder(handle_unknown='ignore'), RidgeCV(alphas=[1000,3160]))
pip6=make_pipeline(RandomForestRegressor(n_estimators=50,verbose=3,max_depth=18))

para={'kbinsdiscretizer__n_bins':[15,20,25]}
gs1=GridSearchCV(pip4,para,scoring='neg_root_mean_squared_error',cv=3,verbose=1)

resu7=cross_validate(pip4,df_train[col_fil95],target,cv=KFold(n_splits=10,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
affiche_score(resu7)

resu8=cross_validate(pip5,df_train[col_init],target,cv=KFold(n_splits=10,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')

resu9=cross_validate(pip6,df_train[col_fil95],target,cv=KFold(n_splits=10,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
affiche_score(resu9)

resu10=cross_validate(pip6,df_train[col_fil90],target,cv=KFold(n_splits=10,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
affiche_score(resu10)

# n_bins=20, encode='ordinal',strategy='kmeans')
# sans max_depth : train = -3.050 test = -8.060 
# avec max_depth = 20 : train = -3.055 test = -8.046 
# avec max_depth = 19 : train = -6.700 test = -7.936 
# avec max_depth = 18 : train = -7.414 test = -7.923 
#   avec n_bins=25 
#                     : train = -6.988 test = -7.928
# avec max_depth = 17 : train = -7.087 test = -7.933 

# avec max_depth = 15 : train = -7.384 test = -7.924 

# sans binning

# avec max_depth = 21 : train = -6.789 test = -7.934 
# avec max_depth = 20 : train = -6.936 test = -7.927 
# avec max_depth = 19 : train = -7.066 test = -7.932 

# avec max_depth = 18 et 5 features en moins : train = -7.187 std 0.009 test = -7.907 std 0.046 
# avec max_depth = 18 et 10 features en moins : train =  -7.185 std 0.015 test = -7.904 std 0.046 

affiche_score(resu8)
affiche_score(resu7)

[resu7['estimator'][i][3].alpha_ for i in range(10)]

#### ETUDE DES RESIDUS ###
df_viz['target']=target

ypred_train= cross_val_predict(pip4,df_train[col_init],target,cv=KFold(n_splits=10,shuffle=True, random_state=0),verbose=3)
df_viz['y_pred']=ypred_train

residus= ypred_train - target.values
df_viz['residus']=residus

plt.hist(residus)
plt.show()
residus.sum()

fig,axs=plt.subplots(2,1)
axs[0].scatter(x=target.values, y=residus, alpha=0.01)
axs[0].set_xlabel('loss')
axs[0].set_ylabel('residus')
axs[1].scatter(x=target.values, y=ypred_train, alpha=0.01)
axs[1].set_xlabel('loss')
axs[1].set_ylabel('predictions')
plt.show()

df_viz[col_init].corrwith(df_viz['residus']).sort_values()

residus
#####################################
# PREDICT DATASET KAGGLE

###################################""

pip4.fit(df_train,target)
y_pred=pip4.predict(test)

test['loss']=y_pred
test_final=test[['loss']]
test_final
test_final.to_csv('resultat.csv')

df_viz['residus']=residus