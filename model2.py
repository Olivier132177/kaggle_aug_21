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
from sklearn.model_selection import cross_validate,cross_val_predict,KFold,GridSearchCV, train_test_split
from sklearn.feature_selection import f_regression, SelectFromModel, SelectPercentile,mutual_info_regression,SelectKBest
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
from sklearn.kernel_approximation import Nystroem

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

stsc=StandardScaler()
df_train_ss=pd.DataFrame(stsc.fit_transform(df_train),columns=col_init)

perm_imp=pd.read_csv('permutation_importance_rfr',index_col=0)
perm_imp.sort_values('feature_importances', ascending=False)
#####
col_sel=perm_imp.sort_values('feature_importances', ascending=False).head(50).index

#df_train_ss['loss']=target

#tab_var=[]
#for i in range(100):
#    col_fil=perm_imp.sort_values('feature_importances').index[i:]
#    km=KMeans(n_clusters=10,random_state=0)
#    cluster_km=km.fit_predict(df_train_ss[col_fil])
#    df_train_ss['clusters']=cluster_km
#    vari=df_train_ss[['clusters','loss']].groupby('clusters').mean().var()
#    tab_var.append(vari)
#    print ('{} {}'.format(i,vari))

#t_var=pd.Series([tab_var[i][0] for i in range(100)])
#t_var.to_csv('var_kmeans')



X_train, X_test, y_train, y_test=train_test_split(df_train,target,random_state=0,train_size=0.75)

rfr_test=RandomForestRegressor(random_state=0,n_estimators=200, verbose=2, max_depth=20, max_features=0.1)

par={'n_estimators':[300,400],'max_depth':[18,19],'max_features':[0.1]}
gs1=GridSearchCV(rfr_test,param_grid=par,scoring='neg_root_mean_squared_error',cv=KFold(n_splits=3,shuffle=True, random_state=0),verbose=3)
gs1.fit(X_train[col_sel],y_train)

df_gs=pd.DataFrame(gs1.cv_results_)
df_gs['variables']='col_50'
df_gs[['param_max_features','param_n_estimators','param_max_depth','mean_test_score',\
    'std_test_score','variables','mean_fit_time']].sort_values('mean_test_score',ascending=False)

df_gs.to_csv('gs_rfr3.csv')

#record
#param_max_features param_n_estimators param_max_depth  mean_test_score  std_test_score variables  mean_fit_time
#                 0.1                300              19        -7.895078        0.030227    col_50     261.579269

fff=pd.read_csv('gs_rfr_combi.csv',index_col=0)
fff['variables']='all'



cv_test=cross_validate(rfr_test,df_train,target,cv=KFold(n_splits=3,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
df_train_ss

sv=SVR(verbose=3)
cv_sv=cross_validate(sv,df_train_ss,target,cv=KFold(n_splits=3,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')

df_fi=pd.DataFrame([cv_test['estimator'][i].feature_importances_ for i in range(3)],columns=col_init)

sns.boxplot(data=df_fi[df_fi.mean().sort_values(ascending=False).index])
plt.xticks(rotation=90)
plt.show()

cv_test['test_score'].mean()
cv_test['test_score'].std()

#(train=-6.954, test=-7.932)
rfr_test.fit(df_20_km,target)

#rfr_test=RandomForestRegressor(random_state=0,n_estimators=500, verbose=3, max_depth=35)
#rfr_test.fit(df_train[col_20],target)
# kaggle 8.02171

y_pred=rfr_test.predict(test[col_20])

permu=permutation_importance(rfr_test,df_20_km,target,scoring='neg_root_mean_squared_error',random_state=0)
permu['importances_mean']

df_perm_sum=pd.DataFrame([permu['importances_mean'],permu['importances_std']],columns=df_20_km.columns,index=['mean','std']).T.sort_values('mean', ascending=False)

df_perm_sum
rfr_test.feature_importances_
plt.errorbar(x=np.arange(0,20,1),y=df_perm_sum['mean'],yerr=df_perm_sum['std'].values)
plt.show()

gbr=GradientBoostingRegressor(random_state=0,n_estimators=200,verbose=3)

cv_test=cross_validate(gbr,df_train[col_init],target,cv=KFold(n_splits=3,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
affiche_score(cv_test)

# (train=-4.568, test=-7.942) avec max_features =0.5 20 features et max_depth=30
# (train=-4.591, test=-7.953) avec 20 features et max_depth=30
# (train=-5.617, test=-7.931) avec 20 features et max_depth=25

#GradiantBoostingRegressor 200 estimators
    #train mean 7.784 std 0.001 #test score 7.87 std 0.003 (kaggle 7.90981)

df_cv=pd.DataFrame([cv_test['estimator'][i].feature_importances_ for i in range(3)], columns=col_init)

sns.boxplot(data=df_cv)
plt.xticks(rotation=90)
plt.show()

rfr_test.fit(df_train[col_init],target)
#pd.DataFrame([perm_imp['importances_mean'],perm_imp['importances_std']],columns=col_fil,index=['mean','std'])


df_perm_sum=pd.DataFrame([permu_imp['importances_mean'],permu_imp['importances_std']],columns=col_fil,index=['mean','std']).T.sort_values('mean', ascending=False)

col_selec=df_perm_sum.index[:5]

kbd=KBinsDiscretizer(n_bins=30,strategy='quantile',encode='ordinal')
df_kdb=pd.DataFrame(kbd.fit_transform(df_train[col_fil]),columns=col_fil)

# avec col_fil (7 variables)

#50 estimateurs
#Train score : mean -3.079 std 0.001 
#Test score : mean -8.084 std 0.003 

#100 estimateurs
#Train score : mean -3.011 std 0.0 
#Test score : mean -8.043 std 0.003 

#100 estimateurs max_features = 6
#Train score : mean -3.01 std 0.002  
#Test score : mean  -8.044 std 0.009  

#100 estimateurs max_depth = 10
#Train score : mean -7.818 std 0.003   
#Test score : mean  - 7.913 std 0.004   

#200 estimateurs max_depth = 9
#Train score : mean -7.845 std 0.003   
#Test score : mean  -7.913 std 0.003    

#200 estimateurs max_depth = 10
    # col_fil 7 variables
#Train score : mean -7.817 std 0.003   
#Test score : mean  -7.913 std 0.004    
    # col_fil 10 variables
#Train score : mean -7.803 std 0.002   
#Test score : mean  -7.91 std 0.004    
    # col_fil 20 variables
#Train score : mean -7.777 std 0.006   
#Test score : mean  -7.904 std 0.003    

# 200 estimateurs toutes les variables max_depth=None, max_features=0.5 
    # train -2.954, test=-7.957 (sur un seul CV)
#   même chose avec max_depth = 20 :
    # train -6.798 std 0.012, test=-7.898 std 0.004 (kaggle 7.93992)
#  
rfr=RandomForestRegressor(n_estimators=2,random_state=0,verbose=3)
rfr.fit(df_train[col_init],target)
[rfr.estimators_[i].get_depth() for i in range(2)]

df_feat_imp=pd.DataFrame([col_init,rfr.feature_importances_], index=['variable','feature_importance']).T.set_index('variable')
df_feat_imp.sort_values('feature_importance')
df_feat_imp.to_csv('feature_importance.csv')
perm_imp=pd.read_csv('permutation_importance_rfr',index_col=0)
col_fil90=perm_imp.sort_values('feature_importances').index[10:]
col_fil80=perm_imp.sort_values('feature_importances').index[20:]
col_fil70=perm_imp.sort_values('feature_importances').index[30:]

col_v2=np.hstack([clu,col_fil90])
col_v2
###################################################

pip4=make_pipeline(KBinsDiscretizer(n_bins=20, encode='ordinal',strategy='kmeans'),OneHotEncoder(handle_unknown='ignore'), RidgeCV(alphas=[316,1000,3160]))
pip40=make_pipeline(KBinsDiscretizer(n_bins=20, encode='ordinal',strategy='kmeans'),OneHotEncoder(handle_unknown='ignore'), LassoCV(alphas=[50,100,500]))

pip6=make_pipeline(RandomForestRegressor(n_estimators=50,verbose=3,max_depth=18,max_features=0.95))
pip60=make_pipeline(KBinsDiscretizer(n_bins=20, encode='ordinal',strategy='kmeans'),RandomForestRegressor(n_estimators=50,verbose=3,max_depth=18))

resu7=cross_validate(pip4,df_train[col_fil70],target,cv=KFold(n_splits=5,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
affiche_score(resu7)

resu8=cross_validate(pip6,df_train[col_v2],target,cv=KFold(n_splits=5,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
affiche_score(resu8)

resu40=cross_validate(pip40,df_train[col_init],target,cv=KFold(n_splits=5,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
affiche_score(resu40)

resu60=cross_validate(pip60,df_train[col_init],target,cv=KFold(n_splits=5,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')
affiche_score(resu60)

pd.DataFrame([resu40['estimator'][i][2].coef_ for i in range(5)])
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
# avec max_depth = 18 et 20 features en moins : train =  -7.153 std 0.019 test = -7.909 std 0.032 
     # max_features = 0.8                     : train = -7.153 std 0.011 test = 7.908 std 0.034


# KBinsDiscretizer(n_bins=20, encode='ordinal',strategy='kmeans'),OneHotEncoder(handle_unknown='ignore'), RidgeCV(alphas=[316,1000,3160]))
    # avec 95% des variables
        #Train score : mean -7.818 std 0.005 
        #Test score : mean -7.87 std 0.044 
        #   -> kaggle 7.90861
    # avec 90% des variables
        #Train score : mean -7.82 std 0.005 
        #Test score : mean -7.87 std 0.044         #   
    # avec 70% des variables
        #Train score : mean  -7.831 std 0.008  
        #Test score : mean  -7.875 std 0.034          #   
    
affiche_score(resu7)

[resu7['estimator'][i][2].alpha_ for i in range(10)]

#### ETUDE DES RESIDUS ###
df_viz['target']=target

df_feat_imp=pd.DataFrame([resu8['estimator'][i][0].feature_importances_ for i in range(5)],columns=col_v2)
df_feat_imp.mean().sort_values()
sns.boxplot(data=df_feat_imp, orient='h')
plt.show()

ypred_train= cross_val_predict(pip4,df_train[col_init],target,cv=KFold(n_splits=5,shuffle=True, random_state=0),verbose=3)
ypred_train2= cross_val_predict(pip6,df_train[col_init],target,cv=KFold(n_splits=5,shuffle=True, random_state=0),verbose=3)

(mean_squared_error(target,ypred_train))**(1/2)
(mean_squared_error(target,ypred_train2))**(1/2)

moy_pred=(ypred_train+ypred_train2)/2
(mean_squared_error(target,moy_pred))**(1/2)

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

rfr_test.fit(df_train[col_init],target)
y_pred=rfr_test.predict(test[col_init])


[rfr_test.estimators_[i].get_depth() for i in range(200)]

pip4.fit(df_train,target)
y_pred_1=pip4.predict(test)

pip6.fit(df_train,target)
y_pred_2=pip6.predict(test)

gbr.fit(df_train,target)
y_pred=gbr.predict(test)


y_pred = (y_pred_1+y_pred_2)/2

test['loss']=y_pred
test_final=test[['loss']]
test_final.sort_values('loss')
test_final.to_csv('resultat.csv')

df_viz['residus']=residus