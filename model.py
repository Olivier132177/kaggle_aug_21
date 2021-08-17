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


def essai_bins(df,variable,target,n_bins, n_bins_q,bins_pers):
    ''' à partir d'une colonne d'un dataframe donné, établit un graph avec 
    3 subplots indiquant le nombre d'enregistrements par bin et la moyenne de la target par bin : 1 subplot avec des bins à largeur fixe, 
    1 subplot avec des bins divisés par quantile, 1 subplot avec des valeurs de bins customisés

    Args :

    df : le DataFrame
    variable : la variable correspondant à un nom de colonne du DataFrame
    target : la target (valeur numérique)
    n_bins : le nombre de bins à intervalle régulier pour le subplot 1
    n_bins_q : le nombre de bins divisés par quantile pour le subplot 2
    bins_pers : les valeurs de bins personnalisés dans un array

    Returns :  un graph avec 3 subplots

    '''    
    exemple=df.copy()[[variable,target]]
    exemple['bin']=pd.cut(fich[variable],n_bins,include_lowest=True)
    exemple['binq']=pd.qcut(fich[variable],n_bins_q)
    exemple['bin_pers']=pd.cut(fich[variable],bins_pers)
    exemple['bin_str']=exemple['bin'].astype('str')
    exemple['binq_str']=exemple['binq'].astype('str')

    xt=exemple.sort_values(variable)['bin_str'].unique()
    xtq=exemple.sort_values(variable)['binq_str'].unique()

    fig, axs=plt.subplots(1,3,sharey='row')

    fig.suptitle('{} pearson corr : {}'.format(variable,exemple[target].corr(exemple[variable])))
    axs[0].bar(np.arange(0,n_bins),exemple.groupby('bin')[target].count().values, label='nombre')
    axs[0].legend(loc=2)
    axs[0].set_xticks(np.arange(0,n_bins))
    axs[0].set_xticklabels(xt,rotation=90)
    axs[0].set_title('Bins à intervalles réguliers')
    fig2=axs[0].twinx()
    fig2.plot(exemple.groupby('bin')[target].mean().values, label='moyenne',c='r',marker='D')
    fig2.legend(loc=1)

    axs[1].bar(np.arange(0,n_bins_q),exemple.groupby('binq')[target].count().values, label='nombre')
    axs[1].legend(loc=2)
    axs[1].set_xticks(np.arange(0,n_bins_q))
    axs[1].set_xticklabels(xtq,rotation=90)
    axs[1].set_title('Bins par quantile')
    axs[1].grid(True)
    fig2q=axs[1].twinx()

    fig2q.plot(exemple.groupby('binq')[target].mean().values, label='moyenne',c='r',marker='D')
    fig2q.legend(loc=1)

    axs[2].set_title('Bins personnalisés')
    axs[2].bar(np.arange(0,len(bins_pers)-1),exemple.groupby('bin_pers')[target].count().values, label='nombre')
    axs[2].legend(loc=2)
    axs[2].set_xticks(np.arange(0,len(bins_pers)))
    fig2pers=axs[2].twinx()
    fig2pers.plot(exemple.groupby('bin_pers')[target].mean().values, label='moyenne',c='r',marker='D')
    fig2pers.legend(loc=1)

    fig2.get_shared_y_axes().join(fig2,fig2q,fig2pers)

    plt.tight_layout()
    plt.show(block=False)


########################################################"
 

fich=pd.read_csv('train.csv').set_index('id')
test=pd.read_csv('test.csv').set_index('id')

target=fich['loss']
df_train=fich.iloc[:,:-1]
df_viz=fich.copy()
col_init=df_train.columns

stsc=StandardScaler()
df_train2=pd.DataFrame(stsc.fit_transform(df_train),columns=col_init)
df_train2
###################################  VISUALISATION
# DISTRIBUTION DE lA TARGET

quan1=QuantileTransformer(output_distribution='uniform')
quan2=QuantileTransformer(output_distribution='normal')
pow=PowerTransformer(method='yeo-johnson')

target_1=quan1.fit_transform(target.values.reshape(-1,1))
target_2=quan2.fit_transform(target.values.reshape(-1,1))
target_3=pow.fit_transform(target.values.reshape(-1,1))

plt.subplot(221)
plt.hist(target)
plt.title('Variable originale')
plt.subplot(222)
plt.hist(pow.fit_transform(target.values.reshape(-1,1)))
plt.title('PowerTransform Yeo-Johnson')
plt.subplot(223)
plt.hist(quan1.fit_transform(target.values.reshape(-1,1)))
plt.title('QuantileTransform uniform')
plt.subplot(224)
plt.hist(quan2.fit_transform(target.values.reshape(-1,1)))
plt.title('QuantileTransform normal')
plt.show()

# DISTRIBUTION DES VARIABLES

for i in range(100):
    plt.subplot(10,10,i+1)
    plt.hist(df_train.iloc[:,i])
    plt.title(col_init[i])
plt.show()

# SCATTERPLOT VARIABLES TARGET

for i in range(100):
    plt.subplot(10,10,i+1)
    plt.scatter(df_train.iloc[:,i].values, target.values, alpha=0.01)
    plt.title('{} : {}'.format(col_init[i],round(df_train.iloc[:,i].corr(target),2)))
plt.show()


# TEST DE CREATION DE FEATURES
'''
# RATIOS

t_coef_div=[]
perm=permutations(col_init,2)
for i in perm :
    if i[1] not in ['f27','f55','f1','f86']:
        t_div= df_train.loc[:,i[0]]/df_train.loc[:,i[1]]
        t_coef_div.append(t_div.corr(target))
np.sort(t_coef_div)

# INVERSES

t_coef_inv=[]
for i in col_init :
    if i not in ['f27','f55','f1','f86']:
        t_inv= 1/df_train.loc[:,i]
        t_coef_inv.append(t_inv.corr(target))
np.sort(t_coef_inv)[:10]
np.sort(t_coef_inv)[-10:]

# ADDITIONS SOUSTRACTIONS

comb=combinations(col_init,2)
t_coef_add=[]
t_coef_sub_1=[]
t_coef_sub_2=[]
for i in comb :
    som= df_train2.loc[:,i[0]] + df_train2.loc[:,i[1]]
    t_coef_add.append(som.corr(target))
    sub_1= df_train2.loc[:,i[0]] - df_train2.loc[:,i[1]]
    sub_2= df_train2.loc[:,i[1]] - df_train2.loc[:,i[0]]
    t_coef_sub_1.append(sub_1.corr(target))
    t_coef_sub_2.append(sub_2.corr(target))

np.sort(t_coef_add)
np.sort(t_coef_sub_1)
np.sort(t_coef_sub_2)


# ACP

acp=PCA(100, random_state=0)
df_acp=pd.DataFrame(acp.fit_transform(df_train[col_init]))
df_acp.corrwith(target).sort_values()

df_acp.hist()
plt.show()

# SCATTERPLOT VARIABLES ACP TARGET

for i in range(100):
    plt.subplot(10,10,i+1)
    plt.scatter(df_acp.iloc[:,i].values, target.values, alpha=0.01)
    plt.title(' comp {} : {}'.format(i+1,round(df_acp.iloc[:,i].corr(target),2)))
plt.show()
'''
# KBD des variables 
qu=QuantileTransformer(output_distribution='normal')

essai_bins(df=fich,variable='f99',target='loss',n_bins=20,n_bins_q=25,bins_pers=[-0.7,1.206,2.787,5])

def crea_var(df_train):
    df_train['c1']=pd.cut(df_train['f1'],bins=[-18,8,97,300],labels=False)
    df_train['c2']=pd.cut(df_train['f2'],bins=[-8,-1.024,-0.14,1.069,10],labels=False)
    df_train['c4']=pd.cut(df_train['f4'],bins=[-8000,98.442,943,11213,50000],labels=False)
    df_train['c7']=pd.cut(df_train['f7'],bins=[-5,-0.764,-0.0262,0.818,0.934,1.07,1.332,5],labels=False)
    df_train['c8']=pd.cut(df_train['f8'],bins=[-503,-110,-39,21.733,50.606,123.86,500],labels=False)
    df_train['c9']=pd.cut(df_train['f9'],bins=[0.9,1.116,1.141,1.181,1.45,1.8],labels=False)
    df_train['c11']=pd.cut(df_train['f11'],bins=[-1.86,-0.776,-0.313,2.5],labels=False)
    df_train['c12']=pd.cut(df_train['f12'],bins=[108,122,124.815,139.79,152.63,175],labels=False)
    df_train['c14']=pd.cut(df_train['f14'],bins=[-9,-2.47,0.723,2.5,10],labels=False)
    df_train['c16']=pd.cut(df_train['f16'],bins=[-3000000,14400000,35000000],labels=False)
    df_train['c17']=pd.cut(df_train['f17'],bins=[-0.4,-0.01,0.0447,0.6],labels=False)
    df_train['c20']=pd.cut(df_train['f20'],bins=[-0.05,0.00388,0.0548,0.109,0.278,0.681,1.5],labels=False)
    df_train['c22']=pd.cut(df_train['f22'],bins=[-11000,11500,50000],labels=False)
    df_train['c29']=pd.cut(df_train['f29'],bins=[0.2,2.2,8.2,25],labels=False)
    df_train['c30']=pd.cut(df_train['f30'],bins=[0.8,1.114,1.179,2],labels=False)
    df_train['c32']=pd.cut(df_train['f32'],bins=[-2,-0.0343,0.1,0.7,2],labels=False)
    df_train['c33']=pd.cut(df_train['f33'],bins=[-0.5,0.782,0.941,1.570,4,8],labels=False)
    df_train['c34']=pd.cut(df_train['f34'],bins=[100,124.66,133.36,159.14,180],labels=False)
    df_train['c35']=pd.cut(df_train['f35'],bins=[-30,22.7,99.95,270,1100],labels=False)
    df_train['c36']=pd.cut(df_train['f36'],bins=[-3,3,4.08,6.123,10.03,25],labels=False)
    df_train['c38']=pd.cut(df_train['f38'],bins=[-6,1.2,1.45,8],labels=False)
    df_train['c39']=pd.cut(df_train['f39'],bins=[-2,0.957,1.027,1.274,3],labels=False)
    df_train['c41']=pd.cut(df_train['f41'],bins=[-7,1.248,1.874,2.094,4.117,15],labels=False)
    df_train['c42']=pd.cut(df_train['f42'],bins=[-0.1,0.0092,0.0796,0.0949,0.147,0.169,0.6],labels=False)
    df_train['c48']=pd.cut(df_train['f48'],bins=[-2,-1.318,-0.226,0.371,0.945,2],labels=False)
    df_train['c50']=pd.cut(df_train['f50'],bins=[-0.6,1.18,1.364,1.827,2.532,2.938,3.504,6],labels=False)
    df_train['c51']=pd.cut(df_train['f51'],bins=[-2,0.0102,0.149,0.854,1.101,1.181,1.309,1.472,3],labels=False)
    df_train['c54']=pd.cut(df_train['f54'],bins=[-2,0.183,1.606,10],labels=False)
    df_train['c55']=pd.cut(df_train['f55'],bins=[-400,3,406,726,1147,5400],labels=False)
    df_train['c57']=pd.cut(df_train['f57'],bins=[0.6,1.21,1.487,2.111,2.986,3.877,5],labels=False)
    df_train['c59']=pd.cut(df_train['f59'],bins=[-2,2.529,10],labels=False)
    df_train['c60']=pd.cut(df_train['f60'],bins=[-94000000,99000000,2731000000,5849000000,8106000000,15000000000],labels=False)
    df_train['c61']=pd.cut(df_train['f61'],bins=[-20,48.5,75.3,90.6,115.3,200],labels=False)
    df_train['c62']=pd.cut(df_train['f62'],bins=[-2,0.501,8.314,20],labels=False)
    df_train['c65']=pd.cut(df_train['f65'],bins=[-1,0.6,2],labels=False)
    df_train['c67']=pd.cut(df_train['f67'],bins=[-2,-0.722,0.819,1.282,1.893,3],labels=False)
    df_train['c68']=pd.cut(df_train['f68'],bins=[-30,-7.7,-0.171,13.65,50],labels=False)
    df_train['c69']=pd.cut(df_train['f69'],bins=[-17,0.64,1.475,4.502,20],labels=False)
    df_train['c70']=pd.cut(df_train['f70'],bins=[-11,4.289,119.815,200],labels=False)
    df_train['c71']=pd.cut(df_train['f71'],bins=[-0.3,-0.0292,0.0337,0.5],labels=False)
    df_train['c74']=pd.cut(df_train['f74'],bins=[-1.5,0.428,3.629,4.871,5.893,13],labels=False)
    df_train['c76']=pd.cut(df_train['f76'],bins=[-3.-0.548,-0.0266,0.552,1.016,1.21,3],labels=False)
    df_train['c77']=pd.cut(df_train['f77'],bins=[-45,-2.152,5.018,60],labels=False)
    df_train['c79']=pd.cut(df_train['f79'],bins=[0.9,1.049,1.104,1.117,1.132,1.158,1.163,1.167,1.173,1.196,1.235,2],labels=False)
    df_train['c80']=pd.cut(df_train['f80'],bins=[-0.6,1.379,1.579,1.686,1.778,1.918,2.281,2.412,2.686, 3.168,6],labels=False)
    df_train['c81']=pd.cut(df_train['f81'],bins=[-0.2,-0.0021,0.0016,0.00506,0.00901,0.0156,0.0247,0.0742,0.3],labels=False)
    df_train['c84']=pd.cut(df_train['f84'],bins=[0.4,1.565,2.05,2.352,6.911,7.133,9],labels=False)
    df_train['c85']=pd.cut(df_train['f85'],bins=[-4,4.101,4.287,4.453,5.063,5.366,11],labels=False)
    df_train['c86']=pd.cut(df_train['f86'],bins=[-27,43,62,109,384,1088,2000],labels=False)
    df_train['c87']=pd.cut(df_train['f87'],bins=[-2,-0.336, 0.133,0.248,0.576,1.04,1.794,4],labels=False)
    df_train['c89']=pd.cut(df_train['f89'],bins=[-10,-1.133,1.007,1.8,35.506,75],labels=False)
    df_train['c90']=pd.cut(df_train['f90'],bins=[-0.5,0.234,0.373,0.464,0.57,0.783,1.565,4],labels=False)
    df_train['c92']=pd.cut(df_train['f92'],bins=[-5,1.42,32.849,80],labels=False)
    df_train['c93']=pd.cut(df_train['f93'],bins=[0.05,1.463,1.885,2.071,3.535,5.403],labels=False)
    df_train['c94']=pd.cut(df_train['f94'],bins=[-3,-0.524,0.349,0.998,2],labels=False)
    df_train['c95']=pd.cut(df_train['f95'],bins=[-25,-10.241,-4.248,6.499,13.996,45],labels=False)
    df_train['c97']=pd.cut(df_train['f97'],bins=[0,0.165,0.323,0.352,0.41,0.438,0.655,0.873,1.2],labels=False)
    df_train['c98']=pd.cut(df_train['f98'],bins=[-0.7,1.206,2.787,5],labels=False)
    
    col_orig=['f0','f3','f5','f13','f21','f23','f25','f27','f28','f31','f43',\
        'f46','f47','f52','f53','f58','f63','f64','f66','f73','f78','f82','f88',\
        'f96','f99']
    # idealement quantiletransform pour f27
    col_bins=['c1','c2','c4','c7','c8','c9','c11','c12','c14','c16','c17',\
        'c20','c22','c29','c30','c32','c33','c34','c35','c36','c38','c39',\
        'c41','c42','c48','c50','c51','c54','c55','c59','c60','c61','c62',\
        'c68','c69','c70','c71','c74','c76','c77','c79','c80','c81','c84',\
        'c85','c86','c87','c89','c90','c92','c93','c94','c95','c97','c98']
    non_utilise=['f6','f10','f15','f18','f19','f24','f26','f37','f40','f44',\
        'f45','f49','f56','f72','f75','f83','f91']

    return df_train,col_orig,col_bins,non_utilise

df_train,col_orig,col_bins,non_utilise=crea_var(df_train)

hbr=HistGradientBoostingRegressor(verbose=3)
gs1=GridSearchCV(hbr)

df_train[col_init]

hbr.get_params()
cv_hbr=cross_validate(hbr,df_train[np.hstack([col_orig,col_bins])],target, \
    scoring='neg_root_mean_squared_error',cv=KFold(n_splits=10,shuffle=True,random_state=0),return_estimator=True,return_train_score=True,verbose=3)

cv_hbr['train_score'].mean()
cv_hbr['train_score'].std()
cv_hbr['test_score'].mean()
cv_hbr['test_score'].std()


test,_,_,_=crea_var(test)

pip_a=make_pipeline(StandardScaler(with_mean=False),PowerTransformer(method='yeo-johnson'))
mct=make_column_transformer(
    (pip_a, col_orig),
    (OneHotEncoder(handle_unknown='ignore'),col_bins)
    )
pip_1=make_pipeline(mct,RidgeCV(alphas=[316,1000,3160]))

cv_bins=cross_validate(pip_1,df_train,target_1,scoring='neg_root_mean_squared_error',cv=KFold(n_splits=10,shuffle=True,random_state=0),\
    return_estimator=True,return_train_score=True,verbose=3)

print('Train set : Mean {} Std {}\nTest set : Mean {} Std {}'.format(round(cv_bins['train_score'].mean(),3),round(cv_bins['train_score'].std(),3),\
    round(cv_bins['test_score'].mean(),3),round(cv_bins['test_score'].std(),3)))
[cv_bins['estimator'][i][1].alpha_ for i in range(10)]

result_cv_bins=cross_val_predict(pip_1,df_train,target,verbose=3,cv=KFold(n_splits=10,shuffle=True,random_state=0))
result_cv_bins

#y_pred=quan1.inverse_transform(result_cv_bins)     #8.58 si quantiletransform uniform sur la target
#rmse= (mean_squared_error(target,y_pred))**(1/2)    
#y_pred=quan2.inverse_transform(result_cv_bins)      #10.03 si quantiletransform normal sur la target
#rmse= (mean_squared_error(target,y_pred))**(1/2)
#y_pred=pow.inverse_transform(result_cv_bins)        #8.49 si powertransform yeo-johnson sur la target
#rmse= (mean_squared_error(target,y_pred))**(1/2)

rmse= (mean_squared_error(target,result_cv_bins))**(1/2)    #7.86
rmse                                                    

#round_y_pred= [np.ceil(x) for x in result_cv_bins]

#rmse= (mean_squared_error(target,round_y_pred))**(1/2)      #7.87
#rmse                                                    

pip_1.fit(df_train,target)
y_pred=pip_1.predict(test)

# Avec les variables jusqu'à f10 : train set : 7.93 std 0.005 test set 7.931 0.044
# Avec les variables jusqu'à f19 : train set : 7.924 std 0.005 test set 7.925 0.044
# Avec les variables jusqu'à f30 : train set : 7.916 std 0.005 test set 7.917 0.044
# Avec les variables jusqu'à f39 : train set : 7.909 std 0.005 test set 7.911 0.044 (kaggle 7.95961)
#      vs modele Ridge produisant le meilleur résultat à date
#                                   train set : 7.890 std 0.005 test set 7.917 0.044
#                avec le même alpha train set : 7.896 std 0.005 test set 7.915 0.044
# Avec les variables jusqu'à f49 : train set : 7.902  std 0.005  test set 7.905 0.44  
# Avec les variables jusqu'à f59 : train set : 7.89  std 0.005  test set 7.893 0.43  
# Avec les variables jusqu'à f69 : train set : 7.883  std 0.005  test set 7.887 0.43  
#      vs modele Ridge produisant le meilleur résultat à date
#                                   train set :  7.849 std 0.05 test set 7.888 0.043 
# Avec les variables jusqu'à f79 : train set : 7.874  std 0.05 test set 7.879 0.043   
# Avec les variables jusqu'à f89 : train set :  7.863 std 0.05 test set 7.869 0.043    
# Avec toutes les variables :      train set :  7.858 std 0.05 test set 7.865 0.043 (kaggle 7.91842)
#    après un powertransform des variables numériques non binées :
#                                   train set :  7.857 std 0.005 test set 7.865 0.043 
#df_train[col_init].corrwith(target)

col_ajou=df_train.columns[np.isin(df_train.columns,col_init, invert=True)]

acp=PCA(100, random_state=0)
df_acp=pd.DataFrame(acp.fit_transform(df_train[col_init]))

# MODELISATION

kbd=KBinsDiscretizer(n_bins=10, encode='ordinal',strategy='kmeans')

df_kbd=pd.DataFrame(kbd.fit_transform(df_train))
col_kbd=['{}kbd'.format(x) for x in col_init]
df_kbd.columns=col_kbd
df_kbd.index=df_train.index
df_train_g=df_train.join(df_kbd)

df_test=pd.DataFrame(kbd.fit_transform(test))
df_test.index=test.index
df_test.columns=col_kbd
df_test2=test.join(df_test)

df_train_g.columns==df_test2.columns

###################################################


pip4=make_pipeline(KBinsDiscretizer(n_bins=20, encode='ordinal',strategy='quantile'),OneHotEncoder(handle_unknown='ignore'), RidgeCV(alphas=[1000,3160]))
#  SelectPercentile(f_regression,percentile=95),
pip4.get_params()
para={'kbinsdiscretizer__n_bins':[15,20,25]}
gs1=GridSearchCV(pip4,para,scoring='neg_root_mean_squared_error',cv=3,verbose=1)

resu7=cross_validate(pip4,df_train[col_init],target,cv=KFold(n_splits=10,shuffle=True, random_state=0),\
    verbose=3,return_estimator=True,return_train_score=True,scoring='neg_root_mean_squared_error')

resu7['test_score'].mean()
resu7['test_score'].std()
resu7['train_score'].mean()
resu7['train_score'].std()

[resu7['estimator'][i][2].alpha_ for i in range(10)]

#### ETUDE DES RESIDUS ###

ypred_train= cross_val_predict(pip4,df_train,target,cv=KFold(n_splits=10,shuffle=True, random_state=0),verbose=3)

residus= y_pred - target.values
residus.sum()

fig,axs=plt.subplots(2,1)
axs[0].scatter(x=target.values, y=residus, alpha=0.01)
axs[0].set_xlabel('loss')
axs[0].set_ylabel('residus')
axs[1].scatter(x=target.values, y=y_pred, alpha=0.01)
axs[1].set_xlabel('loss')
axs[1].set_ylabel('predictions')
plt.show()

#####################################
# PREDICT DATASET KAGGLE

###################################""

pip4.fit(df_train,target)
y_pred=pip4.predict(test)

plt.hist([int(x) for  x in y_pred])
plt.show()

#test['loss']=pw0.inverse_transform(y_pred.reshape(-1,1))
test['loss']=y_pred
test_final=test[['loss']]
test_final
test_final.to_csv('resultat.csv')


#1ere soumission
# standardscaler + ridgecv
# cross validate 10 kfold shuffle mean 7.898 std 0.044 
# 7.93925 sur le test set

# Nouvelle pipeline
# make_pipeline(StandardScaler(with_std=False),PowerTransformer(method='yeo-johnson'),RidgeCV())
# sans effet 

#LassoCV à la place de RidgeCV
# mean 7.897 std 0.044 (quasi identique)

#Powertransform(yeo-johnson) du label
# cross validate 10 kfold shuffle mean 0.992 std 0.002
#8.60001 sur le test set

#Quantiletransform(normal) du label
# cross validate 10 kfold shuffle mean 2.458 std 0.008
#10.29488 sur le test set

#pipeline avec PolynomialFeatures : bug

#pip4=make_pipeline(KBinsDiscretizer(encode='ordinal'), OneHotEncoder(handle_unknown='ignore'),RidgeCV()) 
#cross validate 10 kfold shuffle 7.884 0.042 5 bins # (7.92455 sur me test set)
#4 bins : 7.89 0.042
#6 bins : 7.882 0.043
#7 bins : 7.878 0.043 (7.92031 sur le test set)
#10 bins : 7.875 0.043 (7.91585 sur le test set)
#   avec strategy "uniform" 7.874 0.042 (7.91585 sur le test set)
#   avec strategy "kmeans" 7.868 0.043 (7.90894 sur le test set)
#       suppression des target avec valeur >35 : 7.683 std 0.041 test set 7.90971
#       suppression de 2 variables : train 7.837 0.005 test 7.868 0.043
#       sur ACP : test 7.882 0.045
#       sur ACP + kbindisc 7.833 0.006 7.887 0.045
#       avec variables originales + version kbd : test 7.867 0.043 kaggle 7.92858
#       avec les 90 Kbest avant onehotencoder: test 7.872 0.043
#       avec les 0.95 KPercentile(mutual_info_regression) après onehotencoder: test 7.934 0.044
#           avec alpha=100 test 7.866 0.043
#       avec alpha =20 7.868 0.043
#       avec alpha =100 7.686 0.043
#       avec 15 bins au lieu de 10 : test set 7.875 0.043 (kaggle  7.90929)
#       25 bins : train set 7.808 0.005 test set 7.874 0.044
#       augmentation de l'alpha à 3160 (kaggle 7.90874)
#18 bins : 7.886 0.043
#20 bins et strategy quantile : train 7.818 0.005 test 7.869 0.043
#25 bins : 7.882 0.044
#50 bins : 7.894 0.044

# même chose avec PowerTransform de la target : 8.57534 sur le test_set

#ajout d'un selectFromModel (Lasso avec alpha = 0.01)
#meme chose avec un SelectPercentile avec gridsearch
# 7.874 0.043

# Randomforestregressor
# make_pipeline(KBinsDiscretizer(n_bins=10, encode='ordinal',strategy='kmeans'), OneHotEncoder(handle_unknown='ignore'),\
#     RandomForestRegressor(n_estimators=100, max_depth=13,verbose=3)) #7.916 0.005
# sans preprocessing train set 7.554 0.015 test set 7.904 0.0078 (7.94618 sur kaggle)
#   suppression de 2 variables train set 7.552 0.019 test set 7.905 0.0069
#       avec n_estimators=500 train set 7.551 0.019 test set 7.902 0.0076
# avec max_depth=18 train set 7.030 0.021 test set 7.907 0.0091
# avec max_depth=16 train set  7.270 0.009 test set 7.905 0.021
# avec max_depth=10 train set  7.747 0.013 test set 7.908 0.008
