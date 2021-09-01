import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer,OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

fich=pd.read_csv('train.csv').set_index('id')
test=pd.read_csv('test.csv').set_index('id')

target=fich['loss']
df_train=fich.iloc[:,:-1]

pip_rid=make_pipeline(KBinsDiscretizer(encode='ordinal',strategy='kmeans'),\
    OneHotEncoder(handle_unknown='ignore'), RidgeCV(alphas=[316,1000,3160]))
pip_rid.fit(df_train,target)
y_pred_rid=pip_rid.predict(test)

# Etude des coefficents
df_coefs=pd.DataFrame(pip_rid[2].coef_,pip_rid[1].get_feature_names(df_train.columns))
df_coefs.columns=['coef']
df_coefs['abs_coef']=np.abs(df_coefs['coef'])
df_coefs=df_coefs.sort_values('abs_coef')
plt.barh(np.arange(0,50),df_coefs['coef'].head(50))
plt.yticks(np.arange(0,50),df_coefs.index[:50])
plt.title('Ridgee : les 50 coefficients les plus importants')
plt.xlabel('coefficient')
plt.ylabel('variables')
plt.show()

gbr=GradientBoostingRegressor(random_state=0,n_estimators=500,verbose=3)
gbr.fit(df_train,target)
y_pred_gbr=gbr.predict(test)

#Importance des variables
perm=permutation_importance(gbr, df_train, target, scoring='neg_root_mean_squared_error',n_jobs=4, random_state=0)
df_feat_imp=pd.DataFrame(perm['importances'], index=df_train.columns).T
ordre=pd.Series(perm['importances_mean'],df_train.columns).sort_values(ascending=False).index
sns.boxplot(data=df_feat_imp[ordre])
plt.xticks(rotation=90)
plt.title('permutation importance')
plt.ylabel('RMSE')
plt.xlabel('features')
plt.show()

# moyenne des 2 modèles 

test['loss']=(y_pred_rid+y_pred_gbr)/2
test_final=test[['loss']]
test_final.to_csv('resultat.csv')
