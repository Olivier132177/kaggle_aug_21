import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer,OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline

fich=pd.read_csv('train.csv').set_index('id')
test=pd.read_csv('test.csv').set_index('id')

target=fich['loss']
df_train=fich.iloc[:,:-1]

gbr=GradientBoostingRegressor(random_state=0,n_estimators=300,verbose=3)
gbr.fit(df_train,target)
y_pred_2=gbr.predict(test)

pip_rid=make_pipeline(KBinsDiscretizer(n_bins=20, encode='ordinal',strategy='kmeans'),OneHotEncoder(handle_unknown='ignore'), RidgeCV(alphas=[316,1000,3160]))
pip_rid.fit(df_train,target)
y_pred_1=pip_rid.predict(test)

test['loss']=(y_pred_1+y_pred_2)/2
test_final=test[['loss']]
test_final.to_csv('resultat.csv')