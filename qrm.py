import pandas as pd
import numpy as np

df = pd.read_excel('QRM_4_Years_data_trimmed.xlsx')
df.replace(r'\s+', np.nan, regex=True)
X= df.iloc[:,:-1].values
y= df.iloc[:,-1].values

#Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in range(X.shape[1]):
    X[:,i] = labelencoder.fit_transform(X[:,i])  

y = labelencoder.fit_transform(y)   

#Splitting Training and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# #Fitting Random Forest Classifier Accuracy=96.1,F1 Score=98.01
# from sklearn.ensemble import RandomForestClassifier
# classifier =RandomForestClassifier(n_estimators=200,criterion='entropy',random_state=1234)
# classifier.fit(X_train,y_train)
# y_pred =classifier.predict(X_test)
# y_temp = labelencoder.inverse_transform(y_pred)
# y_final = y_temp.astype(str).reshape(y_temp.size,1)
# =============================================================================


#import xgboost as xgb
from xgboost import XGBClassifier
classifier1 = XGBClassifier(n_estimators=200,objective='binary:logistic',max_depth=3,min_child_weight=5,gamma=0.0,subsample=0.9,colsample_bytree=0.6,reg_alpha=0,learning_rate=0.02,random_state=1234)  #default n_estimators=100 trees objective="binary:logistic" i.e for binary
classifier1.fit(X_train,y_train)
y_pred =classifier1.predict(X_test)
y_temp = labelencoder.inverse_transform(y_pred)
y_final = y_temp.astype(str).reshape(y_temp.size,1)
# =============================================================================
# ###########Grid Search CV
# from sklearn.grid_search import GridSearchCV
# param_test1 = {
# 'learning_rate':[0.1,0.01,0.001,0.005,0.2,0.02,0.002,0.4,0.04,0.004,0.5,0.05,0.7,0.07,0.007,0.9,0.09,0.009]
# 'subsample':[i/100.0 for i in range(75,90,5)]
# }
# gsearch1 = GridSearchCV(estimator=XGBClassifier(n_estimators=200,objective='binary:logistic',max_depth=3,min_child_weight=5,gamma=0.0,subsample=0.9,colsample_bytree=0.6,reg_alpha=0,learning_rate=0.02),param_grid = param_test1,scoring='roc_auc',cv=5)
# gsearch1.fit(X_train,y_train)
# print(gsearch1.best_params_)
# 
# =============================================================================
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)

#Calculating F1 score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

f1 = f1_score(y_pred,y_test)*100
ac1= accuracy_score(y_pred,y_test)*100

