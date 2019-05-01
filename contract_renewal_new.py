# Importing the libraries
import numpy as np
import pandas as pd


tp = pd.read_csv('training_set_reduced1.tsv', sep='\t',header=None,low_memory=False,iterator=True, chunksize=1000)
df=pd.DataFrame()
df = pd.concat(tp, ignore_index=True)
df.replace(r'\s+', np.nan, regex=True)
# save features
pd.to_pickle(df, './training_set_reduced1.pkl')
#df = pd.read_pickle('training_set_reduced1.pkl')
X = df.iloc[:, 1:128].values
y = df.iloc[:, 0].values
         
#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
#df.apply(LabelEncoder().fit_transform)
#df = df.apply(LabelEncoder().fit_transform(df.astype(str)))

labelencoder = LabelEncoder()
for i in range(0,127):
    X[:,i] = labelencoder.fit_transform(X[:,i].astype(str))  
    
y = labelencoder.fit_transform(y.astype(str))   

del(df)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#y_train = y_train.reshape(y_train.shape[0],1)
#y_test = y_test.reshape(y_test.shape[0],1)
#del(X)
#del(y)


#==============================================================================
# y_train.astype(float)
# y_train = sc.fit_transform(y_train)
# y_test = sc.transform(y_test)
# 
# ###########Implementing XGboost
# #from xgboost import XGBClassifier
# from xgboost.sklearn import XGBClassifier
#==============================================================================


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#==============================================================================
# 
# import xgboost as xgb
# clf = xgb.XGBModel()
# clf.fit(X_train, y_train.ravel(),eval_metric='logloss')
# y_pred_xg= clf.predict(X_test)
# classifier1 = XGBClassifier()  #default n_estimators=100 trees objective="binary:logistic" i.e for binary
# classifier1.fit(X_train,y_train)
#==============================================================================

#==============================================================================
# # Predicting the Test set results
# from sklearn.metrics import accuracy_score  #Added by Raks
# y_pred = classifier.predict(X_test)
# accuracy_score(y_test,y_pred)*100     #Added by Raks
#==============================================================================

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred_prob = classifier.predict_proba(X_test)[:,1]

#==============================================================================
# #Fitting Random Forest Regressor
# from sklearn.ensemble import  RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0,n_jobs=-1)
# regressor.fit(X_train, y_train)
# y_pred_new = regressor.predict(X_test)
#==============================================================================


from sklearn.metrics import log_loss
lloss = log_loss(y_test, y_pred_prob)
print(lloss)