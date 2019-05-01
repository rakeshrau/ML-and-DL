# Importing the libraries
import numpy as np
import pandas as pd
import pickle

#==============================================================================
# 
# ######Import training set
# tp = pd.read_csv('training_set_reduced1.tsv', sep='\t',header=None,low_memory=False,iterator=True, chunksize=1000)
# df=pd.DataFrame()
# df = pd.concat(tp, ignore_index=True)
# df.replace(r'\s+', np.nan, regex=True)
# # save features
# pd.to_pickle(df, './training_set_reduced1.pkl')
# df = pd.read_pickle('training_set_reduced1.pkl')
# X = df.iloc[1:, 1:128].values
# y = df.iloc[1:, 0].values
#   
# ##########Import test set
# tp1 = pd.read_csv('sorted_scoring_set.csv', sep='\t',low_memory=False,iterator=True, chunksize=1000)
# df1=pd.DataFrame()
# df1 = pd.concat(tp1, ignore_index=True)
# df1.replace(r'\s+', np.nan, regex=True)
# # save features
# pd.to_pickle(df1, './scoring_set_reduced1.pkl')
# #df = pd.read_pickle('training_set_reduced1.pkl')
# X_score = df1.iloc[1:, 1:128].values
# #y_score = df.iloc[:, 0].values
#==============================================================================
#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()        
#################Reading pickled training set############
df = pd.read_pickle('./training_set_reduced1.pkl')                
X = df.iloc[2:, 1:128].values
y = df.iloc[2:, 0].values
####Label Encoding Training set
for i in range(0,127):
    X[:,i] = labelencoder.fit_transform(X[:,i].astype(str))  
    
y = labelencoder.fit_transform(y.astype(str))   
###Feature Scaling for training  set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X) 
del(df)


######################Part 2:lets make ANN
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout


##Initializing the ANN
classifier = Sequential()
##Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu', input_dim = 127)) #output_dim=number of hidden nodes= (imput nodes + o/p nodes)/2 =11+1/2
classifier.add(Dropout(0.5))                                                      #Weights initialized using uniform function,relu id for rectifier avtivation function
classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu')) 
classifier.add(Dropout(0.5))                                                     #input_dim = n.o of independent variables
#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))#sigmoid for 0 or 1 otherwise softmax if more than 2 categories
##Compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy',metrics = ['accuracy'])#adam is a type of stochastic gradient descent

##Fitting the ANN to the Training Set
classifier.fit(X,y,batch_size =128,epochs =100)



###############Reading pickled test set################
df1 = pd.read_pickle('./scoring_set_reduced1.pkl') 
X_score = df1.iloc[1:, 1:128].values
#Scoring/Test set
for i in range(0,127):
    X_score[:,i] = labelencoder.fit_transform(X_score[:,i].astype(str))

###Feature Scaling for test set
X_score = sc.transform(X_score)
del(df1)
##Predicting the Test Set results
y_pred_xg = classifier.predict(X_score)
y_pred_xg = y_pred_xg.ravel()
y_pred = (y_pred > 0.5) # if greater than 0.5 return TRUE else FLASE    

#==============================================================================
# # Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#==============================================================================

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

##########Feature Scaling for training and test set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_score = sc.transform(X_score)

X1 =X[0:1502510,:]
X2= X[1502511:3005021,:]
X3= X[3005022:4507530,:]
X4= X[4507531:6010042,:]

y1 =y[0:1502510,]
y2= y[1502511:3005021,]
y3= y[3005022:4507530,]
y4= y[4507531:6010042,]

##############XGBoost#########################
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
clf = xgb.XGBModel()
clf_cv = xgb.XGBModel()
clf.n_estimators =1000
clf.learning_rate = 0.3
clf.objective = "binary:logistic"

clf.fit(X, y.ravel(),eval_metric='logloss')
pickle.dump(clf, open('./xgbst.sav', 'wb'))
y_pred_xg= clf.predict(X_score)
#################################Grid Search CV

clf_cv.fit(X1, y1.ravel())
parameters = [{'gamma':[i/10.0 for i in range(0,5)]}    
             ]

cvresult = xgb.cv(params = parameters,folds=10)

grid_search = GridSearchCV(estimator = XGBClassifier(n_estimators=1000,objective='binary:logistic', learning_rate=0.3,max_depth=3,min_child_weight=5), param_grid = parameters,cv =2,scoring = 'neg_log_loss') 
gs = grid_search.fit(X1,y1)
best_parameters = gs.best_params_
classifier1 = XGBClassifier(n_estimators=1000,objective='binary:logistic', learning_rate=0.3,max_depth=3,min_child_weight=5)  #default n_estimators=100 trees objective="binary:logistic" i.e for binary
classifier1.fit(X1, y1.ravel())
pickle.dump(classifier1, open('./xgbst_class_500.sav', 'wb'))

y_pred_xg= classifier1.predict_proba(X_score)[:,1]
#==============================================================================
# # Predicting the Test set results
# from sklearn.metrics import accuracy_score  #Added by Raks
# y_pred = classifier.predict(X_test)
# accuracy_score(y_test,y_pred)*100     #Added by Raks
#==============================================================================

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy',random_state = 0)
classifier.fit(X, y)
classifier.fit(X1, y1)
classifier.fit(X2, y2)
classifier.fit(X3, y3)
classifier.fit(X4, y4)

y_pred_prob = classifier.predict_proba(X_score)[:,1]
y_pred_prob_xg = y_pred_xg[:,1]

pd5 = pd.read_csv('instance_id_unsorted.csv', header=None)
ids = pd5.iloc[:, 0:1888195].values
ids = ids.ravel()


predictions= y_pred_prob
predictions=y_pred_xg
pd6 = pd.DataFrame({"INNOVATION_CHALLENGE_KEY" : ids, "RENEWAL_PROBABLIITY" : predictions})
pd6.to_csv("submission_xgb1000.csv", index=False) 
           
f=open('abc.tsv','ab')
np.savetxt(f,y_pred_prob,delimiter=',')
f.close()



pickle.dump(classifier, open('./contract.sav', 'wb'))

classifier= pickle.load( open( './contract.sav', 'rb' ) )


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