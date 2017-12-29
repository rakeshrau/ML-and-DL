###This program compares the accuracy of the dataset in prediction using Random Forest vs. Improved Weighted voting in RF 
###based on the prediction accuracy of each tree in CV set

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Chronic_Kidney_Disease.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 24].values
                
# Splitting the dataset into the Training set and Test set
#Training set=60%,Cross Validation=20%,Test Set=20% 
from sklearn.cross_validation import train_test_split
X, X_test, y, y_test = train_test_split(X, y, test_size = 0.20, train_size= 0.80,random_state =1234)#,random_state = 1234567
X_train,X_cv,y_train, y_cv = train_test_split(X,y,test_size = 0.35,train_size =0.65,random_state =1234)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_cv = sc.transform(X_cv)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'gini',max_features='log2')
classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score  #Added by Raks
from sklearn.metrics import f1_score   #Added by Raks
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
num_tree=50
tree_vote =[]
for tree in range(num_tree):
    print(classifier.estimators_[tree].predict(X_cv))
    #ac=accuracy_score(y_cv,classifier.estimators_[tree].predict(X_cv))    
    
    ##Get the F1 Score for each Decision Tree
    ac=f1_score(y_cv,classifier.estimators_[tree].predict(X_cv))
    #ac=recall_score(y_cv,classifier.estimators_[tree].predict(X_cv))
    tree_vote.append(ac)
##get the voting right of each Decision Tree
print(tree_vote)
# Predicting the Cross Validation set results
y_pred_cv = classifier.predict(X_cv)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_cv, y_pred_cv)
accuracy_score(y_cv,y_pred_cv)*100     

##Predicting the values of test set from Random Forest Classification Algorithm
y_pred_test = classifier.predict(X_test)

##Validating the Hypothesis on the Test Set
ones = np.zeros(len(y_test))
zeros = np.zeros(len(y_test))
for tree in range(num_tree):
    tree_op=classifier.estimators_[tree].predict(X_test)
    for i in range(len(tree_op)):
        if (tree_op[i] == 1):
            ones[i] = ones[i]+ tree_vote[tree]
        else:
            zeros[i] = zeros[i] + tree_vote[tree]


y_pred_newhypo =[]           
for i in range(len(y_test)):
    if (ones[i] > zeros[i]):
        y_pred_newhypo.append(1)
    elif (ones[i] < zeros[i]):
        y_pred_newhypo.append(0)
    else:
        y_pred_newhypo.append(2)
print(y_pred_newhypo)
y_pred_mod=np.asarray(y_pred_newhypo,dtype=np.int64)  

###Calculate the performance of regular Random Forest Algorithm
cm1 = confusion_matrix(y_test, y_pred_test)
ac1=accuracy_score(y_test, y_pred_test)*100 
           
###Calculating the Performance of new Hypothesis proposed
cm2 = confusion_matrix(y_test, y_pred_mod)
ac2=accuracy_score(y_test, y_pred_mod)*100 
   
    
   
