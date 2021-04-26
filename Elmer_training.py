#This Program is used for training the Elmer model and saving the pickle file
"""
Created on Mon Sep 21 2020
@author: rraushan
"""
import pandas as pd
import numpy as np
import pickle
import pyodbc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import sys
from codecs import decode
import datetime
import traceback

def training():

    try:
        ##Database connection using pyODBC
        dbdriver = 'SQL Server Native Client 11.0'
        dbserver = decode(sys.argv[1], 'unicode_escape') #Name of Database server
        dbname = decode(sys.argv[2], 'unicode_escape') #Name of the Database
        preprocessor_path = decode(sys.argv[3], 'unicode_escape') #Path of Elmer_preprocessor.pickle
        standardscalar_path = decode(sys.argv[4], 'unicode_escape') #Path of Elmer_standardscalr.pickle
        classifier_path = decode(sys.argv[5], 'unicode_escape') #Path of Elmer_classifier.pickle
        logfile_path = decode(sys.argv[6], 'unicode_escape') #Path of log file
        connection_string = 'Driver={}; Server={};Database={};Trusted_Connection=yes;'.format(dbdriver, dbserver, dbname)
        conn=pyodbc.connect(connection_string)
        conn.setencoding('utf-8')

        ##Reading the data from the Database to Pandas dataframe
        dataset=pd.read_sql_query("select * from [WR_QRM_TRAINING_DATA]",conn)
        dataset.fillna(value = np.nan,inplace=True) 
        X = dataset.iloc[:,np.r_[0:1,2:7,8:21,24:25]].values   
        Y = dataset.iloc[:,25].values
        X_train = np.array([x.lower() if isinstance(x,str) else x for x in X ])
        y_train = np.array([y.lower() if isinstance(y,str) else y for y in Y ])
        
        ##Closing Database connection
        conn.close()
        
        ##To free memory
        del(dataset) 
        del(X) 
        del(Y)
        
        ##Encoding 
        cat_cols= [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, cat_cols)],remainder='passthrough')
        
        X_train = preprocessor.fit_transform(X_train)
        
        ##Feature Scaling Training Set
        standscalar =StandardScaler(with_mean=False)
        X_train = standscalar.fit_transform(X_train)    
        
        ##LabelEncoding Target variable
        labelencoder_y = LabelEncoder()
        y_train = labelencoder_y.fit_transform(y_train.astype(str)) 

        ##Training the model on XGBoost
        classifier = XGBClassifier(n_estimators=50000,objective='binary:logistic',random_state=0,n_jobs=-1)
        classifier.fit(X_train,y_train)
        ##Pickling the model and variables
        pickle.dump(classifier,open(classifier_path,'wb'))
        pickle.dump(standscalar,open(standardscalar_path,'wb'))
        pickle.dump(preprocessor,open(preprocessor_path,'wb'))
        
    except Exception as e:
        ##Creating a log file and capturing exception
        date_time= datetime.datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
        filename = logfile_path+date_time+'.txt'
        f= open(filename,"w+")
        f.write('Exception:' + str(e) +"\n")
        f.write(traceback.format_exc())
        f.close()

if __name__ == "__main__":
    training()
#############################Sample usage###############################
#python ./Elmer_training.py USMDCENDB70655,1113 Elmer_New C:\\RPA_Local\\pickle\\Elmer_preprocessor.pickle C:\\RPA_Local\\pickle\\Elmer_standardscalar.pickle C:\\RPA_Local\\pickle\\Elmer_classifier.pickle C:\\RPA_Local\\logs\\Elmer_training_logs_
