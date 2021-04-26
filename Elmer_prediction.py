#This Program is used for prediction using the saved pickle file
"""
Created on Mon Sep 21 2020
@author: rraushan
"""
import pandas as pd
import numpy as np
import pickle
import pyodbc
import sys
from codecs import decode
import datetime
import traceback

def prediction():
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
        dataset=pd.read_sql_query("select * from [ELMER_ML_DATA]",conn)
        dataset.fillna(value = np.nan,inplace=True)
        X = dataset.iloc[:,np.r_[0:1,2:7,8:21,24:25]].values   
        X_test = np.array([x.lower() if isinstance(x,str) else x for x in X ])
        
        ##Saving WR# before deleting dataset
        WR = dataset.iloc[:,[1,18]].values
        ##To free memory
        del(dataset) 
        del(X)
        
        ##Reading the Pickle file
        preprocessor = pickle.load(open(preprocessor_path,'rb'))
        standscalar = pickle.load(open(standardscalar_path,'rb'))
        classifier = pickle.load(open(classifier_path,'rb'))
        
        ##Transforming and Standard Scaling 
        X_test = preprocessor.transform(X_test)
        X_test = standscalar.transform(X_test)
        
        ##Prediction
        y_pred = classifier.predict_proba(X_test)
        
        ##Setting threshold to 0.01 and getting Pass Fail prediction
        Predicted= np.array(['Fail']*len(y_pred))
        for i in range(len(Predicted)):
            if y_pred[i,0] <= 0.01 :    
                Predicted[i] = 'Pass'
        
        ##Reporting the Result
        Result =pd.DataFrame(np.c_[WR,Predicted])
        Result.columns = ["WORKREQUEST_NUMBER","TIME_LOGGED_BY","ML_RESULT"]
        cursor = conn.cursor()
        for index, row in Result.iterrows():
            cursor.execute("UPDATE [ELMER_ML_DATA]  SET ML_RESULT = ?  WHERE WORKREQUEST_NUMBER = ? AND TIME_LOGGED_BY = ?", (row['ML_RESULT'], row['WORKREQUEST_NUMBER'],row['TIME_LOGGED_BY']))
        conn.commit() 
        cursor.close()
        conn.close()
    except Exception as e:
        ##Creating a log file and capturing exception
        date_time= datetime.datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
        filename = logfile_path+date_time+'.txt'
        f= open(filename,"w+")
        f.write('Exception:' + str(e) +"\n")
        f.write(traceback.format_exc())
        f.close()   
   
if __name__ == "__main__":
    prediction()
#############################Sample usage###############################
#python ./Elmer_prediction.py USMDCENDB70655,1113 Elmer_New C:\\RPA_Local\\pickle\\Elmer_preprocessor.pickle C:\\RPA_Local\\pickle\\Elmer_standardscalar.pickle C:\\RPA_Local\\pickle\\Elmer_classifier.pickle C:\\RPA_Local\\logs\\Elmer_prediction_logs_