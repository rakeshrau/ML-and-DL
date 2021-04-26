#This Program is used for generating sentiments-Negative,Positve,Neutral and also the scores 
"""
Created on Mon Nov 11 2020
@author: rraushan
"""
import pandas as pd
import numpy as np
import pyodbc
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
from codecs import decode
import datetime
import traceback

def sentiments():
    try:
        ##Database connection using pyODBC
        dbdriver = 'SQL Server Native Client 11.0'
        dbserver = decode(sys.argv[1], 'unicode_escape') #Name of Database server
        dbname = decode(sys.argv[2], 'unicode_escape') #Name of the Database
        logfile_path = decode(sys.argv[3], 'unicode_escape') #Path of log file
        connection_string = 'Driver={}; Server={};Database={};Trusted_Connection=yes;'.format(dbdriver, dbserver, dbname)
        conn=pyodbc.connect(connection_string)
        conn.setencoding('utf-8')
        ########################Getting Feedback data
        df=pd.read_sql_query("select * from ELMER_ML_VIEW_SENTIMENT_ANALYSIS",conn)
        df.fillna(value = np.nan,inplace=True)
        
        ########################Getting polarity scores
        y  =  [None]*len(df)
        analyzer = SentimentIntensityAnalyzer()
        for i in range(len(df)):
            y[i] = analyzer.polarity_scores(df['FEEDBACK_COMMENT'][i]) 
        
         
        positive  = pd.DataFrame(y)['pos']
        negative = pd.DataFrame(y)['neg']
        neutral = pd.DataFrame(y)['neu']
        compound = pd.DataFrame(y)['compound']
        
        WR = df.iloc[:,0].values
        sentiment = np.array(['sentiment']*len(df))
        feedback_comment = df.iloc[:,2]
        feedback_rating = df.iloc[:,1]
        
        Keyword_neutral = ['additionalwork','onshorenotes']
        Keyword_positive = ['appreciative comments']
        Keyword_negative =['incomplete','additionalwork','onshorenotes','rework','comment']
        
        ########################Classifying the sentiment
        for i in range(len(feedback_rating)):
            for element in Keyword_positive:
                    if(element in feedback_comment[i].lower()):
                        sentiment[i]='Positive'
                        continue
            if (y[i]['pos'] > 0.75 and y[i]['neg']== 0.0):
                sentiment[i]='Positive'
                      
        for i in range(len(sentiment)):
            if (sentiment[i] != 'Positive' and y[i]['neg'] > 0.0):
                sentiment[i]='Negative'
            elif (sentiment[i] != 'Positive' and sentiment[i]!='Negative'):
                sentiment[i]='Neutral'
        
        for i in range(len(feedback_rating)):
            for j in range(len(Keyword_negative)):
                if ((sentiment[i]=='Negative') and (feedback_rating[i].find(Keyword_negative[j])!=-1)):
                    sentiment[i]='Neutral'
                           
        ########################Generating the Result with scores and sentiment                
        Result_sentiment =pd.DataFrame(np.c_[WR,positive,negative,neutral,compound,feedback_rating,feedback_comment,sentiment])
        Result_sentiment.columns = ['WORKREQUEST_NUMBER','POSITIVE_PROBABILITY','NEGATIVE_PROBABILITY','NEUTRAL_PROBABILITY','COMPOUND_PROBABILITY','FEEDBACK_RATING','FEEDBACK_COMMENT','SENTIMENT']
        #Result_sentiment.to_excel("sentiment_result.xlsx")   #To get Excel file locally    
                      
        cursor = conn.cursor()
        for index, row in Result_sentiment.iterrows():
            cursor.execute("UPDATE [ELMER_ML_VIEW_SENTIMENT_ANALYSIS]  SET SENTIMENT = ?,POSITIVE_PROBABILITY = ?,NEGATIVE_PROBABILITY = ?,NEUTRAL_PROBABILITY = ?,COMPOUND_PROBABILITY = ? WHERE WORKREQUEST_NUMBER = ?", (row['SENTIMENT'], row['POSITIVE_PROBABILITY'],row['NEGATIVE_PROBABILITY'],row['NEUTRAL_PROBABILITY'],row['COMPOUND_PROBABILITY'],row['WORKREQUEST_NUMBER']))     
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
    sentiments()
#############################Sample usage###############################
#python ./Elmer_sentiment.py "USMDCENDB70655,1113" "Elmer_New" "C:\\RPA_Local\\logs\\Elmer_sentiment_logs_"


