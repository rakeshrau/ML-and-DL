import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_excel('Sentiment1.xlsx')
df.replace(r'\s+', np.nan, regex=True)

X= df.iloc[:,0].str.encode('utf-8')
#w= [dict(wlb=0,pi=0,wfh=0,bs=0,mi=0,ds=0)]*(X.size)
k= pd.DataFrame(index=range(X.size),columns=['Work_Life','Payment','WFH','Bias','Manager','Discrimination'])
y=[]
z=[]
Work_Life =['working hours','long hours','flexibility','work-life','work life','Work-Life','Work life','Work-life']
Payment = ['payment','overtime','OT']
WFH =['Remote working','work from home','work-from-home']
Bias =['bias','favoritism','favouristism','favorite','favourite','favour','favor']
Manager= ['manager','AD','director']
Discrimination = ['discrimination','discriminate']
NA = []
cnt1=cnt2=cnt3=cnt4=cnt5=cnt6=cnt7=cnt8=cnt9=0
analyzer = SentimentIntensityAnalyzer()
for i in range(X.size):
    y.append(analyzer.polarity_scores(X[i])) #Getting the scores 
    
    for ele2 in Work_Life:
        if(ele2 in X[i].lower()):
            k.loc[i,'Work_Life']='found'
            
        
    for ele2 in Payment:
        if(ele2 in X[i].lower()):
            k.loc[i,'Payment']='found'
            
            
    
    for ele2 in WFH:
        if(ele2 in X[i].lower()):
            k.loc[i,'WFH']='found'
            
            
    for ele2 in Bias:
        if(ele2 in X[i].lower()):
            k.loc[i,'Bias']='found'
            
        
    for ele2 in Manager:
        if(ele2 in X[i].lower()):
            k.loc[i,'Manager']='found'
            
            
    for ele2 in Discrimination:
        if(ele2 in X[i].lower()):
            k.loc[i,'Discrimination']='found'
           
##Deciding on Sentiment by setting threshhold    
for i in range(len(y)):
    if (y[i]['neg'] > 0.0) :
        z.append('Negative Sentiment')
        
    elif (y[i]['compound'] >= 0.35):
        z.append('Positive Sentiment')
        
    else:
        z.append('Neutral Sentiment')
        

####Plotting Grpah
import seaborn as sns

plot1= pd.read_excel('Sentiment1.xlsx')
sentiments= ['Positive','Negative','Neutral']
key_metrics=['WorkLife','Payment Issue','WFH','Bias','Mgr Issues','Discrimination']

for i in range(X.size):
    if (plot1.iloc[i,2] == 'found'):
        cnt1+=1
    if (plot1.iloc[i,3] == 'found'):
        cnt2+=1
    if (plot1.iloc[i,4] == 'found'):
        cnt3+=1
    if (plot1.iloc[i,5] == 'found'):
        cnt4+=1
    if (plot1.iloc[i,6] == 'found'):
        cnt5+=1
    if (plot1.iloc[i,7] == 'found'):
        cnt6+=1
    if (plot1.iloc[i,1] == 'Positive Sentiment'):
        cnt7+=1
    if (plot1.iloc[i,1] == 'Negative Sentiment'):
        cnt8+=1
    if (plot1.iloc[i,1] == 'Neutral Sentiment'):
        cnt9+=1
        
key_values=[cnt1,cnt2,cnt3,cnt4,cnt5,cnt6]
key_values_sent=[cnt7,cnt8,cnt9]
df2=pd.DataFrame(dict(Key_Metrics=key_metrics, Occurences=key_values))
df3=pd.DataFrame(dict(Sentiment=sentiments, Occurences=key_values_sent))
sns.set(style="whitegrid")

ax1 = sns.barplot(x="Key_Metrics", y="Occurences",hue="Key_Metrics", data=df2)
ax2 = sns.barplot(x="Sentiment", y="Occurences",hue="Sentiment", data=df3)      
    

