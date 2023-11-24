##This program launches an html file on hitting a url and takes user input in the form of Component and Headline.It runs a querry on 
### third party tool cdets and get all the data from it which matches the query. Natural Language processing is done on this data
###followed by Neural Network using ANN
from __future__ import absolute_import
from __future__ import print_function
from flask import Flask, render_template,request
import sys
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
#import gensim
import spacy
#from utils import generate_glove
#from gensim.utils import tokenize
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge,Concatenate,concatenate,add,Add
from keras import backend as K
from sklearn.preprocessing import normalize
from keras.models import load_model
from keras.optimizers import RMSprop, SGD, Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge,concatenate,Concatenate
from keras.layers.advanced_activations import LeakyReLU
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('myfile.html')

@app.route('/',methods = ['POST', 'GET'])
def duplicate():
   if request.method == 'POST':
        component=request.form['Component']
        search_str=request.form['Headline']
        
        #Getting the Cdets query
        #component= 'ie-switch-hsr'
        try:
            os.environ['CDETS_INSTALL_DIR'] = 'C:\cdets'
            os.system('findcr -p CSC.labtrunk -s AFHOMWRCD -w Identifier,Headline \"Product = \'%s\'\" >> tmp.tsv'%component)
        except:
            print('Unable to return result for requested query')
            
        #dataframe = pd.read_csv("cdets_duplicatebugs.csv")
        dataframe = pd.read_csv("tmp.tsv",header=None,delimiter='\t')
        #search_str='first interface of hsr ring 1 is down,still its ingress statistics'
        if os.path.exists('tmp.tsv'):
            os.remove('tmp.tsv') 
        else:
            pass
        df = pd.DataFrame()
        df['question1'] = ''
        df['question2'] = dataframe.iloc[:,0].values
        df['is_duplicate'] = ''
        df['question1'] = search_str
        # encode questions to unicode
        df['question1'] = df['question1'].apply(lambda x: str(x))
        df['question2'] = df['question2'].apply(lambda x: str(x))
        ##############################################################################
        # TFIDF
        ##############################################################################
        
        # merge texts
        questions = list(df['question1']) + list(df['question2'])
        tfidf = TfidfVectorizer(lowercase=False, ) #lowercase=False, doesn`t convert all to lowercase
        tfidf.fit_transform(questions)
        
        # dict key:word and value:tf-idf score
        word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
        del questions
        ##############################################################################
        # WORD2VEC
        ##############################################################################
        if os.path.exists('2_cdets_word2vec_tfidf.pkl'):
            df = pd.read_pickle('2_cdets_word2vec_tfidf.pkl')
        else:
            # exctract word2vec vectors
            nlp = spacy.load('en')
            vecs1 = []
            for qu in tqdm(list(df['question1'])):
                doc = nlp(qu) 
                mean_vec = np.zeros([len(doc), 300])
                for word in doc:
                    # word2vec
                    vec = word.vector
                    # fetch df score
                    try:
                        idf = word2tfidf[str(word)] #Access Tf-idf score of the word from dictionary.Provide key to get value 
                    except:
                        #print word
                        idf = 0
                    # compute final vec
                    mean_vec += vec * idf
                mean_vec = mean_vec.mean(axis=0)
                vecs1.append(mean_vec)
            df['q1_feats'] = list(vecs1)
            
            vecs2 = []
            for qu in tqdm(list(df['question2'])):
                doc = nlp(qu) 
                mean_vec = np.zeros([len(doc), 300])
                for word in doc:
                    # word2vec
                    vec = word.vector
                    # fetch df score
                    try:
                        idf = word2tfidf[str(word)]
                    except:
                        print(word)
                        idf = 0
                    # compute final vec
                    mean_vec += vec * idf
                mean_vec = mean_vec.mean(axis=0)
                vecs2.append(mean_vec)
            df['q2_feats'] = list(vecs2)
        
            # save features
            #pd.to_pickle(df, 'data/2_word2vec_tfidf_cdets.pkl')
        
        ##############################################################################
        # CREATE TRAIN DATA
        ##############################################################################
        # shuffle df
        #df = df.reindex(np.random.permutation(df.index))
        
        
        # set number of train and test instances
        #num_train = int(df.shape[0] * 0.88)
        #num_test = df.shape[0] - num_train      
        num_test = int(df.shape[0])           
        #print("Number of training pairs: %i"%(num_train))
        #print("Number of testing pairs: %i"%(num_test))
        print(num_test)
        # init data data arrays
        #X_train = np.zeros([num_train, 2, 300]) #2 rows 300 columns
        X_test  = np.zeros([num_test, 2, 300])
        #Y_train = np.zeros([num_train]) 
        Y_test = np.zeros([num_test])
        
        # format data 
        b = [a[None,:] for a in list(df['q1_feats'].values)]
        q1_feats = np.concatenate(b, axis=0)
        
        b = [a[None,:] for a in list(df['q2_feats'].values)]
        q2_feats = np.concatenate(b, axis=0)
        
        # fill data arrays with features
        #X_train[:,0,:] = q1_feats[:num_train]
        #X_train[:,1,:] = q2_feats[:num_train]
        #Y_train = df[:num_train]['is_duplicate'].values
        X_test[:,0,:] = q1_feats
        X_test[:,1,:] = q2_feats
        Y_test = df[num_test:]['is_duplicate'].values
        
        # remove useless variables
        del b
        del q1_feats
        del q2_feats
        
        def euclidean_distance(vects):
            x, y = vects
            return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
        
        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)
        
        def cosine_distance(vests):
            x, y = vests
            x = K.l2_normalize(x, axis=-1)
            y = K.l2_normalize(y, axis=-1)
            return -K.mean(x * y, axis=-1, keepdims=True)
        
        def cos_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0],1)
        
        def contrastive_loss(y_true, y_pred):
            margin = 1
            return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
#y_true,(1-y_true) terms are exchanged since similar values are represented by y=1,dissimilar b y=0.        
        
        def create_base_network(input_dim):
            '''
            Base network for feature extraction.
            '''
            input = Input(shape=(input_dim, ))
            dense1 = Dense(200)(input)
            bn1 = BatchNormalization()(dense1)
            relu1 = Activation('elu')(bn1)
        
            dense2 = Dense(200)(relu1)
            bn2 = BatchNormalization()(dense2)
            res2 = Dense(200)(bn2)
            relu2 = Activation('elu')(res2)  
              
        
            dense3 = Dense(200)(relu2)
            bn3 = BatchNormalization()(dense3)
            res3 = Dense(200)(bn3)
            relu3 = Activation('relu')(res3) 
           
            bn4 = BatchNormalization()(relu3)
        
            model = Model(input=input, output=bn4)
        
            return model
        
        
        def compute_accuracy(predictions, labels):
            '''
            Compute classification accuracy with a fixed threshold on distances.
            '''
            #return labels[predictions.ravel() < 0.5].mean()
            return np.mean(np.equal(predictions.ravel() < 0.5, labels))
        
        def create_network(input_dim):
            # network definition
            base_network = create_base_network(input_dim)
            
            input_a = Input(shape=(input_dim,))
            input_b = Input(shape=(input_dim,))
            
            # because we re-use the same instance `base_network`,
            # the weights of the network
            # will be shared across the two branches
            processed_a = base_network(input_a)
            processed_b = base_network(input_b)
            
            distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
            
            model = Model(input=[input_a, input_b], output=distance)
            return model
        #Normalizing is degrading the performance
        '''
        X_train_norm = np.zeros_like(X_train)
        X_train_norm[:,0,:] = normalize(X_train[:,0,:], axis=0)
        X_train_norm[:,1,:] = normalize(X_train[:,1,:], axis=0)
        X_test_norm = np.zeros_like(X_test)
        X_test_norm[:,0,:] = normalize(X_test[:,0,:], axis=0)
        X_test_norm[:,1,:] = normalize(X_test[:,1,:], axis=0)
        '''
        #==============================================================================
        # # create model
        # net = create_network(300)
        # 
        # # train
        # #optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
        # optimizer = Adam(lr=0.001)  #lr=Learning Rate
        # net.compile(loss=contrastive_loss, optimizer=optimizer)
        # 
        # #for epoch in range(50):
        # net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
        #           validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
        #           batch_size=128, nb_epoch=50, shuffle=True, )
        #     
        #     # compute final accuracy on training and test sets
        # pred = net.predict([X_test[:,0,:], X_test[:,1,:]], batch_size=128)
        # te_acc = compute_accuracy(pred, Y_test)
        # 
        # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
        # net.save('my_model.h5')
        # del net
        #==============================================================================
        model = load_model('cdets.h5',custom_objects={'contrastive_loss' : contrastive_loss})
        #model = load_model('cdets_lower.h5',custom_objects={'contrastive_loss' : contrastive_loss})
        df['is_duplicate'] = model.predict([X_test[:,0,:], X_test[:,1,:]], batch_size=128)
        df.sort_values(by = 'is_duplicate' ,inplace = True, ascending=True)
        op = [df.iloc[i,1] for i in range(num_test) if df.iloc[i,2] <= 0.5]
        #op = list(df.iloc[0:5,2].values)
        return render_template("myfile.html", output=op) #replace result with 5 similar bugs
        del op
        del df
        del dataframe
        del model
        #te_acc = compute_accuracy(pred, Y_test)
            
        #    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        #print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
        

if __name__=='__main__':
    app.run(debug=True)
