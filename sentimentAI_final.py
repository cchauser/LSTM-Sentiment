import nltk
from nltk.corpus import stopwords
import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
from theano.tensor.opt import register_canonicalize
import time
import operator
import shelve
import itertools
import os
import math
import random
from copy import deepcopy

unknown_token = 'UNKNOWN_TOKEN'

class reviewAI:
    def __init__(self, word_dim, hidden_dim = 72, bptt_truncate = 6, num_Classes = 2):
        self.vocab_limit = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.num_Classes = num_Classes
        
        self.w2i = {}

        self.test_name = input("Model name: ").strip()
        try:
            #Perhaps I'll add a way to input your own string and have it spit out the classification in the future.
            #For now I'll just give the ability to load old models.
            db = shelve.open("{0}/{0}DB".format(self.test_name), "r")
            self.E = db['E'] 
            self.V = db['V']
            self.d = db['d']
            self.U_in = db['U_in']
            self.U_out = db['U_out']
            self.U_forget = db['U_forget']
            self.U_cand = db['U_cand']
            self.W_in = db['W_in']
            self.W_out = db['W_out']
            self.W_forget = db['W_forget']
            self.W_cand = db['W_cand']
            self.b_in = db['b_in']
            self.b_out = db['b_out']
            self.b_forget = db['b_forget']
            self.b_cand = db['b_cand']
            self.U1_in = db['U1_in']
            self.U1_out = db['U1_out']
            self.U1_forget = db['U1_forget']
            self.U1_cand = db['U1_cand']
            self.W1_in = db['W1_in']
            self.W1_out = db['W1_out']
            self.W1_forget = db['W1_forget']
            self.W1_cand = db['W1_cand']
            self.b1_in = db['b1_in']
            self.b1_out = db['b1_out']
            self.b1_forget = db['b1_forget']
            self.b1_cand = db['b1_cand']
            self.w2i = db['w2i']
            self.i2w = db['i2w']
            self.hidden_dim = self.E.get_value().shape[0]
            db.close()
            print("Model loaded")
        except:
            print("Creating new model...")
            self.fileN = input("File name: ").strip()
            print("\nLoading voabulary...")
            self.build_Vocab(self.fileN)
            try:
                os.makedirs('{}'.format(self.test_name))
            except:
                pass
            db = shelve.open("{0}/{0}DB".format(self.test_name), "c")
            db.close()

            #Universal
            E = np.random.uniform(-np.sqrt(1./self.vocab_limit), np.sqrt(1./self.vocab_limit), (self.hidden_dim, self.vocab_limit))
            V = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.num_Classes))
            d = np.zeros(self.num_Classes)
            
            #Layer 1
            U_in = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            U_forget = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            U_out = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            U_cand = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            W_in = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            W_forget = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            W_out = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            W_cand = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            b_in = np.zeros((self.hidden_dim))
            b_forget = np.zeros((self.hidden_dim))
            b_out = np.zeros((self.hidden_dim))
            b_cand = np.zeros((self.hidden_dim))

            #Layer 2
            U1_in = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            U1_forget = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            U1_out = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            U1_cand = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            W1_in = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            W1_forget = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            W1_out = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            W1_cand = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            b1_in = np.zeros((self.hidden_dim))
            b1_forget = np.zeros((self.hidden_dim))
            b1_out = np.zeros((self.hidden_dim))
            b1_cand = np.zeros((self.hidden_dim))
            
            #Theano Shared Variables
            #Universal
            self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
            self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
            self.d = theano.shared(name='d', value=d.astype(theano.config.floatX))
            
            #Layer 1
            self.U_in = theano.shared(name='U_in', value=U_in.astype(theano.config.floatX))
            self.U_forget = theano.shared(name='U', value=U_forget.astype(theano.config.floatX))
            self.U_out = theano.shared(name='U_out', value=U_out.astype(theano.config.floatX))
            self.U_cand = theano.shared(name='U_cand', value=U_cand.astype(theano.config.floatX))
            self.W_in = theano.shared(name='W_in', value=W_in.astype(theano.config.floatX))
            self.W_forget = theano.shared(name='W_forget', value=W_forget.astype(theano.config.floatX))
            self.W_out = theano.shared(name='W_out', value=W_out.astype(theano.config.floatX))
            self.W_cand = theano.shared(name='W_cand', value=W_cand.astype(theano.config.floatX))
            self.b_in = theano.shared(name='b_in', value=b_in.astype(theano.config.floatX))
            self.b_forget = theano.shared(name='b_forget', value=b_forget.astype(theano.config.floatX))
            self.b_out = theano.shared(name='b_out', value=b_out.astype(theano.config.floatX))
            self.b_cand = theano.shared(name='b_cand', value=b_cand.astype(theano.config.floatX))

            #Layer 2
            self.U1_in = theano.shared(name='U1_in', value=U1_in.astype(theano.config.floatX))
            self.U1_forget = theano.shared(name='U1_forget', value=U1_forget.astype(theano.config.floatX))
            self.U1_out = theano.shared(name='U1_out', value=U1_out.astype(theano.config.floatX))
            self.U1_cand = theano.shared(name='U1_cand', value=U1_cand.astype(theano.config.floatX))
            self.W1_in = theano.shared(name='W1_in', value=W1_in.astype(theano.config.floatX))
            self.W1_forget = theano.shared(name='W1_forget', value=W1_forget.astype(theano.config.floatX))
            self.W1_out = theano.shared(name='W1_out', value=W1_out.astype(theano.config.floatX))
            self.W1_cand = theano.shared(name='W1_cand', value=W1_cand.astype(theano.config.floatX))
            self.b1_in = theano.shared(name='b1_in', value=b1_in.astype(theano.config.floatX))
            self.b1_forget = theano.shared(name='b1_forget', value=b1_forget.astype(theano.config.floatX))
            self.b1_out = theano.shared(name='b1_out', value=b1_out.astype(theano.config.floatX))
            self.b1_cand = theano.shared(name='b1_cand', value=b1_cand.astype(theano.config.floatX))
        finally:
            #Derivatives
            #Universal
            self.mE = theano.shared(name='mE', value=np.zeros(self.E.get_value().shape).astype(theano.config.floatX))
            self.mV = theano.shared(name='mV', value=np.zeros(self.V.get_value().shape).astype(theano.config.floatX))
            self.vE = theano.shared(name='vE', value=np.zeros(self.E.get_value().shape).astype(theano.config.floatX))
            self.vV = theano.shared(name='vV', value=np.zeros(self.V.get_value().shape).astype(theano.config.floatX))
            self.md = theano.shared(name='md', value=np.zeros(self.d.get_value().shape).astype(theano.config.floatX))
            self.vd = theano.shared(name='vd', value=np.zeros(self.d.get_value().shape).astype(theano.config.floatX))
            
            #Layer 1
            self.mU_in = theano.shared(name='mU_in', value=np.zeros(self.U_in.get_value().shape).astype(theano.config.floatX))
            self.mU_forget = theano.shared(name='mU_forget', value=np.zeros(self.U_forget.get_value().shape).astype(theano.config.floatX))
            self.mU_out = theano.shared(name='mU_out', value=np.zeros(self.U_out.get_value().shape).astype(theano.config.floatX))
            self.mU_cand = theano.shared(name='mU_cand', value=np.zeros(self.U_cand.get_value().shape).astype(theano.config.floatX))
            self.mW_in = theano.shared(name='mW_in', value=np.zeros(self.W_in.get_value().shape).astype(theano.config.floatX))
            self.mW_forget = theano.shared(name='mW_forget', value=np.zeros(self.W_forget.get_value().shape).astype(theano.config.floatX))
            self.mW_out = theano.shared(name='mW_out', value=np.zeros(self.W_out.get_value().shape).astype(theano.config.floatX))
            self.mW_cand = theano.shared(name='mW_cand', value=np.zeros(self.W_cand.get_value().shape).astype(theano.config.floatX))
            self.mb_in = theano.shared(name='mb_in', value=np.zeros(self.b_in.get_value().shape).astype(theano.config.floatX))
            self.mb_forget = theano.shared(name='mb_forget', value=np.zeros(self.b_forget.get_value().shape).astype(theano.config.floatX))
            self.mb_out = theano.shared(name='mb_out', value=np.zeros(self.b_out.get_value().shape).astype(theano.config.floatX))
            self.mb_cand = theano.shared(name='mb_cand', value=np.zeros(self.b_cand.get_value().shape).astype(theano.config.floatX))
            self.vU_in = theano.shared(name='vU_in', value=np.zeros(self.U_in.get_value().shape).astype(theano.config.floatX))
            self.vU_forget = theano.shared(name='vU_forget', value=np.zeros(self.U_forget.get_value().shape).astype(theano.config.floatX))
            self.vU_out = theano.shared(name='vU_out', value=np.zeros(self.U_out.get_value().shape).astype(theano.config.floatX))
            self.vU_cand = theano.shared(name='vU_cand', value=np.zeros(self.U_cand.get_value().shape).astype(theano.config.floatX))
            self.vW_in = theano.shared(name='vW_in', value=np.zeros(self.W_in.get_value().shape).astype(theano.config.floatX))
            self.vW_forget = theano.shared(name='vW_forget', value=np.zeros(self.W_forget.get_value().shape).astype(theano.config.floatX))
            self.vW_out = theano.shared(name='vW_out', value=np.zeros(self.W_out.get_value().shape).astype(theano.config.floatX))
            self.vW_cand = theano.shared(name='vW_cand', value=np.zeros(self.W_cand.get_value().shape).astype(theano.config.floatX))
            self.vb_in = theano.shared(name='vb_in', value=np.zeros(self.b_in.get_value().shape).astype(theano.config.floatX))
            self.vb_forget = theano.shared(name='vb_forget', value=np.zeros(self.b_forget.get_value().shape).astype(theano.config.floatX))
            self.vb_out = theano.shared(name='vb_out', value=np.zeros(self.b_out.get_value().shape).astype(theano.config.floatX))
            self.vb_cand = theano.shared(name='vb_cand', value=np.zeros(self.b_cand.get_value().shape).astype(theano.config.floatX))

            #Layer 2
            self.mU1_in = theano.shared(name='mU1_in', value=np.zeros(self.U_in.get_value().shape).astype(theano.config.floatX))
            self.mU1_forget = theano.shared(name='mU1_forget', value=np.zeros(self.U_forget.get_value().shape).astype(theano.config.floatX))
            self.mU1_out = theano.shared(name='mU1_out', value=np.zeros(self.U_out.get_value().shape).astype(theano.config.floatX))
            self.mU1_cand = theano.shared(name='mU1_cand', value=np.zeros(self.U_cand.get_value().shape).astype(theano.config.floatX))
            self.mW1_in = theano.shared(name='mW1_in', value=np.zeros(self.W_in.get_value().shape).astype(theano.config.floatX))
            self.mW1_forget = theano.shared(name='mW1_forget', value=np.zeros(self.W_forget.get_value().shape).astype(theano.config.floatX))
            self.mW1_out = theano.shared(name='mW1_out', value=np.zeros(self.W_out.get_value().shape).astype(theano.config.floatX))
            self.mW1_cand = theano.shared(name='mW1_cand', value=np.zeros(self.W_cand.get_value().shape).astype(theano.config.floatX))
            self.mb1_in = theano.shared(name='mb1_in', value=np.zeros(self.b_in.get_value().shape).astype(theano.config.floatX))
            self.mb1_forget = theano.shared(name='mb1_forget', value=np.zeros(self.b_forget.get_value().shape).astype(theano.config.floatX))
            self.mb1_out = theano.shared(name='mb1_out', value=np.zeros(self.b_out.get_value().shape).astype(theano.config.floatX))
            self.mb1_cand = theano.shared(name='mb1_cand', value=np.zeros(self.b_cand.get_value().shape).astype(theano.config.floatX))
            self.vU1_in = theano.shared(name='vU1_in', value=np.zeros(self.U_in.get_value().shape).astype(theano.config.floatX))
            self.vU1_forget = theano.shared(name='vU1_forget', value=np.zeros(self.U_forget.get_value().shape).astype(theano.config.floatX))
            self.vU1_out = theano.shared(name='vU1_out', value=np.zeros(self.U_out.get_value().shape).astype(theano.config.floatX))
            self.vU1_cand = theano.shared(name='vU1_cand', value=np.zeros(self.U_cand.get_value().shape).astype(theano.config.floatX))
            self.vW1_in = theano.shared(name='vW1_in', value=np.zeros(self.W_in.get_value().shape).astype(theano.config.floatX))
            self.vW1_forget = theano.shared(name='vW1_forget', value=np.zeros(self.W_forget.get_value().shape).astype(theano.config.floatX))
            self.vW1_out = theano.shared(name='vW1_out', value=np.zeros(self.W_out.get_value().shape).astype(theano.config.floatX))
            self.vW1_cand = theano.shared(name='vW1_cand', value=np.zeros(self.W_cand.get_value().shape).astype(theano.config.floatX))
            self.vb1_in = theano.shared(name='vb1_in', value=np.zeros(self.b_in.get_value().shape).astype(theano.config.floatX))
            self.vb1_forget = theano.shared(name='vb1_forget', value=np.zeros(self.b_forget.get_value().shape).astype(theano.config.floatX))
            self.vb1_out = theano.shared(name='vb1_out', value=np.zeros(self.b_out.get_value().shape).astype(theano.config.floatX))
            self.vb1_cand = theano.shared(name='vb1_cand', value=np.zeros(self.b_cand.get_value().shape).astype(theano.config.floatX))


        self.theano = {}
        print("Loading functions...")
        try:
            self.__theano_build__()
            print("Done\nReady\n")
        except:
            print("Error loading functions")


    def build_Vocab(self, fileName):
        posts = []
        with open('{}.txt'.format(fileName), 'r') as f:
            reviews = f.readlines()
        x_data = []
        y_data = []
        for rev in reviews:
            rev_split = rev.split("|||")
            x_data.append(rev_split[0].lower().strip())
            y_data.append(rev_split[1])

        y_da = []
        x_da = []
        S = [0] * 2
        for i in range(len(y_data)):
            try:
                if int(y_data[i]) > 3 and int(y_data[i]) <= 5:
                    if(S[1] < S[0]):
                        y_da.append(1)
                        S[1] += 1
                        x_da.append(x_data[i])
                elif int(y_data[i]) <= 3:
                    y_da.append(0)
                    S[0] += 1
                    x_da.append(x_data[i])
            except:
                continue
          
        self.sample_Size = len(x_da)
        sep_Sent = itertools.chain(*[nltk.sent_tokenize(sent) for sent in x_da])
        word_Tok = [nltk.word_tokenize(sent) for sent in sep_Sent]
             
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*word_Tok))
        print("Found {} unique words tokens.".format(len(word_freq.items())))

        self.vocab_limit = min(int((len(word_freq.items()) * .15)), self.vocab_limit)
        stop_words = set(stopwords.words('english'))


        rem = []
        for i in range(len(x_da)):
            x_da[i] = nltk.word_tokenize(x_da[i])
            temp = [] 
            for w in x_da[i]: 
                    if w not in stop_words:
                        temp.append(w)

            if len(temp) >= 5:
                x_da[i] = temp
            else:
                rem.append(i)
                
        i = 0
        for n in range(len(rem)):
            x_da.remove(x_da[rem[n-i]])
            y_da.remove(y_da[rem[n-i]])
            i += 1
                
         
        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(self.vocab_limit-1)
        self.i2w = [x[0] for x in vocab]
        self.i2w.append(unknown_token)
        self.w2i = dict([(w,i) for i,w in enumerate(self.i2w)])

        print("Using a vocabulary size of {}.".format(len(self.w2i)))
        print("The least frequent word in the vocabulary is '{}', appearing {} times.".format(vocab[-1][0], vocab[-1][1]))

        for i in range(len(x_da)):
            for w in range(len(x_da[i])):
                try:
                    x_da[i][w] = self.w2i[x_da[i][w]]
                except:
                    x_da[i][w] = self.w2i[unknown_token]

        y_d = []
        x_d = []
        S = [0] * 2
        for i in range(len(y_da)):
            try:
                if((x_da[i].count(self.w2i[unknown_token])/len(x_da[i])) <= .5):
                    if(S[1] < S[0] and y_da[i] == 1):
                        y_d.append(1)
                        S[1] += 1
                        x_d.append(x_da[i])
                    elif(y_da[i] == 0):
                        y_d.append(0)
                        S[0] += 1
                        x_d.append(x_da[i])
            except:
                continue

        print(S)
        x_data = np.asarray(x_d)
        y_data = np.asarray(y_d)

        #Verification Data
        val_perc_of_data = .85
        self.x_val = x_data[len(x_data) * val_perc_of_data:]
        self.x_train = x_data[0:len(x_data) * val_perc_of_data]
        self.y_val = y_data[len(y_data) * val_perc_of_data:]
        self.y_train = y_data[0:len(y_data) * val_perc_of_data]

        print("\nSample size: {}".format(len(self.x_train)))
        print("Validation size: {}".format(len(self.x_val)))


    def _step(self, z, o_L2_Prev, o_L1_Prev, C_L1_Prev, C_L2_Prev):
        #This maps the value passed to the function to the word's
        #unique weighted matrix. All known words have a weighted matrix
        #contained in E
        x_e = self.E[:,z] #256,1

        """
        LAYER 1
        """
        #Gate calculations; these modulate how much of the previous layer's data is used
        #The sigmoid function squishes the values to between 0 and 1
        f_g_L1 = T.nnet.sigmoid(T.dot(self.W_forget, x_e) + T.dot(self.U_forget, o_L1_Prev) + self.b_forget) #256,1
        i_g_L1 = T.nnet.sigmoid(T.dot(self.W_in, x_e) + T.dot(self.U_in, o_L1_Prev) + self.b_in)
        o_g_L1 = T.nnet.sigmoid(T.dot(self.W_out, x_e) + T.dot(self.U_out, o_L1_Prev) + self.b_out)

        #Candidate value for the cell memory that runs along the neural network
        C_c_L1 = T.nnet.nnet.softsign(T.dot(self.W_cand, x_e) + T.dot(self.U_cand, o_L1_Prev) +  self.b_cand)#256,1
        
        #New cell value actual
        C_L1 = i_g_L1 * C_c_L1 + f_g_L1 * C_L1_Prev

        #Layer1 output
        o_L1 = o_g_L1 * T.nnet.nnet.softsign(C_L1)

        """
        LAYER 2
        """
        #Gate calculations; these modulate how much of the previous layer's data is used
        f_g_L2 = T.nnet.sigmoid(T.dot(self.W1_forget, o_L1) + T.dot(self.U1_forget, o_L2_Prev) + self.b1_forget) #256,1
        i_g_L2 = T.nnet.sigmoid(T.dot(self.W1_in, o_L1) + T.dot(self.U1_in, o_L2_Prev) + self.b1_in)
        o_g_L2 = T.nnet.sigmoid(T.dot(self.W1_out, o_L1) + T.dot(self.U1_out, o_L2_Prev) + self.b1_out)

        #Candidate value for the cell memory that runs along the neural network
        C_c_L2 = T.nnet.nnet.softsign(T.dot(self.W1_cand, o_L1) + T.dot(self.U1_cand, o_L2_Prev) +  self.b1_cand)#256,1
        
        #New cell value
        C_L2 = i_g_L2 * C_c_L2 + f_g_L2 * C_L2_Prev

        #Output value
        o_L2 = o_g_L2 * T.nnet.nnet.softsign(C_L2)

        return o_L2, o_L1, C_L1, C_L2
        
    def __theano_build__(self):

        x = T.ivector('x') #Input sequence stored as theano variable x
        y = T.vector('y') #Target output value stored as theano variable y
        learnRate = T.scalar('learnRate')
        t = T.scalar('t') #Time step
        
        print("Loading _step")
        [o, o_L1, C_L1, C_L2], updates = theano.scan(
            self._step,
            sequences=x,
            truncate_gradient = self.bptt_truncate,
            outputs_info=[theano.shared(value = np.zeros(self.hidden_dim).astype(theano.config.floatX)),
                          theano.shared(value = np.zeros(self.hidden_dim).astype(theano.config.floatX)),
                          theano.shared(value = np.ones(self.hidden_dim).astype(theano.config.floatX)),
                          theano.shared(value = np.ones(self.hidden_dim).astype(theano.config.floatX))])


        #Average of all of the hidden states. Shape (hidden_dim,)
        out_Pool = o.sum(axis = 0)/x.shape[0]
        
        self.debug_opool = theano.function([x], out_Pool)
        self.debug_o = theano.function([x], o)
        
        #Logistic regression on out_Pool
        #Returns a vector with .sum = 1 and whose contents describe
        #the probability that the input is of a given class
        pred_Prob = T.nnet.softmax(T.dot(out_Pool, self.V) + self.d)[0]#returns a matrix with one row. so just take that row

        #Predicts error of the output using categorical cross entropy
        pred_error = T.sum(T.nnet.categorical_crossentropy(pred_Prob, y))

        print("Loading f_pred_Prob")
        #Declare theano functions for predicting outcomes
        self.f_pred_Prob = theano.function([x], pred_Prob) #Returns the probability vector
        print("Loading f_pred")
        self.f_pred = theano.function([x], pred_Prob.argmax(axis = 0)) #Returns the predicted class

        #Define function for calculating error
        print("Loading ce_error")
        self.ce_error = theano.function([x, y], pred_error, allow_input_downcast=True) #Returns cross-entropy error

        print("Loading gradients")
        
      ###Gradients###
        print("-Loading derivatives")
        #Universal
        dE = T.grad(pred_error, self.E)
        dV = T.grad(pred_error, self.V)
        dd = T.grad(pred_error, self.d)

        #Layer 1
        print("--Layer1")
        dW_in = T.grad(pred_error, self.W_in)
        dW_out = T.grad(pred_error, self.W_out)
        dW_forget = T.grad(pred_error, self.W_forget)
        dW_cand = T.grad(pred_error, self.W_cand)
        dU_in = T.grad(pred_error, self.U_in)
        dU_out = T.grad(pred_error, self.U_out)
        dU_forget = T.grad(pred_error, self.U_forget)
        dU_cand = T.grad(pred_error, self.U_cand)
        db_in = T.grad(pred_error, self.b_in)
        db_out = T.grad(pred_error, self.b_out)
        db_forget = T.grad(pred_error, self.b_forget)
        db_cand = T.grad(pred_error, self.b_cand)

        #Layer 2
        print("--Layer2")
        dW1_in = T.grad(pred_error, self.W1_in)
        dW1_out = T.grad(pred_error, self.W1_out)
        dW1_forget = T.grad(pred_error, self.W1_forget)
        dW1_cand = T.grad(pred_error, self.W1_cand)
        dU1_in = T.grad(pred_error, self.U1_in)
        dU1_out = T.grad(pred_error, self.U1_out)
        dU1_forget = T.grad(pred_error, self.U1_forget)
        dU1_cand = T.grad(pred_error, self.U1_cand)
        db1_in = T.grad(pred_error, self.b1_in)
        db1_out = T.grad(pred_error, self.b1_out)
        db1_forget = T.grad(pred_error, self.b1_forget)
        db1_cand = T.grad(pred_error, self.b1_cand)

     ###Adam cache updates###
        beta1 = .9
        beta2 = .999
        eps = 1e-8
        print("-Loading cache updates")
        #Universal
        mE = beta1 * self.mE + (1 - beta1) * dE
        vE = beta2 * self.vE + (1 - beta2) * (dE ** 2)
        mV = beta1 * self.mV + (1 - beta1) * dV
        vV = beta2 * self.vV + (1 - beta2) * (dV ** 2)
        md = beta1 * self.md + (1 - beta1) * dd
        vd = beta2 * self.vd + (1 - beta2) * (dd ** 2)

        print("--Layer1")
        #Layer 1
        mW_in = beta1 * self.mW_in + (1 - beta1) * dW_in
        mW_out = beta1 * self.mW_out + (1 - beta1) * dW_out
        mW_forget = beta1 * self.mW_forget + (1 - beta1) * dW_forget
        mW_cand = beta1 * self.mW_cand + (1 - beta1) * dW_cand
        mU_in = beta1 * self.mU_in + (1 - beta1) * dU_in
        mU_out = beta1 * self.mU_out + (1 - beta1) * dU_out
        mU_forget = beta1 * self.mU_forget + (1 - beta1) * dU_forget
        mU_cand = beta1 * self.mU_cand + (1 - beta1) * dU_cand
        mb_in = beta1 * self.mb_in + (1 - beta1) * db_in
        mb_out = beta1 * self.mb_out + (1 - beta1) * db_out
        mb_forget = beta1 * self.mb_forget + (1 - beta1) * db_forget
        mb_cand = beta1 * self.mb_cand + (1 - beta1) * db_cand
        vW_in = beta2 * self.vW_in + (1 - beta2) * dW_in ** 2
        vW_out = beta2 * self.vW_out + (1 - beta2) * dW_out ** 2
        vW_forget = beta2 * self.vW_forget + (1 - beta2) * dW_forget ** 2
        vW_cand = beta2 * self.vW_cand + (1 - beta2) * dW_cand ** 2
        vU_in = beta2 * self.vU_in + (1 - beta2) * dU_in ** 2
        vU_out = beta2 * self.vU_out + (1 - beta2) * dU_out ** 2
        vU_forget = beta2 * self.vU_forget + (1 - beta2) * dU_forget ** 2
        vU_cand = beta2 * self.vU_cand + (1 - beta2) * dU_cand ** 2
        vb_in = beta2 * self.vb_in + (1 - beta2) * db_in ** 2
        vb_out = beta2 * self.vb_out + (1 - beta2) * db_out ** 2
        vb_forget = beta2 * self.vb_forget + (1 - beta2) * db_forget ** 2
        vb_cand = beta2 * self.vb_cand + (1 - beta2) * db_cand ** 2

        print("--Layer2")
        #Layer 2
        mW1_in = beta1 * self.mW1_in + (1 - beta1) * dW1_in
        mW1_out = beta1 * self.mW1_out + (1 - beta1) * dW1_out
        mW1_forget = beta1 * self.mW1_forget + (1 - beta1) * dW1_forget
        mW1_cand = beta1 * self.mW1_cand + (1 - beta1) * dW1_cand
        mU1_in = beta1 * self.mU1_in + (1 - beta1) * dU1_in
        mU1_out = beta1 * self.mU1_out + (1 - beta1) * dU1_out
        mU1_forget = beta1 * self.mU1_forget + (1 - beta1) * dU1_forget
        mU1_cand = beta1 * self.mU1_cand + (1 - beta1) * dU1_cand
        mb1_in = beta1 * self.mb1_in + (1 - beta1) * db1_in
        mb1_out = beta1 * self.mb1_out + (1 - beta1) * db1_out
        mb1_forget = beta1 * self.mb1_forget + (1 - beta1) * db1_forget
        mb1_cand = beta1 * self.mb1_cand + (1 - beta1) * db1_cand
        vW1_in = beta2 * self.vW1_in + (1 - beta2) * dW1_in ** 2
        vW1_out = beta2 * self.vW1_out + (1 - beta2) * dW1_out ** 2
        vW1_forget = beta2 * self.vW1_forget + (1 - beta2) * dW1_forget ** 2
        vW1_cand = beta2 * self.vW1_cand + (1 - beta2) * dW1_cand ** 2
        vU1_in = beta2 * self.vU1_in + (1 - beta2) * dU1_in ** 2
        vU1_out = beta2 * self.vU1_out + (1 - beta2) * dU1_out ** 2
        vU1_forget = beta2 * self.vU1_forget + (1 - beta2) * dU1_forget ** 2
        vU1_cand = beta2 * self.vU1_cand + (1 - beta2) * dU1_cand ** 2
        vb1_in = beta2 * self.vb1_in + (1 - beta2) * db1_in ** 2
        vb1_out = beta2 * self.vb1_out + (1 - beta2) * db1_out ** 2
        vb1_forget = beta2 * self.vb1_forget + (1 - beta2) * db1_forget ** 2
        vb1_cand = beta2 * self.vb1_cand + (1 - beta2) * db1_cand ** 2


        #If it's ugly but it works then it's not ugly
        print("Loading adam_step")        
        self.adam_step = theano.function(
            [x, y, learnRate, t],
            [], 
            updates=[(self.E, self.E - learnRate * (mE / (1-(beta1 ** t))) / (T.sqrt((vE / (1-(beta2 ** t)))) + eps)),
                     (self.V, self.V - learnRate * (mV / (1-(beta1 ** t))) / (T.sqrt((vV / (1-(beta2 ** t)))) + eps)),
                     (self.d, self.d - learnRate * (md / (1-(beta1 ** t))) / (T.sqrt((vd / (1-(beta2 ** t)))) + eps)),
                     (self.W_in, self.W_in - learnRate * (mW_in / (1-(beta1 ** t))) / (T.sqrt((vW_in / (1-(beta2 ** t)))) + eps)),
                     (self.W_out, self.W_out - learnRate * (mW_out / (1-(beta1 ** t))) / (T.sqrt((vW_out / (1-(beta2 ** t)))) + eps)),
                     (self.W_forget, self.W_forget - learnRate * (mW_forget / (1-(beta1 ** t))) / (T.sqrt((vW_forget / (1-(beta2 ** t)))) + eps)),
                     (self.W_cand, self.W_cand - learnRate * (mW_cand / (1-(beta1 ** t))) / (T.sqrt((vW_cand / (1-(beta2 ** t)))) + eps)),
                     (self.U_in, self.U_in - learnRate * (mU_in / (1-(beta1 ** t))) / (T.sqrt((vU_in / (1-(beta2 ** t)))) + eps)),
                     (self.U_out, self.U_out - learnRate * (mU_out / (1-(beta1 ** t))) / (T.sqrt((vU_out / (1-(beta2 ** t)))) + eps)),
                     (self.U_forget, self.U_forget - learnRate * (mU_forget / (1-(beta1 ** t))) / (T.sqrt((vU_forget / (1-(beta2 ** t)))) + eps)),
                     (self.U_cand, self.U_cand - learnRate * (mU_cand / (1-(beta1 ** t))) / (T.sqrt((vU_cand / (1-(beta2 ** t)))) + eps)),
                     (self.b_in, self.b_in - learnRate * (mb_in / (1-(beta1 ** t))) / (T.sqrt((vb_in / (1-(beta2 ** t)))) + eps)),
                     (self.b_out, self.b_out - learnRate * (mb_out / (1-(beta1 ** t))) / (T.sqrt((vb_out / (1-(beta2 ** t)))) + eps)),
                     (self.b_forget, self.b_forget - learnRate * (mb_forget / (1-(beta1 ** t))) / (T.sqrt((vb_forget / (1-(beta2 ** t)))) + eps)),
                     (self.b_cand, self.b_cand - learnRate * (mb_cand / (1-(beta1 ** t))) / (T.sqrt((vb_cand / (1-(beta2 ** t)))) + eps)),
                     (self.W1_in, self.W1_in - learnRate * (mW1_in / (1-(beta1 ** t))) / (T.sqrt((vW1_in / (1-(beta2 ** t)))) + eps)),
                     (self.W1_out, self.W1_out - learnRate * (mW1_out / (1-(beta1 ** t))) / (T.sqrt((vW1_out / (1-(beta2 ** t)))) + eps)),
                     (self.W1_forget, self.W1_forget - learnRate * (mW1_forget / (1-(beta1 ** t))) / (T.sqrt((vW1_forget / (1-(beta2 ** t)))) + eps)),
                     (self.W1_cand, self.W1_cand - learnRate * (mW1_cand / (1-(beta1 ** t))) / (T.sqrt((vW1_cand / (1-(beta2 ** t)))) + eps)),
                     (self.U1_in, self.U1_in - learnRate * (mU1_in / (1-(beta1 ** t))) / (T.sqrt((vU1_in / (1-(beta2 ** t)))) + eps)),
                     (self.U1_out, self.U1_out - learnRate * (mU1_out / (1-(beta1 ** t))) / (T.sqrt((vU1_out / (1-(beta2 ** t)))) + eps)),
                     (self.U1_forget, self.U1_forget - learnRate * (mU1_forget / (1-(beta1 ** t))) / (T.sqrt((vU1_forget / (1-(beta2 ** t)))) + eps)),
                     (self.U1_cand, self.U1_cand - learnRate * (mU1_cand / (1-(beta1 ** t))) / (T.sqrt((vU1_cand / (1-(beta2 ** t)))) + eps)),
                     (self.b1_in, self.b1_in - learnRate * (mb1_in / (1-(beta1 ** t))) / (T.sqrt((vb1_in / (1-(beta2 ** t)))) + eps)),
                     (self.b1_out, self.b1_out - learnRate * (mb1_out / (1-(beta1 ** t))) / (T.sqrt((vb1_out / (1-(beta2 ** t)))) + eps)),
                     (self.b1_forget, self.b1_forget - learnRate * (mb1_forget / (1-(beta1 ** t))) / (T.sqrt((vb1_forget / (1-(beta2 ** t)))) + eps)),
                     (self.b1_cand, self.b1_cand - learnRate * (mb1_cand / (1-(beta1 ** t))) / (T.sqrt((vb1_cand / (1-(beta2 ** t)))) + eps)),
                     (self.mE, mE),
                     (self.vE, vE),
                     (self.mV, mV),
                     (self.vV, vV),
                     (self.md, md),
                     (self.vd, vd),
                     (self.mW_in, mW_in),
                     (self.mW_out, mW_out),
                     (self.mW_forget, mW_forget),
                     (self.mW_cand, mW_cand),                 
                     (self.mU_in, mU_in),
                     (self.mU_out, mU_out),
                     (self.mU_forget, mU_forget),
                     (self.mU_cand, mU_cand),                     
                     (self.mb_in, mb_in),
                     (self.mb_out, mb_out),
                     (self.mb_forget, mb_forget),
                     (self.mb_cand, mb_cand),
                     (self.vW_in, vW_in),
                     (self.vW_out, vW_out),
                     (self.vW_forget, vW_forget),
                     (self.vW_cand, vW_cand),  
                     (self.vU_in, vU_in),
                     (self.vU_out, vU_out),
                     (self.vU_forget, vU_forget),
                     (self.vU_cand, vU_cand),                     
                     (self.vb_in, vb_in),
                     (self.vb_out, vb_out),
                     (self.vb_forget, vb_forget),
                     (self.vb_cand, vb_cand),
                     (self.mW1_in, mW1_in),
                     (self.mW1_out, mW1_out),
                     (self.mW1_forget, mW1_forget),
                     (self.mW1_cand, mW1_cand),
                     (self.mU1_in, mU1_in),
                     (self.mU1_out, mU1_out),
                     (self.mU1_forget, mU1_forget),
                     (self.mU1_cand, mU1_cand),                     
                     (self.mb1_in, mb1_in),
                     (self.mb1_out, mb1_out),
                     (self.mb1_forget, mb1_forget),
                     (self.mb1_cand, mb1_cand),
                     (self.vW1_in, vW1_in),
                     (self.vW1_out, vW1_out),
                     (self.vW1_forget, vW1_forget),
                     (self.vW1_cand, vW1_cand),                     
                     (self.vU1_in, vU1_in),
                     (self.vU1_out, vU1_out),
                     (self.vU1_forget, vU1_forget),
                     (self.vU1_cand, vU1_cand),                     
                     (self.vb1_in, vb1_in),
                     (self.vb1_out, vb1_out),
                     (self.vb1_forget, vb1_forget),
                     (self.vb1_cand, vb1_cand)
                    ])


    
    
    #Creates a one-hot vector representing the class
    def Class(self, z):        
        shell = np.zeros(self.num_Classes)
        shell[z] = 1.
        return shell

    #Performs the preperation steps before learning
    def learning_step(self, x, y, learnRate, t):
        y_ = self.Class(y)
        x_ = self.dropout(x)
        if(len(x_) > 0):
            self.adam_step(x_, y_, learnRate, t)

    #Calculates loss and accuracy on the data set.
    #Because the training set can be large I only calculate accuracy on half of it.
    def valid_Test(self):
        valid_Correct = 0
        train_Correct = 0
        loss = 0
        for i in range(len(self.x_val)):
            if self.f_pred(self.x_val[i]) == self.y_val[i]:
                valid_Correct += 1
        for i in range(len(self.x_train)//2):
            if self.f_pred(self.x_train[i]) == self.y_train[i]:
                train_Correct += 1
            loss += self.ce_error(self.x_train[i], self.Class(self.y_train[i]))
        return valid_Correct/len(self.y_val), train_Correct/(len(self.y_train)//2), loss/(len(self.x_train)//2)


    #Drops words from input by percentage.
    #This prevents the model from memorizing the data set.
    def dropout(self, x_):
        x = deepcopy(x_)
        i = 0
        for w in range(len(x)):
            p = random.randint(1,10)
            if p <= 3:
                x.remove(x[w-i])
                i += 1
        return x

    def sentiment(self, string):
        string_tok = nltk.word_tokenize(string.lower().strip())
        stop_words = set(stopwords.words('english'))

        x = []
        for w in string_tok:
            try:
                if w not in stop_words:
                    x.append(self.w2i[w])
            except:
                x.append(self.w2i[unknown_token])

        p = self.f_pred_Prob(x)
        if p.argmax() == 1:
            out = "Positive"
        else:
            out = "Negative"

        print("\nPredicted sentiment: {}".format(out))
        print(p)


def main():
    l = reviewAI(5000)
    print("1. Test inputs on model\n2. Train model\n")
    choice = input("Choice: ")
    if choice == '1':
        print("Press CTRL+C to exit")
        try:
            while True:
                test = input("==============\nINPUT: ")
                l.sentiment(test)
        except KeyboardInterrupt:
            return
    print("Building model...")
    avg_t = 0
    e_Flag = False
    l_Rate = .00075
    epoch_t = 0
    while True:
        try:
            for t in range(1,10000):
                t1 = time.time()
                cc = 0
                for v in range(len(l.x_train)):
                    l.learning_step(l.x_train[v], l.y_train[v], l_Rate, t)
                    if v % 3000 == 0:
                        if epoch_t > 0:
                            print("{} || {} - ETA: {:.0f}".format(time.asctime(), v, (epoch_t - (time.time() - t1))))
                        else:
                            print("{} || {} Done - Elapsed: {:.0f}".format(time.asctime(), v, (time.time() - t1)))

                        
                print("{} || DONE - Time elapsed: {:.0f}\n=========================".format(time.asctime(), time.time() - t1))
                if t % 1 == 0:
                    valid_per, train_per, loss = l.valid_Test()
                    epoch_t = time.time() - t1
                    print("\n{0} || Epoch {1}".format(time.asctime(), t))
                    print("Epoch Time: {0:.0f}".format(epoch_t))
                    print("Train Correct: {0:.2f}%".format(train_per*100))
                    print("Validation Correct: {0:.2f}%".format(valid_per*100))
                    print("Loss: {0:.2f}\n=========================\n".format(loss))
                    if (loss <= .35):
                        e_Flag = True
                        print("SUCCESS")
                        print("Time taken: ", epoch_t * t, "seconds")
                        break
                if e_Flag:
                    break
            if e_Flag:
                break
        except KeyboardInterrupt:
            print("\n==========================\nKEYBOARD INTERRUPT\n==========================")
            e_Flag = True
            break
        except:
            print("UNHANDLED EXCEPTION.\nSomething went horribly wrong.")
            return
            
    if e_Flag:
        s = input("Save? ")
        if s == 'y':
            db = shelve.open("{0}/{0}DB".format(l.test_name), "w")
            db['E'] = l.E
            db['V'] = l.V
            db['d'] = l.d
            db['U_in'] = l.U_in
            db['U_out'] = l.U_out
            db['U_forget'] = l.U_forget
            db['U_cand'] = l.U_cand
            db['W_in'] = l.W_in
            db['W_out'] = l.W_out
            db['W_forget'] = l.W_forget
            db['W_cand'] = l.W_cand
            db['b_in'] = l.b_in
            db['b_out'] = l.b_out
            db['b_forget'] = l.b_forget
            db['b_cand'] = l.b_cand
            db['U1_in'] = l.U1_in
            db['U1_out'] = l.U1_out
            db['U1_forget'] = l.U1_forget
            db['U1_cand'] = l.U1_cand
            db['W1_in'] = l.W1_in
            db['W1_out'] = l.W1_out
            db['W1_forget'] = l.W1_forget
            db['W1_cand'] = l.W1_cand
            db['b1_in'] = l.b1_in
            db['b1_out'] = l.b1_out
            db['b1_forget'] = l.b1_forget
            db['b1_cand'] = l.b1_cand
            db['w2i'] = l.w2i
            db['i2w'] = l.i2w
            db.close()
            print("Model saved")

main()
