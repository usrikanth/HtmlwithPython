import yaml
import pandas as pd
import pickle
import numpy as np
from utils import clean, convert2id, load_data, Vocab, get_char_vocab, get_embed_clean, get_url_passage, get_embed_clean_psg
import re
from tqdm import trange
from itertools import chain
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec,KeyedVectors



def prepare_model_data(data, mode, data_folder, char_vocab_path):
    print('data len!!!!!!!',len(data))
    config = yaml.safe_load(open('config.yml', 'r'))
    emailText1 = data.eMailText
    
    max_word_len = config['max_word_len']
    max_e_len = config['max_e_length']
	
    if char_vocab_path == 'get_char_vocab':
        char_vocab = Vocab()
        char_vocab = get_char_vocab(emailText1, char_vocab)
        output = open("char_vocab.pkl", 'wb')
        pickle.dump(char_vocab, output)
        output.close()
    else:
        char_vocab = pickle.load(open(char_vocab_path, "rb"))
    
    print("preparing {0} data".format(mode))
    data.eMailText, eMailText_char, e_l = get_embed_clean_psg(emailText1, max_e_len, max_word_len, char_vocab)
    tokens = set()
    for d in (data.eMailText):
        for i in d:
            for j in i:
                tokens.add(j)
    tokens = list(tokens)
    i2w = dict(enumerate(tokens))
    w2i = {v: k for k, v in i2w.items()}
    data.eMailText = data.eMailText.apply(lambda x: convert2id(x, w2i))
    
    vec = list()
    c = 0
    wv_model = pickle.load(open("path_to_glove.pkl", "rb"))
    for i in trange(len(tokens)):
        temp = [0] * 300
        if i2w[i] in wv_model:
            temp1 = wv_model[i2w[i]][0]
            cnt = 0
            for t in temp1:
                if str(t).strip() != '':
                    temp[cnt] = float(t)
                    cnt += 1
        elif i2w[i].lower().title() in wv_model:
            temp1 = wv_model[i2w[i].lower().title()][0]
            cnt = 0
            for t in temp1:
                if str(t).strip() != '':
                    temp[cnt] = float(t)
                    cnt += 1
        elif i2w[i].lower() in wv_model:
            temp1 = wv_model[i2w[i].lower()][0]
            cnt = 0
            for t in temp1:
                if str(t).strip() != '':
                    temp[cnt] = float(t)
                    cnt += 1            
        elif i2w[i].upper() in wv_model:
            temp1 = wv_model[i2w[i].upper()][0]
            cnt = 0
            for t in temp1:
                if str(t).strip() != '':
                    temp[cnt] = float(t)
                    cnt += 1            
        else:
            c += 1
            temp = np.random.multivariate_normal(np.zeros(300), np.eye(300))
        vec.append(temp)
    temp = np.random.multivariate_normal(np.zeros(300), np.eye(300))
    vec.append(temp)
    print("{0} tokens not found in vocab".format(c))
    e = pad_sequences(data.eMailText, maxlen=max_e_len, dtype='int32', padding='post', value= len(vec)-1)

    data = list()
    for i in trange(len(q)):
        if start_idx[i] != -1:
            data.append({"e": p[i], "e_c": eMailText_char[i]})
    print('data len'+ str(len(data)))
    pickle.dump(data, open(
        "{0}/data.pkl".format(data_folder), "wb"))
    pickle.dump(
        vec, open("{0}/vectors_fast.pkl".format(data_folder), "wb"))
    pickle.dump(
        i2w, open("{0}/index2word.pkl".format(data_folder), "wb"))
    print('done!!!')

def saveDataModelVec(file, mode, data_folder, start, end, char_vocab_path):
    data_old = open(file, "r").readlines()
    print('file read complete!!!!!')
    #config = yaml.safe_load(open('config.yml', 'r'))
    p_index = 0
    
    if start == -1 and end == -1:
        data = [x.split("\t") for x in data_old]
    elif end == -1:
        data = [data_old[x].split("\t") for x in range(start,len(data_old))]
    else:
        data = [data_old[x].split("\t") for x in range(start,end)]

    #if mode != 'test':
    #    data = np.array([[x[q_index], x[p_index], x[a_start_index].strip(), x[a_end_index].strip(), x[id_index]] for x in data])
    #    print(data.shape)
    #    data = pd.DataFrame({"QueryText": data[:, 0], "PassageText": data[:, 1], "start_index": data[:, 2], "end_index": data[:, 3], "ID": data[:, 4]})
    #else:
    data = np.array([[x[p_index]] for x in data])
    data = pd.DataFrame({"eMailText": data[:, 0]})
    print('data framed!!!!!!')
    data = data[data.QueryText != ""].reset_index(drop=True)
    #data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)
    data_file = data[:].reset_index(drop=True)
    print('preparing model')
    return prepare_model_data(data_file, mode, data_folder, char_vocab_path)

saveDataModelVec('/data/papatha/SQuAD/train_qp_new_3.txt','train','/data/papatha/SQuAD/train_new_3/', -1, -1, 'char_vocab.pkl')
#saveDataModelVec('path_to_input_file','data_type_train_dev_test','path_to_output_folder', -1(begin line nidex from input file), -1(end line nidex from input file), 'char_vocab.pkl')