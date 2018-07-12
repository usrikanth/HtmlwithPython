import tensorflow as tf
import numpy as np
import pickle
from keras.layers import Dense, core
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import re
from tqdm import trange
from itertools import chain
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
import nltk


class Vocab:
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []
    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)
        return self._token2index[token]
    def check(self, token):
        if token in self._token2index:
            return False
        else:
            return True
    def len(self):
        return len(self._token2index)
    @property
    def size(self):
        return len(self._token2index)
    def token(self, index):
        return self._index2token[index]
    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index
    def get(self, token, default=None):
        return self._token2index.get(token, default)
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)
        return cls(token2index, index2token)
        
def get_url_passage(URL, Title, u_len, t_len):
    URL_o = []
    Title_o = []
    for i in range(0, len(URL)):
        u = (URL[i])
        t = Title[i].strip()
        u, t_ = filter_url(u)
        URL_o.append(clean(u,u_len))
        t =str(t).lower()
        if t == 'null':
            t = t_
        Title_o.append(clean(t,t_len))
    return URL_o, Title_o

def filter_url(x):
    x = re.split(r'/', x.rstrip('/'))
    url_tail = x[len(x)-1]
    url_tail = url_tail.replace('-',' ')
    url_tail = url_tail.replace('_',' ')
    url_head = x[2]
    url_head = re.sub('http://|https://|www\.|\.com|\.org|\.uk|\.me|.edu|\.net|\.in|\.eu|\.html|\.htm','', url_head)
    url_head = url_head.replace('.',' ')
    return url_head, url_tail
    
def round_(x):
    return list(map(lambda x: 0 if x < 0.5 else 1, x))

def get_vecs(batch_size, p_tokens, q_tokens, wv_model):
    vec = list()
    for i in range(0, batch_size):
        for j in range(len(p_tokens[i])):
            vec.append(wv_model[j])
        for j in range(len(q_tokens)):
            vec.append(wv_model[j])
    return vec

def masked_softmax(a, max_len):
    l = tf.sign(tf.abs(a))
    base = tf.reduce_sum(tf.exp(a) * l, reduction_indices=-1)
    numerator = tf.exp(a) / tf.tile(tf.expand_dims(base, 1),
                                    tf.constant([1, max_len]))
    return numerator * l


def repeat(seq, times):
    return core.RepeatVector(times)(seq)

def matrics_new(truth_bg, truth_ed, pred_bg, pred_ed):
    total = len(truth_bg)
    em = 0
    partial = 0
    
    for i in range(0, len(truth_bg)):
        t_bg = truth_bg[i]
        t_ed = truth_ed[i]
        p_bg, p_ed, max, rank = getNearestBgEd(pred_bg[i], pred_ed[i], pred_bg[i][t_bg] * pred_ed[i][t_ed])
        if t_bg == p_bg and t_ed == p_ed:
            em += 1
            partial += 1
        elif (p_bg <= t_bg and p_ed >= t_bg) or (t_bg <= p_bg and t_ed >= p_bg): #and p_bg <= p_ed:
            partial += 1
    return em/ total, partial/total

def writetoFile(fileName, obj):
    with open(fileName, 'a') as f:
        f.write(str(obj))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
    f.close()

def getNearestBgEd(mat_bg, mat_ed, or_score):
    bg = -1
    ed = -1
    max = -1.0
    cnt = 0
    for e in range(0, len(mat_bg)):
        if e < 20:
            c_b = 0
        else:
            c_b = e - 20
        for b in range(c_b, e+1):
            curr = mat_bg[b] * mat_ed[e]
            if curr > max:
                max = curr
                bg = b
                ed = e
            if curr > or_score:
                cnt += 1
    return bg, ed, max, cnt
   
def metrics(preds, true):
    preds_c = round_(preds)
    return {
        # "f1_score": f1_score(y_true=true, y_pred=preds_c, average="macro"),
        "confusion-matrix": confusion_matrix(y_true=true, y_pred=preds_c),
        "AUC": roc_auc_score(y_score=preds, y_true=true, average="macro")}


def dense(dim, use_bias=False, init="glorot_uniform", activation=None):
    return Dense(dim, kernel_initializer=init, use_bias=use_bias, activation=activation)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def getExample(q, p, bg, ed, i2w, or_bg, or_ed, epoch, itr, p_length, fileName, max, ids):
    with open(fileName, 'a') as f:
        f.write('--------------------------------------- start epoch\n')
        f.write('epoch:' + str(epoch) + '\t' + str(itr) + '\n' )
        
        i1 = 0
        for i in range(0, len(q)):
            q_ = q[i]
            p_ = p[i]
            or_bg_ = or_bg[i][0]
            or_ed_ = or_ed[i][0]
            org_prob = bg[i][or_bg_] * ed[i][or_ed_]
            bg_, ed_, max_prob, or_rank = getNearestBgEd(bg[i], ed[i], org_prob)
            #f.write(str(bg[i]))
            #f.write('\n')
            #f.write(str(ed[i]))
            #f.write('\n')
            Query = ''
            Passage = ''
            Or_Ans = ''
            Pr_Ans = ''
            p_len = p_length[i]
            for j in range(0, len(q_)):
                if q_[j] < len(i2w) - 1:
                    Query = Query + ' ' + i2w[q_[j]]
            for j in range(0, len(p_)):
                if p_[j] < len(i2w) - 1:
                    Passage = Passage + ' ' + i2w[p_[j]]
            for j in range(or_bg_, (or_ed_ + 1)):
                if p_[j] < len(i2w) - 1:
                    Or_Ans = Or_Ans + ' ' + i2w[p_[j]]
            for j in range(bg_, (ed_ + 1)):
                if p_[j] < len(i2w) - 1:
                    Pr_Ans = Pr_Ans + ' ' + i2w[p_[j]]
            if (or_bg_ == bg_ and or_ed_ == ed_) or (bg_ <= or_bg_ and ed_ >= or_ed_) or (or_bg_ <= bg_ and or_ed_ >= bg_) or i1 < max:
                f.write('------------------------------------\n')
                if or_bg_ == bg_ and or_ed_ == ed_:
                    f.write('-------------exact match-----------\n')
                elif (bg_ <= or_bg_ and ed_ >= or_ed_) or (or_bg_ <= bg_ and or_ed_ >= bg_): #and bg_ <= ed_):
                    f.write('-------------partial match-----------\n')
                f.write('id:' + str(ids[i]) + '\n')
                f.write('or bg ed: ' + str(or_bg_) + '\t' + str(or_ed_)+ '\n')
                f.write('pr bg ed: ' + str(bg_) + '\t' + str(ed_) + '\n')
                f.write('or prob: ' + str(org_prob)+ '\n')
                f.write('pred prob: ' + str(max_prob) + '\n')
                f.write('or rank: ' + str(or_rank)+ '\n')
                f.write(Query + '\t' + Passage + '\n or ans: ' + Or_Ans + '\n pr ans: ' + Pr_Ans + '\n')
                f.write('passage length: '+ str(p_len))
                i1 += 1
        f.write('----------------------------------------end epoch\n')
        f.close()
        
def getExampleNew(q, p, bg, ed, i2w, or_bg, or_ed, epoch, itr, p_length, fileName, max):
    with open(fileName, 'a') as f:
        #f.write('--------------------------------------- start epoch\n')
        #f.write('epoch:' + str(epoch) + '\t' + str(itr) + '\n' )
        i1 = 0
        for i in range(0, len(q)):
            q_ = q[i]
            p_ = p[i]
            or_bg_ = or_bg[i][0]
            or_ed_ = or_ed[i][0]
            org_prob = bg[i][or_bg_] * ed[i][or_ed_]
            bg_, ed_, max_prob, or_rank = getNearestBgEd(bg[i], ed[i], org_prob)
            Query = ''
            Passage = ''
            Or_Ans = ''
            Pr_Ans = ''
            p_len = p_length[i]
            for j in range(0, len(q_)):
                if q_[j] < len(i2w) - 1:
                    Query = Query + ' ' + i2w[q_[j]]
            for j in range(0, len(p_)):
                if p_[j] < len(i2w) - 1:
                    Passage = Passage + ' ' + i2w[p_[j]]
            for j in range(or_bg_, (or_ed_ + 1)):
                if p_[j] < len(i2w) - 1:
                    Or_Ans = Or_Ans + ' ' + i2w[p_[j]]
            for j in range(bg_, (ed_ + 1)):
                if p_[j] < len(i2w) - 1:
                    Pr_Ans = Pr_Ans + ' ' + i2w[p_[j]]
            if (or_bg_ == bg_ and or_ed_ == ed_) or (bg_ <= or_bg_ and ed_ >= or_ed_) or (or_bg_ <= bg_ and or_ed_ >= bg_) or i1 < max:
                if or_bg_ == bg_ and or_ed_ == ed_:
                    f.write('1\t0\t')
                elif (bg_ <= or_bg_ and ed_ >= or_ed_) or (or_bg_ <= bg_ and or_ed_ >= bg_): #and bg_ <= ed_):
                    f.write('0\t1\t')
                else:
                    f.write('0\t0\t')
                f.write(str(or_bg_) + '\t' + str(or_ed_)+ '\t')
                f.write(str(bg_) + '\t' + str(ed_) + '\t')
                f.write(str(org_prob)+ '\t')
                f.write(str(max_prob) + '\t')
                f.write(str(or_rank)+ '\t')
                f.write(Query + '\t' + Passage + '\t' + Or_Ans + '\t' + Pr_Ans + '\n')
                i1 += 1
        #f.write('----------------------------------------end epoch\n')
        f.close()
        
        
def get_batch(data, start, size, ids, p_seq, q_seq):
    if ids is None:
        ids = np.arange(len(data))
    batch = data[ids][start: start + size]
    batch_p = np.array([x['p'] for x in batch])
    batch_q = np.array([x['q'] for x in batch])
    batch_pc = np.array([x['p_c'] for x in batch])
    batch_qc = np.array([x['q_c'] for x in batch])
    batch_a_start = np.reshape(np.array([x['a_start'] for x in batch]), [len(batch), 1])
    batch_a_end = np.reshape(np.array([x['a_end'] for x in batch]), [len(batch), 1])
    batch_ql = np.array([x['q_l'] for x in batch])
    batch_pl = np.array([x['p_l'] for x in batch])
    batch_q_seq = np.array([q_seq for x in batch])
    batch_p_seq = np.array([p_seq for x in batch])
    batch_id = np.array([x['ID'] for x in batch])
    return batch_p, batch_q, batch_a_start, batch_a_end, batch_ql, batch_pl, batch_q_seq, batch_p_seq, batch_id, batch_qc, batch_pc


def cosine_similarity(x, y):
    numerator = tf.reduce_sum(tf.multiply(x, y), -1)
    denominator = tf.sqrt(tf.reduce_sum(tf.multiply(x, x), -1)) * \
        tf.sqrt(tf.reduce_sum(tf.multiply(x, x), -1))
    return numerator / denominator


def load_data(mode):
    data = np.array(pickle.load(
        open('/data/papatha/SQuAD/{0}/data.pkl'.format(mode), 'rb')))
    vec = np.array(pickle.load(
        open('/data/papatha/SQuAD/{0}/vectors_fast.pkl'.format(mode), 'rb')))
    i2w = pickle.load(
        open('//data/papatha/SQuAD/{0}/index2word.pkl'.format(mode), 'rb'))
    i2w[-1] = ''
    tf.logging.info("Loaded {0} data of length: {1}".format(mode, len(data)))
    return data, vec, i2w


def oversample(data, label, frac):
    data = data.copy()
    data_minor = np.array([x for x in data if x["a"] == label])
    print(len(data_minor))
    data_minor = data_minor[np.random.rand(len(data_minor)) < frac]
    print(len(data_minor))
    data = np.r_[data, data_minor]
    return data


def resume_model(sess, model_name):
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

def get_char_vocab(data, char_vocab):    
    for sent in data:
        for word in sent:
            for char in word:
                char_vocab.feed(char)
    char_vocab.feed('unk')
    print('char vocab size', len(char_vocab._token2index))
    return char_vocab

def get_embed_clean(data, max_sent_len, max_word_len, char_vocab):
    char_input = np.zeros((len(data), max_sent_len, max_word_len))
    word_input = []
    length = []
    max_wrd = 0
    for i, sent in enumerate(data):
        words = clean(sent, max_sent_len)
        word_input.append(words)
        len_to_append = len(words)
        if len_to_append > max_wrd:
            max_wrd = len_to_append
        if len_to_append > max_sent_len:
            len_to_append = max_sent_len
        length.append(len(words))
        for j, word in enumerate(words):
            if j >= max_sent_len:
                break
            for k,c in enumerate(word):
                if k >= max_word_len:
                    break
                try:
                    char_input[i,j,k] = char_vocab._token2index[c]
                except:
                    #print(word)
                    char_input[i,j,k] = char_vocab._token2index['unk']
            for k_t in range(k, max_word_len):
                char_input[i,j,k_t] = char_vocab._token2index['unk']
        for j_t in range(j, max_sent_len):
            for k_t in range(0, max_word_len):
                char_input[i,j_t,k_t] = char_vocab._token2index['unk']
    print('max word len: '+str(max_wrd))
    return word_input, char_input, length
    
def get_embed_clean_psg(data, max_sent_len, max_word_len, char_vocab):
    char_input = np.zeros((len(data), max_sent_len, max_word_len))
    word_input = []
    length = []
    start_index = []
    end_index = []
    max_wrd = 0
    for i, sent in enumerate(data):
        words = clean(sent, max_sent_len)
        word_input.append(words)
        len_to_append = len(words)
        if len_to_append > max_wrd:
            max_wrd = len_to_append
        if len_to_append > max_sent_len:
            len_to_append = max_sent_len
        length.append(len(words))
        for j, word in enumerate(words):
            if j >= max_sent_len:
                break
            for k,c in enumerate(word):
                if k >= max_word_len:
                    break
                try:
                    char_input[i,j,k] = char_vocab._token2index[c]
                except:
                    #print(word)
                    char_input[i,j,k] = char_vocab._token2index['unk']
            for k_t in range(k, max_word_len):
                char_input[i,j,k_t] = char_vocab._token2index['unk']
        for j_t in range(j, max_sent_len):
            for k_t in range(0, max_word_len):
                char_input[i,j_t,k_t] = char_vocab._token2index['unk']
    return word_input, char_input, length
    
    
def getAnswerBeginEndToken(org_text, word_token, start_index, end_index):
    answer_text = org_text[int(start_index) : int(end_index)]
    try:
        prev_start_token = len(re.split(answer_text, org_text.rstrip(' ')))
        prev_end_token = prev_start_token + len(re.split(r' +', answer_text.rstrip(' ')))
    except:
        return -1, -1, answer_text
    #answer_text = answer_text.replace("\"", "")
    
    word_split = clean(answer_text, 200)
    if len(word_split) <= 0:
     return -1, -1, ""

    beginFound = False
    bg = 0
    j = 0
    lastStart = 0
    found_text = ''
    with open('foundBgEd.txt', 'a') as f:
        for i in range(0, len(word_token)):
            #print(word_token[i])
            if word_token[i] == word_split[j] and j == 0:        
                beginFound = True
                lastStart = i
                bg = i
                j += 1
                found_text = found_text + ' ' + word_token[i]
                if j >= len(word_split):
                    f.write(org_text + '\t' + answer_text + '\t' + found_text + '\n')
                    return i, i, answer_text
            elif beginFound == True and word_token[i] == word_split[j]:
                j += 1
                found_text = found_text + ' ' + word_token[i]
                if j >= len(word_split):
                    f.write(org_text + '\t' + answer_text + '\t' + found_text + '\n')
                    return bg, i, answer_text
            else:
                j = 0
                found_text = ''
                if beginFound == True:
                    i = lastStart + 1
                    beginFound = False
    f.close()
    return -1, -1, answer_text
       
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    
    return distances[-1]       
    
def clean(x, length):
    #x = "parth."
    #x = str(x).lower()
    x = re.sub('-', ' - ', x)
    x = re.sub('–', ' – ', x)
    x = re.sub('—', ' — ', x)
    x = re.sub('\.', ' . ', x)
    x = re.sub('\"', ' " ', x)
    x = re.sub('``', ' `` ', x)
    x = re.sub('\'', ' \' ', x)
    x = x.encode("utf-8", errors="ignore").decode()
    x = nltk.word_tokenize(x)
    #if len(x) > length:
    #    x = x[:length]
    return x
    
def store_result(preds, ids, store_result, fileName):
    file = ''.join([store_result+fileName])
    with open(file, 'w') as f:
        for i in range(0, len(preds)):
            #print("{0}\t{1}\t\n".format(ids[i],preds[i]))
            f.write("{0}\t{1}\t".format(ids[i], preds[i][1]))
            f.write("\n")
        f.close()


def convert2id(x, w2i):
    return [w2i[y] for y in x]
