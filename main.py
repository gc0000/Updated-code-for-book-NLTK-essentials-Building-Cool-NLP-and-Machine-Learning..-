from sklearn.metrics import confusion_matrix

globals().clear()

#Load the Brown corpus with nltk
import nltk
nltk.data.path.append('F:\\nltk_data')#add a new path for nltk data
#nltk.download('popular',download_dir='F:\\nltk_data')
#or select packages from https://www.nltk.org/nltk_data/, download and unpack to a subfolder "corpora" of the new path added
#how to set subfolders: see https://www.nltk.org/data.html

#import brown
from nltk.corpus import brown
print(brown.categories())#text catagories of brown

#sentence splitter
nltk.download('punkt_tab',download_dir='F:\\nltk_data')
from nltk.tokenize import sent_tokenize
inputstring = ' This is an example sent. The sentence splitter will split on sent markers. Ohh really !!'
all_sent = sent_tokenize(inputstring)

#word tokenization
s = "Hi Everyone !    hola gr8" # simplest tokenizer
print(s.split())
from nltk.tokenize import word_tokenize
word_tokenize(s)#word tokenization choice 1
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize
regexp_tokenize(s, pattern='\w+')#words
regexp_tokenize(s, pattern='\d+')#digits
wordpunct_tokenize(s)#word tokenization choice 2
blankline_tokenize(s)

#stemming
from nltk.stem import PorterStemmer # import Porter stemmer
from nltk.stem.lancaster import LancasterStemmer
#nltk.download('snowball_data',download_dir='F:\\nltk_data')
from nltk.stem.snowball import SnowballStemmer
pst = PorterStemmer()   # create obj of the PorterStemmer
lst = LancasterStemmer() # create obj of LancasterStemmer
lst.stem("eating")
pst.stem("shopping")

#lemmatization
from nltk.stem import WordNetLemmatizer
wlem = WordNetLemmatizer()
print(wlem.lemmatize('ate'))
#???why it's still 'ate'

#stop words
from nltk.corpus import stopwords
stoplist = stopwords.words('english') # config the language name
text = "This is just a test"
cleanwordlist = [word for word in text.split() if word not in stoplist]
print(cleanwordlist)

#rare words removal
freq_dist = nltk.FreqDist(token)# tokens is a list of all tokens in corpus
rarewords = freq_dist.keys()[-50:]
after_rare_words = [ word for word in token not in rarewords]

#spelling correction (edit distance caculation)
from nltk.metrics import edit_distance
edit_distance("rain","shine")

#POS tagging
from nltk import word_tokenize
s = "I was watching TV"
print(nltk.pos_tag(word_tokenize(s)))
tagged = nltk.pos_tag(word_tokenize(s))#a MEC tagger
allnoun = [word for word,pos in tagged if pos in ['NN','NNP'] ]
print(allnoun)
#an alternative !!!
from nltk.tag.stanford import StanfordPOSTagger
import sys
sys.path.append('F:/stanford-tagger-4.2.0/')
model='F:/stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/models/english-bidirectional-distdim.tagger'
jar='F:/stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
stan_tagger = StanfordPOSTagger(model,jar)

#frquency of POS tags in Brown
from nltk.corpus import brown
import nltk
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
print(nltk.FreqDist(tags))

#accurate prediction of NN / total NN tag
brown_tagged_sents = brown.tagged_sents(categories='news')
default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.evaluate(brown_tagged_sents))

#N-gram tagger (a subclass of sequential taggers)
from nltk.tag import UnigramTagger
from nltk.tag import DefaultTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
train_data = brown_tagged_sents[:int(len(brown_tagged_sents) * 0.9)]
test_data = brown_tagged_sents[int(len(brown_tagged_sents) * 0.9):]
unigram_tagger = UnigramTagger(train_data,backoff=default_tagger)
print(unigram_tagger.evaluate(test_data))
bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)
print(bigram_tagger.evaluate(test_data))
trigram_tagger = TrigramTagger(train_data,backoff=bigram_tagger)
print(trigram_tagger.evaluate(test_data))

#regex tagger
from nltk.tag.sequential import RegexpTagger
regexp_tagger = RegexpTagger(
         [( r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
          ( r'(The|the|A|a|An|an)$', 'AT'),   # articles
          ( r'.*able$', 'JJ'),                # adjectives
          ( r'.*ness$', 'NN'),         # nouns formed from adj
          ( r'.*ly$', 'RB'),           # adverbs
          ( r'.*s$', 'NNS'),           # plural nouns
          ( r'.*ing$', 'VBG'),         # gerunds
          (r'.*ed$', 'VBD'),           # past tense verbs
          (r'.*', 'NN')                # nouns (default)
          ])
print(regexp_tagger.evaluate(test_data))

#NER (Named Entity Recognition)
import nltk
import numpy
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
Sent = "Mark is studying at Stanford University in California"
print(ne_chunk(nltk.pos_tag(word_tokenize(Sent)), binary=False))
#!!!an alternative
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger('F:/nlp-stanford/stanford-ner-4.2.0/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz','F:/nlp-stanford/stanford-ner-4.2.0/stanford-ner-2020-11-17/stanford-ner.jar')
st.tag('Rami Eid is studying at Stony Brook University in NY'.split())

#toy CFG
from nltk import CFG
toy_grammar = nltk.CFG.fromstring(
 """
  S -> NP VP                
  VP -> V NP              
  V -> "eats" | "drinks"  
  NP -> Det N   
  Det -> "a" | "an" | "the"  
  N -> "president" |"Obama" |"apple"| "coke"  
    """)
print(toy_grammar.productions())

#Regex Parser
import nltk
from nltk.chunk.regexp import *
reg_parser = RegexpParser('''
        NP: {<DT>? <JJ>* <NN>*}     # NP
         P: {<IN>}                  # Preposition
         V: {<V.*>}                 # Verb
        PP: {<P> <NP>}              # PP -> P NP
        VP: {<V> <NP|PP>*}          # VP -> V (NP|PP)*
  ''')
test_sent="Mr. Obama played a big role in the Health insurance bill"
test_sent_pos=nltk.pos_tag(nltk.word_tokenize(test_sent))
paresed_out=reg_parser.parse(test_sent_pos)
print(paresed_out)

#Dependency parsing
#Stanford parser
from nltk.parse.stanford import StanfordParser
english_parser = StanfordParser('stanford-parser.jar', 'stanfordparser-3.4-models.jar')
english_parser.raw_parse_sents("this is the english parser test")

#Chunking (shallow parsing)
from nltk.chunk.regexp import *
test_sent="The prime minister announced he had asked the chief government whip, Philip Ruddock, to call a special party room meeting for 9am on Monday to consider the spill motion."
test_sent_pos= nltk.pos_tag(nltk.word_tokenize(test_sent))
rule_vp = RegexpChunkRule(r'(<VB.*>)?(<VB.*>)+(<PRP>)?', 'Chunk VPs', 'simple_vp')
parser_vp = RegexpChunkParser([rule_vp],chunk_label='VP')
print(parser_vp.parse(test_sent_pos))#!!!
rule_np = RegexpChunkRule(r'(<DT>?<RB>?)?<JJ|CD>*(<JJ|CD><,>)*(<NN.*>)+','Chunk NPs', 'simple_np')
parser_np = RegexpChunkParser([rule_np],chunk_label="NP")
print(parser_np.parse(test_sent_pos))#!!!

#NER in chunks
text='Chancellor Rachel Reeves has suggested that Labour manifesto commitments will be ditched amid a “significantly worse” economic inheritance than expected last year. In an interview with the BBC, Reeves said she would prefer to increase government spending on infrastructure rather than cut taxes for Britons. When asked whether the government would stick to manifesto commitments, Reeves struck a coy tone – suggesting that manifesto pledges not to raise income tax rates would be dropped.'
sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
for sent in tagged_sentences:
   print(nltk.ne_chunk(sent))

#relation extraction
import re
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern=IN):
     print(nltk.sem.rtuple(rel))

#summarization:
# importance score
results=[]
for sent_no,sentence in enumerate(nltk.sent_tokenize(text)):
    #here a score is assigned to each sentence in "text"
    #score=(number of NERs + number of nouns)/number of tokens, for each sentence
    no_of_tokens = len(nltk.word_tokenize(sentence))
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    no_of_nouns = len([word for word, pos in tagged if pos in ["NN", "NNP"]])
    ners = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)),binary=False)
    no_of_ners = len([chunk for chunk in ners if hasattr(chunk,'node')])
    score = (no_of_ners + no_of_nouns) / float(no_of_tokens)
    results.append((sent_no, no_of_tokens, no_of_ners, no_of_nouns, score, sentence))
#rank the sentences by scores and print
for sent in sorted(results,key=lambda x: x[2],reverse=True):
    print(sent[5])

#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
results=[]
news_content="Mr. Obama planned to promote the effort on Monday during \
a visit to Camden, N.J. The ban is part of Mr. Obama's push to ease \
tensions between law enforcement and minority \communities in reaction to \
the crises in Baltimore; Ferguson, Mo. We are, without a doubt, sitting \
at a defining moment in American policing, Ronald L. Davis, the director \
of the Office of Community Oriented Policing Services at the Department \
of Justice, told reporters in a conference call organized by the White \
House"
sentences=nltk.sent_tokenize(news_content)
vectorizer = TfidfVectorizer(norm='l2',min_df=0.1, use_idf=True, smooth_idf=False, sublinear_tf=True)
sklearn_binary=vectorizer.fit_transform(sentences)
print(sklearn_binary.toarray)
for i in sklearn_binary.toarray():
    results.append(i.sum() / float(len(i.nonzero()[0])))

#Text classification
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
def preprocessing(text):
    # text=text.encode('utf8')
    # text=text.decode('utf8')
    #tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    #remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]
    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]
    # lower capitalization
    tokens = [word.lower() for word in tokens]
    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

#preprocess and label the sms spam collection dataset
sms=open('F:\\TextAnalyses_data\\SMSSpamCollection', encoding='utf8') #code changed
sms_data=[]
sms_labels=[]
csv_reader = csv.reader(sms,delimiter='\t')
csv_reader=[line for line in csv_reader] #code changed
for line in csv_reader: #!!!
    #adding the sms id
    sms_labels.append(line[0])
    # adding the cleaned text We are calling preprocessing method
    sms_data.append(preprocessing(line[1]))
sms.close()

#sampling
import sklearn
import numpy as np
#simple 7:3 sampling
trainset_size = int(round(len(sms_data)*0.70))
x_train = np.array([''.join(el) for el in sms_data[0:trainset_size]])
y_train = np.array([el for el in sms_labels[0:trainset_size]])
x_test = np.array([''.join(el) for el in sms_data[trainset_size+1:len(sms_data)]])
y_test = np.array([el for el in sms_labels[trainset_size+1:len(sms_labels)]])

#generate a term-document matrix with bag-of-word representation
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
X_exp = vectorizer.fit_transform(sms_data)
print(X_exp.toarray())

#using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2),  stop_words='english',  strip_accents='unicode',  norm='l2')
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

#the naive bayes method
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics._classification import confusion_matrix, classification_report #code changed
clf = MultinomialNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print(' \n confusion_matrix \n ')
cm = confusion_matrix(y_test, y_pred) #code changed
print(classification_report(y_test, y_pred))

#extract top n features that contribute to pos/neg classes
feature_names = vectorizer.get_feature_names_out()
# coefs = clf.coef_ #!!!
# intercept = clf.intercept_
# coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
# n=10
# top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
# for (coef_1, fn_1), (coef_2, fn_2) in top:
#     print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))

#Decision trees
#CART classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier().fit(X_train.toarray(), y_train)
y_tree_predicted = clf.predict(X_test.toarray())
print(y_tree_predicted)
print(' \n Here is the classification report:')
print(classification_report(y_test, y_tree_predicted))

#SGD (Stochastic gradient descent)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
clf = SGDClassifier(alpha=.0001, n_iter_no_change=50).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

#SVM
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC().fit(X_train, y_train)
y_svm_predicted = svm_classifier.predict(X_test)
print(classification_report(y_test, y_svm_predicted))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Random forest
from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier(n_estimators=10)
RF=RF_clf.fit(X_train, y_train) #code added
predicted = RF_clf.predict(X_test)
print(classification_report(y_test, predicted))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# K-means
from sklearn.cluster import KMeans, MiniBatchKMeans
import collections #code added
true_k=5 #num of clusters
#not using mini-batch
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#using mini-batch
kmini = MiniBatchKMeans(n_clusters=true_k, init='k-means++',
        n_init=1, init_size=1000, batch_size=1000, verbose=1) #code changed
# we are using the same test,train data in TFIDF form as we did in text classification
km_model=km.fit(X_train)
kmini_model=kmini.fit(X_train)
print("For K-mean clustering ")
clustering = collections.defaultdict(list)
for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
print("For K-mean Mini batch clustering ")
clustering = collections.defaultdict(list)
for idx, label in enumerate(kmini_model.labels_):
        clustering[label].append(idx)

#Topic Modeling

#LDA (Latent Dirichlet Allocation) and LSI (Latent Semantic Indexing)
from gensim import corpora, models, similarities
from itertools import chain
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import re
documents = [document for document in sms_data]
stoplist = stopwords.words('english')
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
#covert documents to a BOW model and then to a TF_IDF corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
#use LDA / LSI
si = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
n_topics = 5
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
#print terms related to the topic
for i in range(n_topics):#code changed
    temp = lda.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term[0])
    print("Top 10 terms for topic #" + str(i+1) + ": "+ ", ".join(terms))

# Chap 8
# introducing numpy
import numpy as np

# 1-D array
x=[1,2,5,7,3,11,14,25]
np_arr=np.array(x)
print(np_arr)
# 2-D array
arr=[[1,2],[13,4],[33,78]]
np_2darr= np.array(arr)
print(type(np_2darr))
print(np_2darr.tolist())
np_2darr[:]
np_2darr[:2]
np_2darr[:1]
np_2darr[2]
np_2darr[2][0]
np_2darr[:-1]
# generate a numeral series from 0 to 1 with a step size of 0.1
np.arange(0.0, 1.0, 0.1)
# generate unit / zero matrix
np.ones([2, 4])
np.zeros([3,4])
# linespace returns number samples which are evenly spaced
np.linspace(0, 2, 10)
# logspace returns numbers spaced evenly on a log scale.
np.logspace(0,1)
# this returns 50 evenly spaced numbers from 10**0 to 10**1

B = np.array([n for n in range(4)])
print(B)
less_than_3 = B<3 # we are filtering the items that are less than 3.
less_than_3
# array([ True,  True,  True, False])
B[less_than_3] = 0
B
# array([0, 0, 0, 3])

# get the diagonal of a matrix
A = np.array([[0, 0, 0],
              [0, 1, 2],
              [0, 2, 4],
              [0, 3, 6]])
print(np.diag(A))
# [0 1 4]

# matrix multiplication
A = np.array([[1,2],[3,4]])
print(A*A)

# dot product
print(np.dot(A,A))

A-A
A+A
np.transpose(A)
# another way of transposition
A.T

# convert ndarrays to matrices
M=np.matrix(A)
M
np.conjugate(M)
# binary negation (invert)
np.invert(M)

# 1-D, 10-elements random values
N = np.random.randn(1,10)
N
N.mean()
N.std()

(r, c) = A.shape # r is rows and c is columns
r,c
A.reshape(1, r*c)
A.flatten()
np.repeat(A, 2)
# array([1, 1, 2, 2, 3, 3, 4, 4])
np.tile(A, 4)
# array([[1, 2, 1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4, 3, 4]])
B = np.array([[5, 6]])
np.concatenate((A, B), axis=0)
np.vstack((A, B))
np.concatenate((A, B.T), axis=1)

from numpy import random
random.rand(2,5)
# if you want random values to be normally distributed
random.randn(2, 5)

# SciPy
import scipy as sp
# calculate the integral of f(x)=x in the interval [0,1]
# and an "absolute error" estimate for an approximate estimation
from scipy.integrate import quad, dblquad, tplquad
def f(x):
    return x
x_lower=0
x_upper=1
val, abserr = quad(f, x_lower, x_upper)
print(val, abserr)

# linear algebra beginning
import numpy as np
from scipy import linalg as LA
A=np.random.rand(2,2)
B=np.random.rand(2,2)
X=LA.solve(A,B)
print(X)
print(np.dot(A,B))

# finding eigenvalues and eigenvectors
# Say A is our matrix, and there exists a vector v such that Av=λv.
# In this case, λ will be our eigenvalue and v will be our eigenvector.
evals = LA.eigvals(A)
print(evals)
evals, evect=LA.eig(A)
print(evals, evect)

# some other matrix operations
print(LA.inv(A))
print(LA.det(A)) # determinant of matrix A

# storing sparse matrice using the CSR format
from scipy import sparse as s
A = np.array([[1,0,0],[0,5,0],[0,0,3]])
A
from scipy import sparse as spa
C = spa.csr_matrix(A)
C
C.toarray() # 2 ways of re-converting to non-CSR representations
C.todense()

# find the minima of a function
def f(x):
    return x**2-4
sp.optimize.fmin_bfgs(f,0) # the function, and a first estimations
# finding the f(x)=0 solution
sp.optimize.fsolve(f,0.2)
# array([2.]) not -2 because initial estimation 0.2>0
# polynomial
def f1(x,y):
    return x**2+y**2-4
sp.optimize.fsolve(f1, 0, 2)

# Pandas
import pandas as pd
data = pd.read_csv('F:/TextAnalyses_data/iris.data', header=None)
data.head()
# add headers
data = pd.read_csv("F:/TextAnalyses_data/iris.data", names=["sepal length",
"sepal width", "petal length", "petal width", "Cat"], header=None)
data.head()
# see the descriptive statistics of the dataframe
data.describe()
# value counts of sepal length
sepal_len_cnt=data['sepal length'].value_counts()
sepal_len_cnt
data['Cat'].value_counts()
# this returns a column of True and False values
data['Cat']=='Iris-setosa'
# filter the dataframe
sntsosa=data[data['Cat'] == 'Iris-setosa']
sntsosa[:5]

# series data: dealing data with dates
stockdata = pd.read_csv(
    'F:\\TextAnalyses_data\\dow+jones+index\\dow_jones_index.data',
    parse_dates=['date'], index_col=['date'], nrows=100
    )
stockdata.head()
max(stockdata['volume'])
max(stockdata['percent_change_price'])
stockdata.index
stockdata.index.day
stockdata.index.month
stockdata.index.year
# get monthly sums
import numpy as np
stockdata.resample('M').sum()
# remove a column
stockdata.drop(["percent_change_volume_over_last_wk"],axis=1)
# reserve only columns specified
stockdata_new = pd.DataFrame(stockdata, columns=["stock","open","high"
    ,"low","close","volume"])
stockdata_new.head()
# set all values of 'previous_week_volume' to 0
stockdata["previous_weeks_volume"] = 0
# remove Na values
stockdata.dropna().head(2)
# show descriptions
stockdata_new.open.describe()
# remove $ symbols
stockdata_new.open = stockdata_new.open.str.replace('$', '').astype(str).astype(float)
# some arithmetics
stockdata_new['newopen'] = stockdata_new.open.apply(lambda x: 0.8 * x)

# visualization of data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# make one more frame for the company CSCO
stockCSCO = stockdata_new.query('stock=="CSCO"')
stockCSCO.head()
from matplotlib import figure
plt.figure()
plt.scatter(stockdata_new.index.date,stockdata_new.volume)
plt.xlabel('day') # added the name of the x axis
plt.ylabel('stock close value') # add label to y-axis
plt.title('title') # add the title to your graph
plt.savefig("F:\\TextAnalyses_data\\matplot1.jpg") # savefig in local
# subplots
stockAA = stockdata_new.query('stock=="AA"')
plt.subplot(2, 2, 1)
plt.plot(stockAA.index.isocalendar().week, stockAA.open, 'r--')
plt.subplot(2, 2, 2)
plt.plot(stockCSCO.index.isocalendar().week, stockCSCO.open, 'g-*')
plt.subplot(2, 2, 3)
plt.plot(stockAA.index.isocalendar().week, stockAA.open, 'g--')
plt.subplot(2, 2, 4)
plt.plot(stockCSCO.index.isocalendar().week, stockCSCO.open, 'r-*')
plt.savefig("F:\\TextAnalyses_data\\matplot2.png")
# alternative
x = stockAA.index.isocalendar().week
y = stockAA.open
fig, axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
# add axes
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.plot(x, y, 'r')
# more options
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(stockAA.index.isocalendar().week,stockAA.open,label="AA")
ax.plot(stockAA.index.isocalendar().week,stockCSCO.open,label="CSCO")
ax.set_xlabel('weekofyear')
ax.set_ylabel('stock value')
ax.set_title('Weekly change in stock price')
ax.legend(loc=2); # upper left corner
plt.savefig("F:\\TextAnalyses_data\\matplot3.jpg")
# scatter plot
plt.scatter(stockAA.index.isocalendar().week,stockAA.open)
plt.close()
# bar plot
n=12
X=np.arange(n)
Y1 = np.random.uniform(0.5, 1.0, n)
Y2 = np.random.uniform(0.5, 1.0, n)
plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
plt.show()

# Social Media Mining
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import sys
consumer_key = 'PHG9tkvUpVdCLHu1uiQFAA'
consumer_secret = 'dqpNZnLTwteX1YGnQ0VQ3Fv26ensEFeaS8MnQDE'
access_token = '38744894-0TBlSZlcuDE5Sm1Vl6VqZXGVYH9Yn63e9ZM8v7ei'
access_token_secret ='g6ElhezlPulcrPzM1DyqqjXMH25EDeJncHaxvQeuo'
class StdOutListener(StreamListener):
    def on_data(self, data):
        with open(sys.argv[1],'a') as tf:
            tf.write(data)
        return
    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=['Apple watch'])

# Data extraction
import json
import sys
with open('F:\\TextAnalyses_data\\example_tweets.json', 'r', encoding='utf-8') as file:
    tweets = json.load(file)
tweet_texts = [tweet['text'] for tweet in tweets]
tweet_source = [tweet ['source'] for tweet in tweets]
tweet_geo = [tweet['geo'] for tweet in tweets]
tweet_locations = [tweet['place'] for tweet in tweets]
hashtags = [tweet['entities']['hashtags']
            for tweet in tweets]

# find trending topics
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk import FreqDist
tweets_tokens = []
for tweet in tweet_texts:
    tweets_tokens.extend(word_tokenize(tweet))
Topic_distribution = nltk.FreqDist(tweets_tokens)
Topic_distribution.plot(5, cumulative=False)
plt.savefig('F:\\TextAnalyses_data\\example_tweets.png')
plt.show()

# topic modeling with POS,
# because nouns are more likely to be topics
Topics = []
for tweet in tweet_texts:
    tagged = nltk.pos_tag(word_tokenize(tweet))
    Topics_token = [word for word,pos in tagged
                    if pos in ['NN','NNP']]
print(Topics_token)

# Influencer detection
klout_scores = [(tweet['user']['followers_count']/
                  tweet['user']['friends_count'],
                  tweet['user'])
                 for tweet in tweets]

# Facebook mining
import facebook
import json
fo = open("F:\\TextAnalyses_data\\fdump.txt",'w')
ACCESS_TOKEN = 'XXXXXXXXXXX'
fb = facebook.GraphAPI(ACCESS_TOKEN)
company_page = "326249424068240"
content = fb.get_object(company_page)
fo.write(json.dumps(content))
# example for searching
fb.request("search", {'q' : 'nitin', 'type' : 'user'})
fb.request("search", {'q' : 'starbucks', 'type' : 'place'})
fb.request("search", {'q' : 'Stanford university', 'type' : page})
fb.request("search", {'q' : 'beach party', 'type' : 'event'})
# find influencers
friends = fb.get_connections("me", "friends")["data"]
print(friends)
for frd in friends:
    print(fb.get_connections(frd["id"], "friends"))

# big data - text classification with pyspark
from pyspark import SparkContext
from pyspark.sql import Row
import os
print(os.environ.get('JAVA_HOME'))
# if None:
# download java8+ and openjdk eclipse adoptium 11+
os.environ['JAVA_HOME'] = \
    'C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot'  # 替换为你的路径
from pyspark import SparkConf
conf = SparkConf()
conf.set("spark.executor.memory", "2g")
conf.set("spark.driver.memory", "2g")
sc = SparkContext(conf=conf, appName="comment_classifcation")
# sc = SparkContext(appName="comment_classifcation")
lines = sc.textFile("F:\\TextAnalyses_data\\testcomments.txt")
parts = lines.map(lambda l: l.split("\t"))
corpus = parts.map(lambda row: Row(id=row[0], comment=row[1], clas=row[2]))
# 'class' changed to avoid key word collision error
comment = corpus.map(lambda row: " " + row.comment)
class_var = corpus.map(lambda row:row.clas)
# tokenization, term frequency, inverse document frequency with pyspark
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
print(os.environ.get('PYSPARK_PYTHON'))
os.environ['PYSPARK_PYTHON'] = 'python'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
comment_tokenized = comment.map(lambda line: line.strip().split(" "))
hashingTF = HashingTF(1000) # to select only 1000 features
comment_tf = hashingTF.transform(comment_tokenized)
comment_idf = IDF().fit(comment_tf)
comment_tfidf = comment_idf.transform(comment_tf)