#NOTE: Any calls to functions must be done at end of program, this program does not contain
#a main function.
import math
import os
import sys
import time
import copy

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

#used to test time program takes to run
#timestart = time.time()

#this bit of code is to read in the debate file for tokenization
filename = './debate.txt'
file = open(filename, "r", encoding = 'UTF-8')
doc = file.read().lower()
file.close()

#This is to read in the file with each paragraph being split for tfidf
with open(filename,"r") as lines:
    paragraphs = lines.read().lower().split("\n\n")
lines.close()

numParagraphs = len(paragraphs)
   
#split document into tokens
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
tokens = tokenizer.tokenize(doc)

#get list of stopwords
stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()

#create new array for tokens with stop words removed and stemmed
tokensFinal = []
for i in tokens:
    if i not in stopWords:
        tokensFinal.append(stemmer.stem(i))

#tokenize each word in each string of paragraphs to make calculations easier
partokens=[]
for i in paragraphs:
    partokens.append(tokenizer.tokenize(i))
    
partokensfinal=[]
    
#remove stopwords from paragraphs array
for i in partokens:
    temp = []
    for j in i:
        if j not in stopWords:
            temp.append(stemmer.stem(j))
    partokensfinal.append(temp)

#get rid of duplicate tokens in token list
distinctToken = list(set(tokensFinal))
tfidfdict = []
#create dictionary of tokens for each paragraph
for element in partokensfinal:
    tempdict={}
    for i in distinctToken:
        tempdict.update({i:0.0})
    tfidfdict.append(tempdict)

#Create dictionary for storing idf values
idfval ={}
for i in distinctToken:
    idfval.update({i:0.0})

#This is the code for getting idf vector
for word in idfval:
    docfreq=0
    for elem in partokensfinal:
        if word in elem:
            docfreq += 1
    idfval.update({word: ( math.log10(numParagraphs/docfreq)) })
    
#This is the code for getting TF vector
tfidfdict2 = []
parindex = 0 

for par in tfidfdict:
    tempd = copy.deepcopy(par)
    for word in par:
        count = 0
        for word2 in partokensfinal[parindex]:
            if word2 == word:
                count +=1
        if count !=0:
            tempd.update({word: (1+math.log10(count))})
       
    tfidfdict2.append(tempd)
    parindex +=1
       
#getidf function, returns idf of token
def getidf(token):
    val=0.0
    if(idfval.get(token) != None):
        val=idfval.get(token)
    else:
        return -1

    return val

#gettfidf function gets normalized tfidf vectors  
def gettfidf():
    finaltfidf=[]
    mean = []
    ind=0
    for par in tfidfdict2:
        k=0
        tempd = copy.deepcopy(par)
        for word in par:
            tempd.update({word: tempd.get(word) * getidf(word)})
            k+=(tempd.get(word)**2)
        mean.append(k)
        finaltfidf.append(tempd)
    for par in finaltfidf:
        for word in par:
            if mean[ind]!=0:
                par.update({word: (par.get(word) / math.sqrt(mean[ind]))})
        ind+=1
    return finaltfidf   

#create tfidf vector to use globally
tfidf=gettfidf()

#getqvec computes the querty vector for qstring    
def getqvec(qstring):
    qstring = qstring.lower()
    wordsfinal=[]
    querydict={}
    sum =0
    words = tokenizer.tokenize(qstring)
    for i in words:
        if i not in stopWords:
            wordsfinal.append(stemmer.stem(i))
            if getidf(stemmer.stem(i)) > 0:
                querydict.update({stemmer.stem(i): getidf(stemmer.stem(i))})
                sum += getidf(stemmer.stem(i))**2
            else:
                querydict.update({stemmer.stem(i): math.log10(numParagraphs)})
                sum += math.log10(numParagraphs)**2
    div=math.sqrt(sum)

    for i in querydict:
        querydict.update({i: querydict.get(i) / div })

    return querydict 

#query computes the cosine similarity between qstring and paragraphs and returns highest one
def query(qstring):
    #get query vector
    qvec = getqvec(qstring)
    similarityValues = []
    #loop through tfidf vectors
    for par in tfidf:
        temp = 0
        for word in qvec:
            if (par.get(word) != None):
                temp += (qvec.get(word) * par.get(word))     
        similarityValues.append(temp)
        
    if(max(similarityValues) != 0):
        p = paragraphs[similarityValues.index(max(similarityValues))] + ("\n")
    else:
        p = "No Match\n"
    return p,max(similarityValues)


def val(d):
    s= 0
    for i in d:
        s += d.get(i)**2
    s=math.sqrt(s)
    return s
    
#test print statements
#print("%.4f" % getidf("hispanic"))
#print("%.4f" % getidf(stemmer.stem("oil")))
#print(getqvec("vector entropy"))
#print("%.4f" % getidf(stemmer.stem("immigration")))
#print("%s%.4f" % query("unlike any other time, it is under attack"))
#print("--- %.4s seconds ---" % (time.time()-timestart))
