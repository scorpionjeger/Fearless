import os
import pandas as pd
import json
from datetime import datetime
import numpy as np
import dill
import pprint
import emoji
import re
from html.parser import HTMLParser
import itertools
import string
from autocorrect import Speller
spell = Speller(lang='en')
import nltk

# download the stopwords from nltk using
#nltk.download('stopwords')
# import stopwords
from nltk.corpus import stopwords
from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import MLE, WittenBellInterpolated

# import english stopwords list from nltk
stopwords_eng = stopwords.words('english')


#https://spacy.io/universe/project/spacy-langdetect
import spacy

from langdetect import DetectorFactory, detect, detect_langs


#from operator import and_, or_, contains

punhash={"#","@",".","?","!",'"',")"}

#from https://www.geeksforgeeks.org/python-efficient-text-data-cleaning/
#open the fle slang.txt
file=open("slang.txt","r")
slang=file.read()
slang = slang.split('\n')

slang_word = []
meaning = []


for line in slang:
    temp=line.split("=")
    slang_word.append(temp[0])
    meaning.append(temp[-1])



model = dill.load(open("modelTSLyrics" + ".pkd", "rb"))


def WhichLanguage(text):
        if len(text.split(" ")) > 4:
            try:
                return detect(text)
            except:
                return "None"
        else:
            return "None"




def TaylorQuoted(text):
    #test_text = re.sub(r'[^\w\s]', "", filteredTweets[a])
    test_text=text
    n = 4
    # Tokenize and pad the text
    testing_data = list(pad_sequence(word_tokenize(test_text), n,
                                     pad_left=True,
                                     left_pad_symbol="<s>"))
    scores = []
    scorespeices = []
    itemstodel=set()

    for i, item in enumerate(testing_data[n - 1:]):
        #print(i,item)
        s = model.score(item, testing_data[i:i + n - 1])
        scores.append(s)
        if s > .85:

            itemstodel.update(range(i,i+n))
            #print(range[i,i+n])
            #itemstodel.update([j for j in range[i,i+n]])
            scorespeices.append((item,testing_data[i:i + n - 1]))

    try:
        mxscore = max(scores)
    except:
        mxscore = 0
    # if mxscore>.7:
    if len([f for f in scores if f > .7]) > 1:
        print(itemstodel)
        print(mxscore, len([f for f in scores if f > .7]))
        print(scores)
        for b in scorespeices:
            #print(" ".join(b))
            print(b)
        print(text)
        for ji in sorted(list(itemstodel), reverse=True):
            testing_data.pop(ji)
        print(" ".join(testing_data[n - 1:]))
        print(testing_data)


def cleanUp(TotTweText):
    ### dictionary of tweets with an identifier as key
    filteredTweets = {}
    tally = 0
    for a in TotTweText.keys():

        meat=TotTweText[a]
        emoji_list = [c for c in meat if c in emoji.UNICODE_EMOJI_ENGLISH]

        clean_text = ' '.join([str for str in meat.split() if not any(i in str for i in emoji_list)])

        clean_text = HTMLParser().unescape(clean_text)

        clean_text = re.sub(r'http\S+', '', clean_text)

        #hashtage cleaning
        clean_text=clean_text.strip()
        clean_text_List=clean_text.split(" ")

        clean_text=" ".join(clean_text_List)

        clean_text = re.sub(r'@taylorswift13', 'Taylor Swift', clean_text)
        clean_text = re.sub(r'@taylornation13', '', clean_text)

        clean_text = re.sub(r'^RT[\s]+', '', clean_text)
        clean_text = re.sub(r'@RT', '', clean_text)

        clean_text = re.sub(r'(@[A-Za-z0–9]+)', '', clean_text)
        clean_text = re.sub(r'(#[A-Za-z0–9]+)', '', clean_text)

        #from https://www.geeksforgeeks.org/python-efficient-text-data-cleaning/
        Apos_dict = {"'s": " is", " n't": " not", "'m": " am", "'ll": " will",
                     "'d": " would", "'ve": " have", "'re": " are"}

        for key, value in Apos_dict.items():
            if key in clean_text:
                clean_text = clean_text.replace(key, value)

        clean_text = clean_text.lower()

        tweet_tokens = clean_text.split(" ")

        for i, word in enumerate(tweet_tokens):
            if word in slang_word:
                idx = slang_word.index(word)
                tweet_tokens[i] = meaning[idx]

        clean_text = " ".join(tweet_tokens)
        clean_text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(clean_text))

        filteredTweets[a]=clean_text

    return filteredTweets



if __name__=="__main__":
    print("clean")