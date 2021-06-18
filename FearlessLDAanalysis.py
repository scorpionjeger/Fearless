import os
import pandas as pd
import json
from datetime import datetime
#import matplotlib.pyplot as plt
import numpy as np
import dill
import pprint
import emoji
import re
from html.parser import HTMLParser
import itertools
import string
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['taylor','swift','fearless','version','album','Taylor','Swift','Fearless','Version','Album','taylors','Taylors'])#,'love','good','song'])
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import random
# spacy for lemmatization
import spacy
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

mnum=7
#namest="Total50th_bigr1_moreSW"
#tlnum=80
#namest=str(tlnum)+"Fisrttrial"
#mnum=30
#namest="Taylor Swift"
#day=12
#namest="time_slice_"+str(day)

namest="top50000rt"
#text prep
stage1=0
# LDA
stage2=0
# Visualization
stage3=0

stage4=0
stage5=1

def main():
    namestring=["taylorswift13","TaylorsVersion","FromTheVault","Taylor Swift"]
    #filteredTweets=dill.load(open("Total_filteredTweets.pkd","rb"))
    #filteredTweets = dill.load(open("filteredTweetsHighWCount.pkd", "rb"))
    #filteredTweets = dill.load(open("filteredTweetsSub_lt5.pkd", "rb"))
    #filteredTweets = dill.load(open("filteredTweetsSub.pkd", "rb"))

    #filteredTweets = dill.load(open("timeSlice_text_"+str(day) + ".pkd" , "rb"))

    filteredTweets = dill.load(open("TopRT_text.pkd", "rb"))

    #a lot from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#:~:text=Topic%20Modeling%20is%20a%20technique%20to%20extract%20the,of%20topics%20that%20are%20clear%2C%20segregated%20and%20meaningful.
    print(len(filteredTweets))

    if stage1:
        masterlist=[filteredTweets[a] for a in filteredTweets]
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

        data_words = list(sent_to_words(masterlist))

        print(data_words[0:100])

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=1, threshold=1) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See trigram example
        #print(trigram_mod[bigram_mod[data_words[0]]])


        # Define functions for stopwords, bigrams, trigrams and lemmatization
        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            """https://spacy.io/api/annotation"""
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent))
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)



        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        print(data_lemmatized[:1])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # View
        print(corpus[:1])
        dill.dump(data_lemmatized, open(namest+ "filteredTweets_data_lemmatizedSW.pkd", "wb"))
        dill.dump(corpus, open(namest + "filteredTweets_corpusSW.pkd", "wb"))
        dill.dump(id2word, open(namest + "filteredTweets_id2wordSW.pkd", "wb"))

    if stage2:
        corpus = dill.load(open(namest + "filteredTweets_corpusSW.pkd", "rb"))
        id2word = dill.load(open(namest + "filteredTweets_id2wordSW.pkd", "rb"))
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=mnum,
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)

        # Print the Keyword in the 10 topics
        print("here")
        print(lda_model.print_topics())
        dill.dump(lda_model, open(namest + "filteredTweets_lda_model"+str(mnum)+"SW.pkd", "wb"))


    if stage3:
        lda_model = dill.load(open(namest + "filteredTweets_lda_model"+str(mnum)+"SW.pkd", "rb"))

        data_lemmatized = dill.load(open(namest + "filteredTweets_data_lemmatizedSW.pkd", "rb"))
        corpus = dill.load(open(namest + "filteredTweets_corpusSW.pkd", "rb"))
        id2word = dill.load(open(namest + "filteredTweets_id2wordSW.pkd", "rb"))

        for a in lda_model.print_topics():
            print(a)
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        if 1:
        # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print('\nCoherence Score: ', coherence_lda)

        # Visualize the topics

        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(vis, namest+'LDA_Visualization'+str(mnum)+'SW.html')


    if stage4:
        lda_model = dill.load(open(namest + "filteredTweets_lda_model"+str(mnum)+"SW.pkd", "rb"))
        corpus = dill.load(open(namest + "filteredTweets_corpusSW.pkd", "rb"))
        masterlist = [filteredTweets[a] for a in filteredTweets]
        def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=masterlist):
            # Init output
            sent_topics_df = pd.DataFrame()

            # Get main topic in each document
            ldarows=[]
            for i, row in enumerate(ldamodel[corpus]):
                row0=row[0]
                row0 = sorted(row0, key=lambda x: (x[1]), reverse=True)
                ldarows.append(row0)
                # Get the Dominant topic, Perc Contribution and Keywords for each document
                for j, (topic_num, prop_topic) in enumerate(row0):
                    if j == 0:  # => dominant topic
                        wp = ldamodel.show_topic(topic_num)
                        if i % 1000 == 0:
                            print(i,prop_topic,topic_num,wp)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(
                            pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                    else:
                        break
            sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

            # Add original text to the end of the output
            contents = pd.Series(texts)
            sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
            return (sent_topics_df,ldarows)

        df_topic_sents_keywords,ldarows = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=masterlist)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        # Show
        df_dominant_topic.head(10)
        dill.dump(df_dominant_topic, open(namest + "filteredTweets_df_dominant_topic"+str(mnum)+"SW.pkd", "wb"))
        dill.dump(df_topic_sents_keywords, open(namest + "filteredTweets_df_topic_sents_keywords"+str(mnum)+"SW.pkd", "wb"))
        dill.dump(ldarows,open(namest + "filteredTweets_df_topic_rawrows"+str(mnum)+"SW.pkd", "wb"))

    if stage5:
        lda_model = dill.load(open(namest + "filteredTweets_lda_model"+str(mnum)+"SW.pkd", "rb"))
        corpus = dill.load(open(namest + "filteredTweets_corpusSW.pkd", "rb"))
        df_topic_sents_keywords = dill.load(open(namest + "filteredTweets_df_topic_sents_keywords"+str(mnum)+"SW.pkd", "rb"))
        #print(df_dominant_topic.head(40))
        print(df_topic_sents_keywords.describe())
        sent_topics_sorteddf_mallet = pd.DataFrame()
        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
        print("sent_topics_outdf_grpd",len(sent_topics_outdf_grpd))

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                     grp.sort_values(['Perc_Contribution'], ascending=[0]).head(15)],
                                                    axis=0)

        # Reset Index
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

        # Format
        sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

        # Show
        print(sent_topics_sorteddf_mallet.keys())
        print(len(sent_topics_sorteddf_mallet))
        outputlist=[]
        for a in range(len(sent_topics_sorteddf_mallet['Topic_Num'])):
            print(sent_topics_sorteddf_mallet['Topic_Num'][a], sent_topics_sorteddf_mallet['Text'][a])
            outputlist.append((sent_topics_sorteddf_mallet['Topic_Num'][a], sent_topics_sorteddf_mallet['Text'][a]))
        #for a in lda_model.print_topics():
        #    print(a)
        dill.dump(outputlist,open(namest + "outputlist_"+str(mnum)+"SW.pkd", "wb"))


if __name__ == "__main__":
    main()