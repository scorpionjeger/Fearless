import os
import pandas as pd
import json
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import numpy as np
import dill
import collections
import random
import pprint
import Tweetcleaningbase
from nltk.probability import FreqDist
import nltk
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
tokenizer = nltk.RegexpTokenizer(r"\w+")
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from scipy.ndimage.filters import uniform_filter1d






#GoThoughTweets extracts relevant information from the raw tweet data given from twitter
#returns several dictionaries
def GoThoughTweets(theFiles,names,TweMetadata,TweText,ReTweText,IDMetaData,ReTweMetaData):
    reT=0
    totalbreak=0

    for i in range(len(theFiles)):
        if totalbreak==1:
            break
        with open(Path+"/"+theFiles[i]) as f:
            data=json.load(f)


        checkdate=0
        for a in data.keys():
            thetwitdate=data[a]['created_at']
            thedate=datetime.strptime(thetwitdate,'%a %b %d %H:%M:%S +0000 %Y')
            if checkdate==0:
                checkdate=1
            if thedate<=datetime(2021, 4, 5, 0, 0, 0):
                break
            elif thedate >= datetime(2021, 4, 26, 0, 0, 0):
                totalbreak=1
                break
            try:
                TweMetadata[a]
                continue
            except:
                TweMetadata[a]={}


            TweMetadata[a]['date']=thedate

            try:
                data[a]['retweeted_status']
                reT=1

            except:
                if data[a]['in_reply_to_screen_name'] == None:
                    TweMetadata[a]['type']="Orig"
                    Fulltxt = data[a]['full_text']
                    retweetcount=0
                else:
                    TweMetadata[a]['type']="Reply"
                    Fulltxt = data[a]['full_text']
                    retweetcount = 0



                    #if int(data[a]['created_at'].split(" ")[2]) >= 7 and int(data[a]['created_at'].split(" ")[2]) <= 11:
            if reT==1:
                #retweetDict[a] = thedate
                TweMetadata[a]['type']="Retweet"
                TweMetadata[a]['RetweetID'] = data[a]['retweeted_status']['id']
                Fulltxt=data[a]['retweeted_status']['full_text']
                ReTweText[data[a]['retweeted_status']['id']] = Fulltxt
                reT=0
                retweetcount = data[a]['retweet_count']
                try:
                    ReTweMetaData[data[a]['retweeted_status']['id']]["count"]+=1
                    ReTweMetaData[data[a]['retweeted_status']['id']]["date"].append(thedate)
                    ReTweMetaData[data[a]['retweeted_status']['id']]["userID"].add(data[a]['user']['id'])
                except:
                    ReTweMetaData[data[a]['retweeted_status']['id']]={}
                    ReTweMetaData[data[a]['retweeted_status']['id']]["count"] = 1
                    ReTweMetaData[data[a]['retweeted_status']['id']]["date"]=[thedate]
                    ReTweMetaData[data[a]['retweeted_status']['id']]["userID"]=set()
                    ReTweMetaData[data[a]['retweeted_status']['id']]["userID"].add(data[a]['user']['id'])
                    ReTweMetaData[data[a]['retweeted_status']['id']]["origUserID"]=data[a]['retweeted_status']['user']['id']
                #print(data[a]['retweeted_status'])
            else:
                TweText[a] = Fulltxt
                retweetcount =0
            #subDict[a]=thedate


            TweMetadata[a]['retweet_count']=retweetcount
            TweMetadata[a]['user_id']=data[a]['user']['id']
            TweMetadata[a]['SearchText']=names
            if 1:
                IDMetaData[data[a]['user']['id']]={}
                IDMetaData[data[a]['user']['id']]['location']=data[a]['user']['location']
                IDMetaData[data[a]['user']['id']]['name']=data[a]['user']['name']
                IDMetaData[data[a]['user']['id']]['screen_name']=data[a]['user']['screen_name']
                IDMetaData[data[a]['user']['id']]['lang']=data[a]['user']['lang']
                IDMetaData[data[a]['user']['id']]['followers_count']=data[a]['user']['followers_count']
                IDMetaData[data[a]['user']['id']]['friends_count']=data[a]['user']['friends_count']
                IDMetaData[data[a]['user']['id']]['created_at']=data[a]['user']['created_at']
                IDMetaData[data[a]['user']['id']]['description']=data[a]['user']['description']
                IDMetaData[data[a]['user']['id']]['favourites_count']=data[a]['user']['favourites_count']
                IDMetaData[data[a]['user']['id']]['entities']=data[a]['user']['entities']





def CategorizeTweets(TweMetadata):
    ID="A5_26"
    if 1:
        RetweetList=[]
        replyList=[]
        OriginalList=[]

        ainc = 0
        for a in TweMetadata.keys():
            if ainc % 5000 == 0:
                print(ainc)
            ainc += 1
            thedate = TweMetadata[a]['date']
            thetype=TweMetadata[a]['type']
            if thetype=="Orig":
                OriginalList.append(thedate)
            elif thetype=="Retweet":
                RetweetList.append(thedate)
            elif thetype == "Reply":
                replyList.append(thedate)


    if 1:
        DatesCollectionOrg=np.array(OriginalList)
        DatesCollectionrep=np.array(replyList)
        DatesCollectionreT=np.array(RetweetList)



        dill.dump(DatesCollectionOrg, open(ID + "3WOrig.pkd", "wb"))
        dill.dump(DatesCollectionrep, open(ID + "3WReply.pkd", "wb"))
        dill.dump(DatesCollectionreT, open(ID + "3WRetweet.pkd", "wb"))




def find_randomized_subset_of_users(TweMetadata):
    binnumber=1200
    #168
    totalnumber=len(TweMetadata)
    #4603
    incNumber=int(np.floor(totalnumber/binnumber))

    idwithdate={}
    for a in TweMetadata:
        idwithdate[a]=TweMetadata[a]['date']
    idwithdatelist=list(idwithdate.items())
    print("now to sort")
    idwithdatelist.sort(key=lambda x:x[1])
    print(len(idwithdatelist)/168.0)
    ninc=0
    sortDict={}
    for i in range(binnumber):
        sortDict[i] = []
        for j in range(incNumber):
            sortDict[i].append(idwithdatelist[j+i*incNumber])
    randomsubset={}
    for a in sortDict:
        randomsubset[a]=random.sample(sortDict[a],10)

    if 1:
        dill.dump(randomsubset, open("randomsubsetF" + ".pkd", "wb"))
        dill.dump(sortDict, open("sortDictF" + ".pkd", "wb"))



def find_randomized_subset_of_users2(TweMetadata,sortDict):
    uqID = dill.load(open("uqID10" + ".pkd", "rb"))
    randomsubset={}
    #uqID=list(userText.keys())
    print(len(uqID))
    totinc = 0
    for a in sortDict:
        sortArand=random.sample(sortDict[a],len(sortDict[a]))
        randomsubset[a]=[]
        inc=0

        for j in sortArand:
            if inc>=10:
                break
            if TweMetadata[j[0]]['user_id'] not in uqID:
                randomsubset[a].append(j)
                uqID.append(TweMetadata[j[0]]['user_id'])
                inc+=1
                totinc += 1
            else:
                None
                totinc += 1
                #print("year")
        #print(len(randomsubset[a]))
    print("last",len(uqID)-1632,totinc)


    dill.dump(randomsubset, open("randomsubset12" + ".pkd", "wb"))
    dill.dump(uqID, open("uqID12" + ".pkd", "wb"))





#createUserDataFiles cleans up the text taken from the 50 most recent tweets of
#twitter users in this study and returns a dictionary with user ids as keys and a list
#of cleaned tweets representing the 50 most recent tweets of the user.
def createUserDataFiles():
    lablelist=[1,2,3,4,5,6,7,10]

    inc = 0
    Total_user_text={}
    for f in lablelist:
        print(f)
        userText = dill.load(open("userText"+str(f) + ".pkd", "rb"))
        for a in userText:

            UserTweet = {}
            try:
                for b in userText[a]["Text List"]:
                    UserTweet[str(a) + "_" + str(b[0])] = b[1]
            except:
                continue
            if inc == 5:
                break
            #inc += 1
            filteredTweets = Tweetcleaningbase.cleanUp(UserTweet)
            Total_user_text[a] =[filteredTweets[c] for c in filteredTweets if Tweetcleaningbase.WhichLanguage(filteredTweets[c])=="en"]

    dill.dump(Total_user_text, open("Total_user_text" + ".pkd", "wb"))



#CleanUserDescription passes the twitter self disclosed description though the cleaning
#process. Can return a list of descriptions. and can update IDMetaData
def CleanUserDescription(IDMetaData):

    userDescripDic={a:IDMetaData[a]["description"] for a in IDMetaData}

    FilterduserDescripDic = Tweetcleaningbase.cleanUp(userDescripDic)
    for a in FilterduserDescripDic:
        IDMetaData[a]["clean description"]=FilterduserDescripDic[a]
    FilterduserDescripDic=[FilterduserDescripDic[a] for a in FilterduserDescripDic if FilterduserDescripDic[a]!=""]
    dill.dump(FilterduserDescripDic, open("FilterduserDescripDicF" + ".pkd", "wb"))
    dill.dump(IDMetaData, open("IDMetaDataF" + ".pkd", "wb"))





def InvestigateDescriptions():
    #Taking in cleaned user descriptions and tolkeizing etc the whole batch
    if 0:
        FilterduserDescripDic = dill.load(open("FilterduserDescripDicF" + ".pkd", "rb"))

        tokens=[]
        for a in FilterduserDescripDic:
            tokens += tokenizer.tokenize(a)

        lmtzr = WordNetLemmatizer()
        # Save the list between tokens
        lemmatized = []
        for word in tokens:
            # Lowerize for correct use in stopwords etc
            w = word.lower()
            # Check english terms
            #if not ENGLISH_RE.match(w):
            #    continue
            # Check stopwords
            if w in stopwords.words('english'):
                continue
            lemmatized.append(lmtzr.lemmatize(w))

        fdist1 = nltk.FreqDist(lemmatized)
        print(fdist1.most_common(300))
        dill.dump(fdist1, open("fdist1F" + ".pkd", "wb"))


    if 0:
        fdist1 = dill.load(open("fdist1F" + ".pkd", "rb"))
        count=0
        for a in fdist1:
            print(a,int(fdist1.freq(a)*fdist1.N()),fdist1.freq(a))
            count+=1
            if count>2000:
                break
        fdistDic={}
        for a in fdist1:
            fdistDic[a]=(int(fdist1.freq(a)*fdist1.N()),fdist1.freq(a))
        dill.dump(fdistDic, open("fdistDicF" + ".pkd", "wb"))

        for a in fdistDic:
            print(a, fdistDic[a])
            if fdistDic[a][0]<20:
                break

    #interactive program to choose relevant words
    if 0:
        fdist1 = dill.load(open("fdistDicFc" + ".pkd", "rb"))
        fdist1c = dill.load(open("fdistDicFc" + ".pkd", "rb"))
        count=0
        try:
            keepdis = dill.load(open("keepdis" + ".pkd", "rb"))
        except Exception:
            keepdis={}
        for a in fdist1:
            print(a, fdist1[a])
            answer = input("keep n delete m break v save c")
            print(answer)
            if answer=="n":
                fdist1c.pop(a)
                keepdis[a]=fdist1[a]
                print("len",len(keepdis))

            if answer == "m":
                fdist1c.pop(a)
            if answer=="c":
                dill.dump(fdist1c, open("fdistDicFc" + ".pkd", "wb"))
                dill.dump(keepdis, open("keepdis" + ".pkd", "wb"))
            if answer=="v":
                dill.dump(fdist1c, open("fdistDicFc" + ".pkd", "wb"))
                dill.dump(keepdis, open("keepdis" + ".pkd", "wb"))
                break
    if 1:
        keepdis = dill.load(open("keepdis" + ".pkd", "rb"))
        for a in keepdis:
            print(a,",",keepdis[a][0])

    if 0:
        fdist1 = dill.load(open("fdistDicF" + ".pkd", "rb"))
        numbcollect=collections.defaultdict(int)
        for a in fdist1:
            if a.isnumeric():
                try:
                    if int(a)<80 and int(a)>7:
                        print(int(a),fdist1[a])
                        numbcollect[int(a)]+=fdist1[a][0]

                except Exception:
                    None
        numbcollectL=list(numbcollect.items())
        numbcollectL.sort(key=lambda x: x[0])
        for a in numbcollectL:
            print(a)


#sub function to show that a word is in a description
def isinKWL(kwl,text):
    isin=False
    for a in kwl:
         if re.search(a,text):
            isin=True
            break
    return isin


#investigative function looking at self described demographics.
def giving_users_categories(IDMetaData):
    kwl={}

    kwl["test"]=["liberty"]
    count=0
    for a in IDMetaData:
        if isinKWL(kwl["test"], IDMetaData[a]["clean description"]):
            print(IDMetaData[a]["clean description"])
            count+=1
        if count>200:
            break


#This returns information for graphing the timeline of self described demographics.
def Fill_timeline_with_demographics(TweMetadata,IDMetaData,sortDict):

    if 1:
        character = {}
        kwl = dill.load(open("catarr3" + ".pkd", "rb"))
        #IDMetaData = dill.load(open("IDMetaDataFkwl" + ".pkd", "rb"))

        #kwl=[a for a in catdic]
        for a in sortDict:
            character[a] = {}
            unique_user_IDs = set(TweMetadata[b[0]]['user_id'] for b in sortDict[a])
            lentot = len(unique_user_IDs)
            for d in kwl:
                character[a][d] = 0
            for c in unique_user_IDs:
                for i,d in enumerate(kwl):
                    character[a][d] += IDMetaData[c]["kwl"][i] / lentot

        dill.dump(character, open("characterFkwl3" + ".pkd", "wb"))




def sentiment(TweMetadata,TweText,ReTweText):
    count=0
    for b in ReTweText:
        timeSlice = {b:ReTweText[b]}
        timeSlice_text = Tweetcleaningbase.cleanUp(timeSlice)
        a=timeSlice_text[b]
        testimonial = TextBlob(a)
        score = SentimentIntensityAnalyzer().polarity_scores(a)
        if score["neg"]>.5:
            count += 1
            if 1:
                print(a)
                print(testimonial.sentiment)

                print(testimonial.sentiment.polarity)

                print(score)
                print(" ")
    print(count)




def TopRetweets(ReTweMetaData,ReTweText):
    if 0:
        ReTweHistos=[]
        ReTweIndex=[]
        Binns = [datetime(2021, 4, y, x, 0, 0) for y in range(5, 26) for x in range(24)]
        print("start")
        for a in ReTweMetaData:

            #print(g, "Name", IDMetaData[ReTweMetaData[countsRT[g][0]]["origUserID"]]["name"])
            #print(ReTweText[countsRT[g][0]])
            hist, bin = np.histogram(ReTweMetaData[a]["date"], Binns)
            # print(hist)
            # print(Binns)
            ReTweHistos.append(hist)
            ReTweIndex.append(a)
        ReTweHistos=np.array(ReTweHistos)
        print(ReTweHistos)
        dill.dump(ReTweHistos, open("ReTweHistos.pkd", "wb"))
        dill.dump(ReTweIndex, open("ReTweIndex.pkd", "wb"))

    if 0:
        ReTweHistos = dill.load(open("ReTweHistos" + ".pkd", "rb"))
        ReTweIndex = np.array(dill.load(open("ReTweIndex" + ".pkd", "rb")))
        print(len(ReTweHistos[0]))
        print(np.transpose(ReTweHistos)[0])
        print(len(np.transpose(ReTweHistos)[0]))
        TRetTweHistos=np.transpose(ReTweHistos)
        inds=np.argsort(TRetTweHistos[200])[0:10]
        print(TRetTweHistos[200][inds])
        inds=np.argsort(-TRetTweHistos[200])[0:5]
        print(TRetTweHistos[200][inds])
        print(inds)
        print(ReTweIndex[inds])

        maxRetweetIDs=np.array([ReTweIndex[np.argsort(-a)[0:5]] for a in TRetTweHistos])
        dill.dump(maxRetweetIDs, open("maxRetweetIDs.pkd", "wb"))
        print(maxRetweetIDs)
        #maxRetweetTxt=[ReTweText[b] for a in maxRetweetIDs for b in a ]
        maxRetweetTxt=[]
        for a in maxRetweetIDs:
            subarr=[]
            for b in a:
                subarr.append(ReTweText[b])
            maxRetweetTxt.append(subarr)

        dill.dump(maxRetweetTxt, open("maxRetweetTxt.pkd", "wb"))
        print(maxRetweetTxt)
    if 1:
        maxRetweetTxt = np.array(dill.load(open("maxRetweetTxt" + ".pkd", "rb")))
        print(maxRetweetTxt)
        maxRetweetTxt = np.transpose(maxRetweetTxt)
        print(maxRetweetTxt[0])


#Classifying key words in user descriptions.
def pandasinv():
    df=pd.read_excel("words.xlsx")
    df = df.replace(np.nan, '', regex=True)
    tryl=list(df.word[df.cat1=="entertainment"][df.cat2=="movies/tv"])
    tryl=[a.strip() for a in tryl]

    #
    toxs=list(df.cat1.unique())
    print(toxs)
    toxs.remove("")
    print(toxs)
    keylist=[]
    for a in toxs:
        if a != "":
            tix=list(df.cat2[df.cat1==a].unique())
            try:
                tix.remove("")
            except Exception:
                None
            if len(tix)!=0:
                for b in tix:
                    taxs=list(df.cat3[df.cat2 == b].unique())
                    try:
                        taxs.remove("")
                    except Exception:
                        None
                    if len(taxs)!=0:
                        for c in taxs:
                            keylist.append(a+"_"+b+"_"+c)
                            print(a,b,c)
                        print(a, b)
                        keylist.append(a + "_" + b)
                    else:
                        print(a,b)
                        keylist.append(a + "_" + b)
                print(a)
                keylist.append(a)

            else:
                print(a)
                keylist.append(a)

    cat=["cat1","cat2","cat3"]
    catdic={}
    print(keylist)
    for t in keylist:
        for i in range(len(t.split("_"))):
            if i==0:
                tryl=df.word[df[cat[i]]==t.split("_")[i]]
            else:
                tryl=tryl[df[cat[i]]==t.split("_")[i]]
        tryl=[a.strip() for a in tryl]
        print(t,tryl)
        if len(tryl)!=0 and t!="ns" and t!="geography":
            catdic[t]=tryl
    print(catdic.keys())

    catarr=[a for a in catdic]


    if 1:
        IDMetaData = dill.load(open("IDMetaDataF" + ".pkd", "rb"))
        lmtzr = WordNetLemmatizer()
        for a in IDMetaData:
            tokens = tokenizer.tokenize(IDMetaData[a]["clean description"])


            # Save the list between tokens
            lemmatized = []
            for word in tokens:
                w = word.lower()

                if w in stopwords.words('english'):
                    continue
                lemmatized.append(lmtzr.lemmatize(w))
            catdicres=[]
            for t in catdic:
                if t=="gender_female":

                    if re.search(r'^(?=.*\bher\b)(?=.*\bshe\b).*$', IDMetaData[a]["clean description"]):
                        catdicres.append(1)
                    else:
                        if len(set(catdic[t]).intersection(set(lemmatized))) != 0:
                            catdicres.append(1)
                        else:
                            catdicres.append(0)
                elif t=="gender_male":
                    if re.search(r'^(?=.*\bhim\b)(?=.*\bhe\b).*$', IDMetaData[a]["clean description"]):
                        catdicres.append(1)
                    else:
                        if len(set(catdic[t]).intersection(set(lemmatized))) != 0:
                            catdicres.append(1)
                        else:
                            catdicres.append(0)
                elif t=="parent_motherhood":
                    if len(set(catdic[t]).intersection(set(lemmatized))) != 0:
                        momstr=IDMetaData[a]["clean description"]
                        found=0
                        for hh in catdic[t]:
                            if "cat "+hh in momstr:
                                momstr=momstr.replace("cat "+hh," ")
                                found=1

                            if "dog "+hh in momstr:
                                momstr = momstr.replace("dog " + hh, " ")
                                found=1

                            if "pug "+hh in momstr:
                                momstr = momstr.replace("pug " + hh, " ")
                                found=1

                            if "plant "+hh in momstr:
                                momstr = momstr.replace("plant " + hh, " ")
                                found=1
                        if found==1:
                            print(IDMetaData[a]["clean description"])
                            print(momstr)
                            tokensm = tokenizer.tokenize(momstr)

                            # Save the list between tokens
                            lemmatizedm = []
                            for word in tokensm:
                                w = word.lower()

                                if w in stopwords.words('english'):
                                    continue
                                lemmatizedm.append(lmtzr.lemmatize(w))
                            if len(set(catdic[t]).intersection(set(lemmatizedm))) != 0:
                                catdicres.append(1)
                                print("people mom")
                            else:
                                catdicres.append(0)
                        else:
                            catdicres.append(1)
                    else:
                        catdicres.append(0)

                else:
                    if len(set(catdic[t]).intersection(set(lemmatized)))!=0:
                        catdicres.append(1)
                    else:
                        catdicres.append(0)
            IDMetaData[a]["kwl"]=catdicres
        dill.dump(IDMetaData, open("IDMetaDataFkwl3" + ".pkd", "wb"))
        dill.dump(catarr, open("catarr3" + ".pkd", "wb"))




def retweetInv(ReTweMetaData,ReTweText):

    retweetcount=[(a,ReTweMetaData[a]["count"]) for a in ReTweMetaData]
    retweetcount.sort(key=lambda x: -x[1])
    timeSlice = {}
    for i in range(len(retweetcount)):
    #for i in range(50000):
        timeSlice[retweetcount[i][0]] = ReTweText[retweetcount[i][0]]


    TopRT_text = Tweetcleaningbase.cleanUp(timeSlice)

    dill.dump(TopRT_text, open("fullRT_text.pkd", "wb"))


def PrepairDataForViualization():
    IDMetaData = dill.load(open("IDMetaDataFkwl3" + ".pkd", "rb"))
    character = dill.load(open("characterFkwl3" + ".pkd", "rb"))
    catarr = dill.load(open("catarr3" + ".pkd", "rb"))
    sortDict = dill.load(open("sortDictF" + ".pkd", "rb"))
    userswDem = [a for a in IDMetaData if sum(IDMetaData[a]["kwl"]) is not 0]
    pdDict = {}
    if 1:

        types = ["Orig", "Retweet"]
        ID = "A5_26"

        for i, typeses in enumerate(types):
            DatesCollection = dill.load(open(ID + "3W" + typeses + ".pkd", "rb"))

            Binns = [datetime(2021, 4, y, x, 0, 0) for y in range(5, 26) for x in range(24)]

            hist, bin = np.histogram(DatesCollection, Binns)

            Binns.pop()

            Binns = np.array(Binns)
            hist = np.array(hist)

            pdDict[typeses + "_hist"] = hist
            pdDict["xBinns"] = Binns - timedelta(hours=4)

    if 1:

        kwl = dill.load(open("catarr3" + ".pkd", "rb"))

        for j, a in enumerate(kwl):

            xdate = []
            ycolum = []
            for i in sortDict:
                xdate.append((sortDict[i][0][1] - timedelta(hours=4)).timestamp())
                ycolum.append(character[i][a])

            xdate = np.array(xdate)
            X_ = [datetime.fromtimestamp(x) for x in xdate]
            ycolum = np.array(ycolum)
            ycolumF = uniform_filter1d(ycolum, 20)

            pdDict[a] = ycolum
        pdDict["catBinns"] = X_

    maxRetweetTxt = np.array(dill.load(open("maxRetweetTxt.pkd", "rb")))
    maxRetweetTxt = np.transpose(maxRetweetTxt)
    pdDict["1st"] = maxRetweetTxt[0]
    pdDict["2nd"] = maxRetweetTxt[1]
    pdDict["3rd"] = maxRetweetTxt[2]

    TotCharHisto = 0
    for a in IDMetaData:
        if type(TotCharHisto)=="int":
            TotCharHisto=np.array(IDMetaData[a]["kwl"])
        else:
            TotCharHisto+=np.array(IDMetaData[a]["kwl"])
    pdDict["TotCharHisto"]=TotCharHisto
    pdDict["catarr"]=catarr

    dill.dump(pdDict, open("plotDict3ns" + ".pkd", "wb"))


    ######################################################################
    ######################################################################
    ######################################################################
if __name__=="__main__":

    #step 1: extract information from raw tweet data
    if 0:
        TweMetadata = {}
        TweText = {}
        ReTweText = {}
        IDMetaData = {}
        ReTweMetaData={}
        count=1
        namestring = ["taylorswift13", "TaylorsVersion", "FromTheVault", "Taylor Swift"]
        for names in namestring:
            print(names)
            Path = "F:/DataScience/TaylorData/" + names
            theFiles = os.listdir(Path)
            GoThoughTweets(theFiles,names,TweMetadata,TweText,ReTweText,IDMetaData,ReTweMetaData)

        if 1:
            dill.dump(ReTweMetaData, open("ReTweMetaDataF" + ".pkd", "wb"))
            dill.dump(TweMetadata, open("TweMetadataF" + ".pkd", "wb"))
            dill.dump(TweText, open("TweTextF" + ".pkd", "wb"))
            dill.dump(ReTweText, open("ReTweTextF" + ".pkd", "wb"))
            dill.dump(IDMetaData, open("IDMetaDataF" + ".pkd", "wb"))

    if 1:
        if 1:

            #TweMetadata={}
            TweText = dill.load(open("TweTextF" + ".pkd", "rb"))
            ReTweText = dill.load(open("ReTweTextF" + ".pkd", "rb"))
            ReTweMetaData = dill.load(open("ReTweMetaDataF" + ".pkd", "rb"))
            #IDMetaData = dill.load(open("IDMetaDataF" + ".pkd", "rb"))
            IDMetaData = dill.load(open("IDMetaDataFkwl3" + ".pkd", "rb"))
            sortDict=dill.load(open("sortDictF" + ".pkd", "rb"))
            character = dill.load(open("characterFkwl3" + ".pkd", "rb"))
            TweMetadata = dill.load(open("TweMetadataF" + ".pkd", "rb"))
        UserTally = dill.load(open("UserTally" + ".pkd", "rb"))

        #step 2:  group tweets into categories for display purposes.

        #CategorizeTweets(TweMetadata)


        #find_randomized_subset_of_users(TweMetadata, TweText, ReTweText, IDMetaData)
        #find_randomized_subset_of_users2(TweMetadata, TweText, ReTweText, IDMetaData,userText,sortDict)


        #investigate twitter user text:
        #createUserDataFiles()

        #investigate the self described user description:
        #CleanUserDescription(IDMetaData)


        #manually find key words:
        #InvestigateDescriptions()

        #for checking what information is in the actual tweets
        #giving_users_categories(IDMetaData)


        #for taking the categories from a spreadsheet and finding which tweet users has them
        # pandasinv()


        #for getting a time sieries of users.
        #Fill_timeline_with_demographics(TweMetadata,IDMetaData,sortDict)


        #give_a_time_period_of_cleaned_tweets(TweMetadata,TweText)

        #sentiment(TweMetadata,TweText,ReTweText)
        #TopRetweets(ReTweMetaData,ReTweText)
        #retweetInv(ReTweMetaData,ReTweText)



        PrepairDataForViualization()