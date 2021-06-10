import numpy as np
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from dotenv import load_dotenv
import dill

###
###

# DatesCollection=dill.load(open(nstring+".pkd","rb"))
types = ["Orig", "Reply", "Retweet"]
types = ["Orig", "Retweet"]
typesL = ["Original", "Retweet"]
types = ["Retweet"]
typesL = ["Retweet"]
ID = "A5_26"
print(ID)
if 0:
    for i, typeses in enumerate(types):
        DatesCollection = dill.load(open(ID + "3W" + typeses + ".pkd", "rb"))
        # Binns=[datetime(2021,4,y,x,0,0)  for y in range(5,13) for x in range(24)]
        # Binns = [datetime(2021, 4, y, x, 0, 0) for y in range(5, 26) for x in range(24)]
        Binns = [datetime(2021, 4, y, x, 0, 0) for y in range(5, 26) for x in range(24)]

        hist, bin = np.histogram(DatesCollection, Binns)
        # print(hist)
        # print(Binns)
        Binns.pop()

        # DatesCollection=np.vectorize(lambda x:x.timestamp())

        # plt.plot(Binns,hist,label=nstring)
        Binns = np.array(Binns)
        hist = np.array(hist)
        #plt.plot(Binns - timedelta(hours=4), hist / 10, label=typesL[i])
        # plt.plot(Binns-timedelta(hours=4), hist/1000000, label=typesL[i])
if 1:
    Binns=range(10)
    hist=[x**2 for x in range(10)]
    #recompiling a dataframe for plotting
    pdDict={}
    #pdDict["Binns"]=Binns - timedelta(hours=4)
    pdDict["hist"]=hist
    pdDict["Binns"]=Binns
    Rpd=pd.DataFrame.from_dict(pdDict)


    #Plotting with altair, bokeh did not work for this example
    if 1:
        c=alt.Chart(Rpd).mark_line().encode(
            alt.X('Binns'),
            #alt.Y('hist', axis=alt.Axis(format='$.2f'))
            alt.Y('hist')
        )
        st.write(c)