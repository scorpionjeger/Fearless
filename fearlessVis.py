import pandas as pd
import plotly.express as px
from bokeh.plotting import figure
from datetime import datetime,timedelta
import numpy as np
import dill
#from streamlit_vega_lite import vega_lite_component, altair_component
import streamlit.components.v1 as components
import streamlit as st
import altair as alt
from scipy.ndimage.filters import uniform_filter1d
from scipy.fft import fft, fftfreq
from scipy import interpolate
from bokeh.palettes import Spectral11
from bokeh.models import Range1d

def PlotTrends2(plotDict,kwl,add_selectbox,smth):

    if 0:
        @st.cache
        def altair_plot():
            single_nearest = alt.selection_single(encodings=["x"], name="single_nearest",on='mouseover', nearest=True)
            #single_nearest = alt.selection_single(on='mouseover', nearest=True)
            return (
                alt.Chart(Rpd).mark_line().encode(
                    alt.X('xBinns'),
                    # alt.Y('hist', axis=alt.Axis(format='$.2f'))
                    alt.Y("Retweet_hist"))
                    .add_selection(single_nearest).interactive()
                        )

        event_dict = altair_component(altair_chart=altair_plot())
        st.write(event_dict)
    if 0:
        c=alt.Chart(Rpd).mark_line().encode(
            alt.X('xBinns'),
            #alt.Y('hist', axis=alt.Axis(format='$.2f'))
            alt.Y("Retweet_hist"),
            alt.Y("Orig_hist"),
            tooltip = ["1st","2nd","3rd"]
        ).interactive()
        st.write(c)
    if 0:
        fig = px.scatter(Rpd,
                         x='xBinns',
                         y="Retweet_hist",
                         hover_name='1st',
                         title='yes')#

        st.plotly_chart(fig)
    if 1:

        p = figure(
             title='simple line example',
             x_axis_label='x',
             y_axis_label='y',
            x_axis_type='datetime')

        x=[]
        y=[]
        maxy=0
        labels=[]
        for a in add_selectbox:
            x.append(plotDict["catBinns"])
            ycolumF = uniform_filter1d(plotDict[a], int(smth))
            y.append(ycolumF)
            maxy=max(maxy,max(ycolumF))
            labels.append(a)
        labels.append("retweets")
        x.append(plotDict["xBinns"])
        y.append(np.array(plotDict["Retweet_hist"])*0.5*maxy/max(plotDict["Retweet_hist"]))

        mypalette = Spectral11[0:len(y)]


        for i in range(len(y)):
            p.line(x[i],y[i], legend_label=labels[i], line_width=2,line_color=mypalette[i])

        st.bokeh_chart(p,use_container_width=True)




def Plotfft(plotDict,kwl,add_selectbox,smth):

    if 1:

        p = figure(
             title='simple line example',
             x_axis_label='x',
             y_axis_label='y')

        x=[]
        y=[]
        Binns = [datetime(2021, 4, y, x, z*15, 0) for y in range(5, 26) for x in range(24) for z in range(4) if datetime(2021, 4, y, x, z*15, 0)<datetime(2021, 4, 24, 0, 0, 0)]
        Binns=np.array(Binns)
        Binns = [x.timestamp() for x in Binns]

        x_ = [x.timestamp() for x in plotDict["catBinns"]]

        for a in add_selectbox:

            ycolumF = uniform_filter1d(plotDict[a], int(smth))

            f = interpolate.interp1d(x_, ycolumF)
            yint=f(Binns)

            if 0:
                N = 600
                T = 1.0 / 800.0

                yf = fft(yint)
                xf = fftfreq(N, T)[:N // 2]


                x.append(xf)
                y.append(2.0 / N * np.abs(yf[0:N // 2]))
            if 1:
                yint_fft = fft(yint)
                yint_psd = np.abs(yint_fft) ** 2
                freq = fftfreq(len(yint_psd), 1. / (24.*4.))
                i = freq > 0
                x.append(freq[i])
                y.append(yint_psd[i])


        mypalette = Spectral11[0:len(y)]
        for i in range(len(y)):
            p.line(x[i],y[i], legend_label=add_selectbox[i], line_width=2,line_color=mypalette[i])

        p.x_range = Range1d(.2, 10)
        #p.y_range = Range1d(5, 15)
        st.bokeh_chart(p,use_container_width=True)

def barplots(plotDict,barselectbox):
    bardict= {}
    bardictTot = 0
    TotCharHisto=plotDict["TotCharHisto"]
    catarr=plotDict["catarr"]

    for a in range(len(catarr)):
        if len(catarr[a].split("_")) == 2 and catarr[a].split("_")[0] == barselectbox:
            bardict[catarr[a].split("_")[1]] = TotCharHisto[a]
        if len(catarr[a].split("_")) == 1 and catarr[a].split("_")[0] == barselectbox:
            bardictTot = TotCharHisto[a]

    barTot=[(a[0],a[1]*1.0/bardictTot) for a in bardict.items()]
    barTot.sort(key=lambda x: x[1])
    x=[a[0] for a in barTot]
    y=[a[1] for a in barTot]
    p = figure(x_range=x, plot_height=600, title=barselectbox,
               toolbar_location=None, tools="")

    p.vbar(x=x, top=y, width=0.9)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.major_label_orientation = 1
    st.bokeh_chart(p, use_container_width=True)

if __name__=="__main__":

    if 1:
        st.sidebar.title("Fearless fan demographics")
        st.sidebar.markdown("Fearless by Taylor Swift was rerecorded and released on Apr 09 2021. ")


        # Add a selectbox to the sidebar:
        graphs=["compare demographics with tweets","demographic frequencies","LDA topic modeling"]
        main_selectbox = st.sidebar.selectbox(
            'which graph to view',
            graphs
        )
        plotDict = dill.load(open("plotDict3ns" + ".pkd", "rb"))
        kwl = plotDict["catarr"]

        if main_selectbox==graphs[0]:
            # Add a slider to the sidebar:
            add_slider = st.sidebar.slider(
                'Select smoothing value',
                1.0, 50.0, ( 20.0)
            )

            # Add a selectbox to the sidebar:
            add_selectbox = st.sidebar.multiselect(
                'which demographics to view?',
                kwl
            )

            if st.sidebar.checkbox('FFT'):
                st.header("The FFT of the demographic time series")
                #st.markdown("another statement")
                Plotfft(plotDict, kwl, add_selectbox, add_slider)
            else:
                st.header("The demographic time series")
                st.markdown("The frequency of retweets is graphed for reference")
                PlotTrends2(plotDict,kwl,add_selectbox,add_slider)
        elif main_selectbox==graphs[1]:
            # Add a selectbox to the sidebar:
            barplt=["ocupation","entertainment"]
            barselectbox = st.sidebar.selectbox(
                'which demographics to view?',
                barplt
            )
            st.header("The frequency distribution of various categories")
            st.markdown("Those categories listed as entertaiment are orgainized as things users consume and in occupation are meant to be things users act upon or do")
            barplots(plotDict,barselectbox)




        elif main_selectbox==graphs[2]:
            st.header("Results of LDA topic modeling for the 50000 most popular retweets")
            st.markdown("the use of seven topics gave the highest coherence score")
            HtmlFile = open("top50_7.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            #print(source_code)
            components.html(source_code,height = 700,width=1300)
            outputlist = dill.load(open("top50000rtoutputlist_7SW.pkd", "rb"))

            st.header("The top 15 tweets in each group")
            runi=-1.0
            for i in range(len(outputlist)):
                if runi==outputlist[i][0]:
                    st.write("    "+outputlist[i][1])
                else:
                    st.write("Group "+str(int(outputlist[i][0]+1)))
                    runi=outputlist[i][0]
                    st.write("    " + outputlist[i][1])
