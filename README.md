# Fearless

Project Description:
This project concerns the advertising value of Taylor Swifts re-recording of her 2008 album Fearless.  Taylor has not granted any commercial licenses for 3 years despite getting dozens of license request per week, opting to re-record her music to have full control over her music and how it is used.   The success of her re-recording debut on April 9th 2021 shows that there is significant market value to her music.  However, measuring the consumer demographics is much more complicated now than it was in 2008 when fearless was first released.   In 2008 Taylor and Taylor’s fanbase were largely children, now 13 years later those consumers, Taylor, and Taylor’s image have significantly matured landing them in a significantly different economic group with essentially the same product. This capstone project concerns measuring the demographics and consumer interests of Taylors fan base to inform marketers of the value of using Taylor’s music for commercial projects.  For this capstone project I used twitter to investigate the consumer response of the fearless album release. 

Project Analysis:
Twitter data was mined from April 5th to April 27th 2021 using tweepy and the standard twitter api.   The twitter searches used the key words of taylorswift13, TaylorsVersion, taylorswift13, and FromTheVault. 

The data were analysed as follows:
Step 1:
In the program file FearlessAnalysis.py data from the full tweets databank were extracted and categorized into dictionaries using the function GoThroughTweets.  
Step 2:
The tweets were separated into original tweets and retweets (using CategorizeTweets in FearlessAnalysis.py), and the data was organized by tweet id and user id. Both the text from the tweets and various demographic information from the user info were recorded.

Step 3:
The text from the tweets and user descriptions were cleaned using the program Tweetcleaningbase.py within the function CleanUserDescription in FearlessAnalysis.py.  This process included removing emojis, parsing HTML, removing web links, removing hashtags, replacing contractions.

Step 4:

To investigate twitter user demographics, I used the voluntary description category where users can list identifying characteristics.  Only a small percentage of users have enough data in this category, so a large sample was necessary to gain any reasonable numbers of characteristics. The user category’s text was cleaned with Tweetcleaningbase.py and then tokenized and lemmatized and the frequency of words group was calculated.  The words were sorted then hand chosen for use with demographic characterization.   These words where then classified in various categories. 

Sentiment analysis was used to check the sentiment of the tweets using the function sentiment in FearlessAnalysis.py, but very few tweets were found to be negative, and a majority of those that were negative were falsely negative.  Therefore sentiment filtering was not used in this project. 

Step 5:
Take the categorization from a spreadsheet developed in step 5, and organize the categories into a one-dimensional list and find if the intersection of the key words in these categories and the tokenized and cleaned word lists of user descriptions is non empty using the pandasinv function in FearlessAnalysis.py. 

Step 6:
Bin the tweet timeline into equal density intervals and find the percentage of categories of the users each interval represents using the programs Binn_tweet_timeline and Fill_timeline_with_demographics in FearlessAnalysis.py.

Step 7:
Extract the data from this categorical filtering to be visualized in the visualization program FearlessVis.py

Topic modelling with Latent Dirichlet Allocation: 
The LDA procedure cleaned the text further in the program FearlessLDAanalysis.py.  The text was tokenized and organized also with bigrams.  Stop words were removed, include stop words directly related to this project (like Taylor Swift etc).  The data was lemmatized, and a dictionary and corpus was made out of the clean text to put into the genism LDA model.  The LDA model was performed by varying the number of topics, and a grid search analysis was performed to find the maximum coherence score.  The LDA model was graphically saved using genism graphing tool.  This graph and the top 15 tweets in each category were fed into the visualization program FearlessVis.py.

The Visualization Program:

The visualization program FearlessVis.py uses streamlit to produce an app for a user to explore and visualize the data.  The visualization app has three main categories.  
The first shows a standard bar chart showing the frequencies of data produced using key words that reference entertainment and occupation preferences of twitter users.   Entertainment categories reference activities that a user can consume or passively observe and partake, and occupation categories reference activities that have a more active and defining approach from the user.  
The second graphs the time series of the categorical demographic data overlayed with a graphical representation of the retweets of the Fearless related material with the release of the album happening on April 09 2021 at midnight and represents the tallest part of the graph on the retweet line.  I hope to improve this graph by incorporating visualization of the top retweets in each time period and its category with the topic modelling analysis.   The Fourier transform of this time series data can also be observed by pressing the FFT box. 
The third shows the topic modelling results of the LDA analysis.  This shows the visualization given by gensim and list the top 15 tweets in each category under each graph. 

Some noteworthy observations from this study are:
1.	When looking at the category parent_motherhood, there is a distinct periodicity that can be seen with the Fourier transform.  This shows that mothers follow circadian rhythms most aligned with the US, that may be influenced by small children. 
2.	When looking at the occupation and entertainment categories there is a high representation of writers and visual arts.  Showing perhaps an intelligence quotient for Taylor Swift fans. 
3.	For those who have technical occupations the twitter usage seems to be focused on the weekend after the release, show perhaps a more responsible enjoyment of the albums release. 
