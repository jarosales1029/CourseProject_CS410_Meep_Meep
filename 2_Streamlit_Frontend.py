# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:40:15 2022

@author: Empeekay
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vad

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

def up_file(uploaded_file):
    if uploaded_file is not None:
        df_main = pd.read_csv(uploaded_file)
        df_main['comments']=df_main['comments'].astype(str) 
        st.write(df_main)
    else:
        st.write("File not uploaded")
    return(df_main)
      
def select_option(option,df_main):
    st.subheader('You selected:')
    st.write(option)
    if option == "Austin, TX":
        df_main = df_main.loc[df_main['location'] == "Austin, TX"]
    elif option == "Chicago, IL":
        df_main = df_main.loc[df_main['location'] == "Chicago, IL"]
    elif option == "Denver, CO":
        df_main = df_main.loc[df_main['location'] == "Denver, CO"]
    elif option == "Boston, MA":
        df_main = df_main.loc[df_main['location'] == "Boston, MA"]
    elif option == "New Orleans, LA":
        df_main = df_main.loc[df_main['location'] == "New Orleans, LA"]
    elif option == "Nashville, TN":
        df_main = df_main.loc[df_main['location'] == "Nashville, TN"]     
    elif option == "Los Angeles, CA":
        df_main = df_main.loc[df_main['location'] == "Los Angeles, CA"]
    elif option == "San Francisco, CA":
        df_main = df_main.loc[df_main['location'] == "San Francisco, CA"]      
    else:
        st.write("Select correct options")
    return(df_main.sample(n=1000))

def run_analysis(df_main):
    sentiment = vad()
    
    sen = ['Positive', 'Negative', 'Neutral']
    sentiments = [sentiment.polarity_scores(i) for i in df_main['comments'].values]
    df_main['Vad_Negative Score'] = [i['neg'] for i in sentiments]
    df_main['Vad_Positive Score'] = [i['pos'] for i in sentiments]
    df_main['Vad_Neutral Score'] = [i['neu'] for i in sentiments]
    df_main['Vad_Compound Score'] = [i['compound'] for i in sentiments]
    score = df_main['Vad_Compound Score'].values
    t = []
    
    for i in score:
        if i >= 0.05:
            t.append('Positive')
        elif i <= -0.05:
            t.append('Negative')
        else:
            t.append('Neutral')
            
    df_main['Vad_Overall_Sentiment'] = t
    return(df_main)
    
st.title("VADER Sentiment Analysis")
st.markdown('The purpose of this webpage is to showcase the VADER (Valence Aware Dictionary for Sentiment Reasoning) model used for the Airbnb listings dataset. The VADER model is typically used for text sentiment analysis and is sensitive to both polarity (positive/negative) and intensity (strength) of emotion.')
st.markdown('The dataset needs to be downloaded and cleaned prior to utilizing this tool. In this example implementation, we have combined the data from eight (8) popular cities in the US for our analysis.  ')
with st.form("form1", clear_on_submit=True): 
    org_data = st.file_uploader("Upload Cleaned Dataset as a csv file")
    option = st.selectbox(
        'Which state would you like to perform sentiment analysis on?',
        ('Austin, TX', 'Boston, MA', 'Chicago, IL', 'Denver, CO', 
         'Los Angeles, CA', 'Nashville, TN', 'New Orleans, LA', 'San Francisco, CA'))      
    submit = st.form_submit_button("Submit this form")


if submit:
    
    df_main = pd.read_csv(org_data)
    df_main['comments']=df_main['comments'].astype(str) 
    st.header('Original Uploaded Data')
    st.write('This is a snapshot of the original cleaned dataset uploaded to this webpage.')
    st.write(df_main.head())
    
    # Subset the data based on the selected City
    st.header('Location Filtered Data')
    df_main = select_option(option, df_main)
    st.write('This is the filtered dataset based on the location selection. A sample of 1000 listings is selected randomly in order to reduce the computational burden on the Streamlit app.')
    st.write(df_main)    
    
    #Perform Vader Sentiment Analysis on the Selected City
    df_new2 = run_analysis(df_main)
    st.header('Output of VADER Sentiment Analysis')
    st.write('This is the output of the VADER sentiment analysis on the location filtered dataset')
    st.write(df_new2)
    
    # Word Cloud of the most common words in comments
    st.header('Airbnb Review - Most Common Words')
    st.write('The figure below provides a word cloud illustration of the most common words used in the review comments for the dataset chosen.')
    text = df_main
    text = text.comments.str.cat(sep=' ')
    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'black',
        stopwords = STOPWORDS).generate(str(text))
    fig = plt.figure(
        figsize = (40, 30),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    st.pyplot(plt)
    
    
    st.header('Top 10 Positive Reviews')
    st.markdown('The table below lists the highest 10 positive reviews based on the VADER compound score. **_Double_ _click_** on each comment to view the complete listing comment.')
    st.write(df_new2.nlargest(10, 'Vad_Compound Score')['comments'])
    st.header('Top 10 Negative Reviews')
    st.markdown('The table below lists the highest 10 negative reviews based on the VADER compound score. **_Double_ _click_** on each comment to view the complete listing comment.')
    st.write(df_new2.nsmallest(10, 'Vad_Compound Score')['comments'])

    #Plotting geotags for positive and negative reviews
    st.header('Interactive Map')
    st.write('This interactive map provides a visual representation of the output of the listings and associated sentiment analysis. Use **_scroll_** to zoom in and out of the map area to view the listings. The tooltip provides information about the neighborhood of the listing, latitude, longitude, VADER sentiment score, price and outcome of the Vader categorization (positive/neutral or negative). The color scheme of the tooltip varies from blue to red based on the VADER sentiment score.')
    df_new2.dropna(
        axis=0,
        how='any',
        thresh=None,
        subset=None,
        inplace=True
    )
    
    color_scale = [(0, 'red'), (1, 'blue')]
    
    fig = px.scatter_mapbox(df_new2, 
                            lat="latitude", 
                            lon="longitude", 
                            hover_name="neighbourhood_cleansed", 
                            hover_data=["Vad_Overall_Sentiment","price"],
                            color="Vad_Compound Score",
                            color_continuous_scale=color_scale,
                            zoom=10, 
                            height=800,
                            width=800)
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)