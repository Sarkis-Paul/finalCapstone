# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:30:27 2024

@author: sps220
"""
# NOTE: The current code is set to read through all the reviews, hence the results take some time to load
# I have chosen to use all of them as they provide a more holistic picture

# Libraries & NLP loading
import pandas as pd
import numpy as np
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("spacytextblob")
# polarity = doc._.blob.polarity


# Data preprocessing
dataframe = pd.read_csv("amazon_product_reviews.csv")

# Removing and replacing words that do not add any sentimental meaning
reviews = dataframe['reviews.text'].str.replace(" is ", " ")
reviews = reviews.str.replace(" and ", " ")

# Remove any 'Nan's and 
cleanreviews = reviews.dropna()

# Function definition for sentiment analysis
def sentiment_analysis(description):
    
    sentiment = nlp(description)
    
    return sentiment._.blob.polarity, sentiment._.blob.subjectivity

######################################################################################################

# Results and Data Post-processing
total_no = len(cleanreviews) # Total number of reviews
polarity = np.zeros(total_no) # Array to contain all polarity values 
subjectivity = np.zeros(total_no) # Array to contain all subjectivity/sentiment strength values

for i in range(0, total_no):    
    polarity[i], subjectivity[i] = sentiment_analysis(cleanreviews[i])

mean_polarity = sum(polarity)/total_no
mean_subjectivity = sum(subjectivity)/total_no

# Printing out numerical results
print("Mean Review Polarity:", mean_polarity)
print("Mean Review Subjectivity:", mean_subjectivity)

# Printing out a message to interpret the numerical results
if mean_polarity < 0.4 and mean_polarity >= -0.4:
    print("The reviews are a Neutral rating on average.")
    
elif mean_polarity < -0.4:
    print("The reviews have a Negative rating on average.")
    
else:
    print("The reviews have a Positive rating on average.")    
    
if mean_subjectivity < 0.5:
    print("Results may be potentially biased - there little objective confidence in these results.")

else:
    print("Results are reliable as reviews appear to be objective.")
    
######################################################################################################
#%%
# Calculating the average and maximum similarity of the first review with the other reviews
comparison_to = nlp(cleanreviews[0]) # First review
    
# variable to store and update mean and max values of similarity
maxsimilarityvalue = 0
meansimilarityvalue = 0
    
# start iterating through the reviews
for i in range(1, total_no):
        
    similarity = nlp(cleanreviews[i]).similarity(comparison_to) # the description of the movie is after the 9th character on the list
    meansimilarityvalue += similarity
        
    if similarity > maxsimilarityvalue:
        maxsimilarityvalue = similarity       

meansimilarityvalue = meansimilarityvalue/total_no # dividing the total sum to obtain the actual mean value

# Printing out the results
print("Mean Similarity of reviews with the first one:", meansimilarityvalue)
print("Maximum value of similarity observed with the first review:", maxsimilarityvalue)


