#!/usr/bin/env python
# coding: utf-8

# # Udacity
# # Project 4: Wrangle and Analyze Data
# Vasileios Garyfallos, April 2020
# 
# ## Table of Contents
# <ul>
#     <li><a href="#intro">Introduction</a></li>
#     <li><a href="#sources">Data Sources</a></li>
#     <li><a href="#gathering">Data Gathering</a></li>
#     <li><a href="#assessing">Data Assessing</a></li>
#         <li><a href="#assessingsum">Assessing Summary</a></li>
#     <li><a href="#cleaning">Data Cleaning</a></li>
#     <li><a href="#analysis">Data Analysis</a></li>
#     <li><a href="#conclusion">Summary and Conclusions</a></li>
# </ul>

# ## 1. Introduction
# For this project, it is asked to gather and analyze data (tweet archive) from the Twitter account <a href = "https://twitter.com/dog_rates?lang=de">"WeRateDogs"</a>. 
# 
# The data will be gathered using manual download, programmatical download and quering an API. 
# 
# After data gathering, data assessment is required. in order to define any issues regarding data cleanliness and tidiness. The next step will be to clean the data with the aim to get a clean and tidy dataset for the analysis.

# <a id='sources'></a>
# ## Data Sources
# 
# 
# >1. **Name:** WeRateDogs Twitter Archive (twitter-archive-enhanced.csv)</li>
# ><ul>   
# >    <li><b>Source:</b> <a href = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/59a4e958_twitter-archive->enhanced/twitter-archive-enhanced.csv">Udacity</a></li>    
# 
# >    <li><b>Method of gathering:</b> Manual download</li>
# ></ul>
# 
# >2. **Name:** Tweet image predictions (image_predictions.tsv)</li>
# ><ul>   
# >    <li><b>Source:</b> <a href="https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image->predictions.tsv">Udacity</a></li>     
# 
# >    <li><b>Method of gathering:</b> Programmatical download</li>
# ></ul>
# 
# >3. **Name:** Additional Twitter data (tweet_json.txt)
# ><ul>   
# >    <li><b>Source:</b> <a href = "https://twitter.com/dog_rates">WeRateDogs</a></li>    
# 
# >    <li><b>Method of gathering:</b> API (Tweepy)</li>
# ></ul>
# 

# #### Import Python libraries:

# In[1]:


import requests
import numpy as np 
import pandas as pd 
import json 
import seaborn as sns
import re
import tweepy
import matplotlib.pyplot as plt


# <a id='gathering'></a>
# ## Data Gathering
# 
# #### 1. WeRateDogs Twitter Archive (twitter-archive-enhanced.csv)
# 
# Downloaded the .csv file from Udacity and loaded it into a  Pandas dataframe.

# In[2]:


df_twitter = pd.read_csv("twitter-archive-enhanced.csv")

df_twitter.head(3)


# #### 2. Tweet image predictions (image_predictions.tsv)
# 
# requested the file's URL and wrote the content of the response to a file.

# In[3]:


url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"

#get response
response = requests.get(url)

#write return to an image
with open("image_predictions.tsv", mode = "wb") as file:
    file.write(response.content)


# In[4]:


df_predict = pd.read_csv("image_predictions.tsv", sep='\t')

df_predict.head(3)


# #### 3. Additional Twitter data (tweet_json.txt)

# I created a Twitter developer account, in order to gather the data from the Twitter API. Then pulled the data using tweepy. I wrote the data in a new file called "tweet_json.txt".

# In[5]:


#API keys
consumer_key = 'EMPTIED'
consumer_secret = 'EMPTIED'
access_token = 'EMPTIED'
access_secret = 'EMPTIED'

#access the API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# #get all the twitter ids in the df
# twitter_ids = list(df_twitter.tweet_id.unique())
# 
# #https://realpython.com/python-json/
# #save the gathered data to a file
# with open("tweet_json.txt", "w") as file:
#     for ids in twitter_ids:
#         print(f"Gather id: {ids}")
#         try:
#             #get all the twitter status - extended mode gives us additional data
#             tweet = api.get_status(ids, tweet_mode = "extended")
#             #dump the json data to our file
#             json.dump(tweet._json, file)
#             #add a linebreak after each dump
#             file.write('\n')
#         except Exception as e:
#             print(f"Error - id: {ids}" + str(e))
#             

# 

# In[6]:


#https://towardsdatascience.com/flattening-json-objects-in-python-f5343c794b10
api_data = []

#read the created file
with open("tweet_json.txt", "r") as f:
    for line in f:
        try: 
            tweet = json.loads(line)
            #append a dictionary to the created list            
            api_data.append({
                "tweet_id": tweet["id"],
                "retweet_count": tweet["retweet_count"],
                "favorite_count": tweet["favorite_count"],
                "retweeted": tweet["retweeted"],
                "display_text_range": tweet["display_text_range"]                
            })               
                
            #tweet["entities"]["media"][0]["media_url"]
        except:
            print("Error.") 
            
df_api = pd.DataFrame(api_data, columns = ["tweet_id", "retweet_count", "favorite_count", "retweeted", "display_text_range"])
df_api.head()


# In[7]:


df_twitter.head(1)


# In[8]:


df_predict.head(1)


# In[9]:


df_api.head(1)


# <a id='assessing'></a>
# ## Data Assessing
# 

# Assessing the quality/tidiness of the data.

# #### df_twitter
# 
# Let's first look for missing data.

# In[10]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_twitter.isnull(), vmin=0, vmax = 1)


# As we can see, there is a lot of missing data in the columns about the reply and the retweeted status. Since we only want original posts with images, we have to drop them later - the missing data in the "expanded_urls" column will also disappear with that cleaning operation.

# In[11]:


df_twitter.sample(5)


# visual assessment:
# - not all tweets could be classified as doggo, floofer, pupper or puppo and all columns contain "None"
# - the source contains unnecessary HTML code
# - there is the name "None" in the name column

# In[12]:


df_twitter.info()


# Also the datatypes are incorrect:
# - tweet_id should be a str
# - timestamp - columns should be datetime objects
# 
# Now let's see how many wrong names we can find.

# In[13]:


df_twitter.name.value_counts()


# As we can see, the name column contains wrong names like "None", "a", "the", "an". 

# In[14]:


df_twitter[df_twitter.duplicated()]


# There are no duplicates in this data, so the number of unique tweet_ids should be the length of the df.

# In[15]:


df_twitter.tweet_id.nunique()


# Proof. Now let's create a copy of the twitter dataframe for further assessing. I want to see in how many cases there were no classification of dog_size possible via text processing.

# In[16]:


df_twitter_assess = df_twitter.copy()


# In[17]:


#returns true if there is no dog classification in any of the columns
df_twitter_assess[["doggo","floofer","pupper","puppo"]].apply(lambda x: True if
    (x[0] == "None" and x[1] == "None" and x[2] == "None" and x[3] =="None") 
    else False, axis = 1).value_counts()


# In[18]:


df_twitter_assess["doggo"].value_counts()


# In[19]:


df_twitter_assess["floofer"].value_counts()


# In[20]:


df_twitter_assess["pupper"].value_counts()


# In[21]:


df_twitter_assess["puppo"].value_counts()


# Only for 16% of the rows the data is not missing. Now let's take a look at the ratings. By what we have seen so far, it looks like the ratings have always a format of 13/10 or 12/10 and so on. So we would expect a numerator > 10 and denominator = 10.

# In[22]:


df_twitter.rating_numerator.value_counts()


# We can observe that there is a wide range of numbers as rating_numerator, with a maximum of 1776. 

# In[23]:


print(df_twitter.query("rating_numerator == '1776'").text)


# This rating is correct - so a lot higher values as 12 or 13 would be valid to this rating system. But what is with the very small ones?

# In[24]:


print(df_twitter.query("rating_numerator == '1'").text)


# The entry 605 shows, that these tweets contain pictures that don't contain any dogs. Also in the entry 2335 the rating got extracted wrongly (misinterpreted the 1/2 of 3 1/2 as the rating). 

# In[25]:


print(df_twitter.query("rating_numerator == '0'").text)


# No clear doggo photos again. Let's make the same check for the denominator.

# In[26]:


df_twitter.rating_denominator.value_counts()


# In[27]:


print(df_twitter.query("rating_denominator == '170'").text)


# In[28]:


print(df_twitter.query("rating_denominator == '0'").text)


# In[29]:


print(df_twitter.query("rating_denominator == '7'").text)


# The same problems as for the numerator. Multiple Dogs or multiple occurences of the pattern \d+\/\d+. Lets extract this by our self and see what we get.

# In[30]:


pattern = "(\d+(\.\d+)?\/\d+(\.\d+)?)" #we could expect an integer rating on what we saw, but maybe some floats are the case

#https://stackoverflow.com/questions/36028932/how-to-extract-specific-content-in-a-pandas-dataframe-with-a-regex
df_twitter_assess["rating"] = df_twitter_assess.text.str.extract(pattern, expand = True)[0]

#https://stackoverflow.com/questions/14745022/how-to-split-a-column-into-two-columns
df_twitter_assess[['num', 'denom']] = df_twitter_assess['rating'].str.split('/', n=1, expand=True)


# In[31]:


df_twitter_assess.rating_numerator = df_twitter_assess.rating_numerator.astype("str")
df_twitter_assess.rating_denominator = df_twitter_assess.rating_denominator.astype("str")


# In[32]:


#look for differences in the original numerator and the newe extract
df_twitter_assess["check_num"] = df_twitter_assess[["rating_numerator", "num"]].apply(lambda x: False if (x[0] != x[1]) else True, axis = 1)


# In[33]:


df_twitter_assess.check_num.value_counts()


# In[34]:


df_twitter_assess.query("check_num == False")[["rating_numerator", "num","check_num"]]


# These are the differences we found by extracting the first occurrence of the pattern. These ratings got transformed to integers and are therefore wrong. 

# In[35]:


df_twitter_assess[["rating_numerator", "num", "check_num"]].sample(15)


# Let's repeat this for the denominator.

# In[36]:


df_twitter_assess["check_denom"] = df_twitter_assess[["rating_denominator", "denom"]].apply(lambda x: False if (x[0] != x[1]) else True, axis = 1)


# In[37]:


df_twitter_assess.check_denom.value_counts()


# In[38]:


df_twitter_assess.query("check_denom == False")[["rating_denominator", "denom","check_denom"]] #problem with integer, maybe also floats?


# This seems like it is no problem that we have to worry about.

# In[39]:


df_twitter_assess[["rating_denominator", "denom", "check_denom"]].sample(5)


# Now we should assess how often there are multiple occurences of the "rating pattern" in one tweet.

# In[40]:


df_twitter_assess["count"] = df_twitter_assess.text.str.count(pattern)


# In[41]:


df_twitter_assess["count"].value_counts()


# In[42]:


#show the full text
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 4000)

df_twitter_assess[["text", "count"]].query("count != 1")


# We can see that:
# - this data contains retweets (as mentioned before)
# - sometimes there are multiple dogs/cats or else in one picture
# - some of these ratings are not clear

# #### df_predict

# Let's also begin with the missing data first.

# In[43]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_predict.isnull(), vmin = 0, vmax = 1)


# Looks fine. Now the visual assessment:

# In[44]:


df_predict.sample(10)


# We can see that:
# - the predicitions are sometimes lowercase, sometimes uppercase
# - there is an underscore instead of a whitespace between the words
# - there are rows with no prediciton of a dog (neither in 1, 2 nor 3)

# In[45]:


df_predict.info()


# - the tweet_id colum should again be string

# The best way to find duplicates is to look at the jpg - url. If there are value counts > 1, then this data contains duplicates/retweets.

# In[46]:


df_predict.jpg_url.value_counts().head(10)


# As predicted - this data contains retweets. 

# In[47]:


df_predict[df_predict.tweet_id.duplicated()]


# In[48]:


df_predict[df_predict.jpg_url == "https://pbs.twimg.com/media/CeRoBaxWEAABi0X.jpg"]


# In[49]:


df_twitter[df_twitter.tweet_id == 798697898615730177]


# We only want Tweets with pictures which contain dogs. Let's see if there are pictures, for which the ML - Algorithm didn't predict any dogs.

# In[50]:


df_predict.query("p1_dog == False and p2_dog == False and p3_dog == False")


# In[51]:


df_predict.query("p1_dog == False and (p2_dog == True or p3_dog == True)")


# After checking some of these pictures it gets clear, that sometimes the doggos are in the background or the pictures doesn't contain any dogs at all.

# **df_api**

# Lets repeat the same pattern for this dataset: missing data → visual assessment.

# In[52]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_api.isnull(), vmin = 0, vmax = 1)


# In[53]:


df_api.sample(10)


# In[54]:


df_api.info()


# Overall it looks good for this dataset. There is no missing data - only the datatype of the tweet_id should again be a string.

# <a id='assessingsum'></a>
# ### Assessing Summary

# #### Quality
# ##### `df_twitter` table
# - the datatype of the id - columns is integer and should be str
# - the datatype of the timestamp - column is object and should be datetime
# - some of the dogs are not classified as one of "doggo", "floofer", "pupper" or "puppo" and contain all "None" instead
# - some of the dog names are not correct (None, an, by, a, ...)
# - contains retweets
# - some of the ratings are not correctly extracted (mostly if there are >1 entries with the pattern "(\d+(\.\d+)?\/\d+(\.\d+)?)"
# - also transforming the ratings to integer created some mistakes (there are also floats)
# - the source column contains html code
# 
# ##### `df_predict` table
# - the datatype of the id - columns is integer and should be str
# - contains retweets (duplicated rows in column `jpg_url`)
# - there are pictures in this table that are not dogs
# - the predictions are sometimes uppercase, sometimes lowercase
# - also there is a "_" instead of a whitespace in the predictions
# 
# ##### `df_api` table
# - the datatype of the id - columns is integer and should be str

# #### Tidiness
# ##### `df_twitter` table
# - the columns `doggo`, `floofer`,`pupper` and `puppo` are not easy to analyze and should be in one column
# 
# ##### `df_predict` table
# - the prediction and confidence columns should be reduced to two columns - one for the prediction with the highest confidence (dog)
# 
# ##### `df_api` table
# - display_text_range contains 2 variables
# 
# ##### `all` tables
# 
# - All three tables share the column `tweet_id` and should be merged together.

# <a id='cleaning'></a>
# ## Data Cleaning

# Cleaning steps:
# 
# <ol>
#     <li>Merge the tables together</li>
#     <li>Drop the replies, retweets and the corresponding columns and also drop the tweets without an image or with images which don't display doggos</li>
#     <li>Clean the datatypes of the columns</li>
#     <li>Clean the wrong numerators - the floats on the one hand (replacement), the ones with multiple occurence of the pattern on the other (drop)</li>
#     <li>Extract the source from html code</li>
#     <li>Split the text range into two separate columns</li>
#     <li>Remove the "None" out of the doggo, floofer, pupper and puppo column and merge them into one column</li>
#     <li>Remove the wrong names of name column</li>
#     <li>Reduce the prediction columns into two - breed and conf</li>
#     <li>Clean the new breed column by replacing the "_" with a whitespace and make them all lowercase</li>
# </ol>

# **1. Merge the tables together**

# I could clean all the tables one by one, but all of them share some cleaning needs or are dependent on each other to do so (for example removing of retweets or pictures not containig dogs). By merging them all together as first step, I can save some coding time and avoid repetition. 

# In[55]:


#outer join to not loose rows at first
df_master = pd.merge(df_twitter, df_api, on = "tweet_id", how = "outer")


# In[56]:


df_master = pd.merge(df_master, df_predict, on = "tweet_id", how = "outer")


# In[57]:


df_master_clean = df_master.copy()


# In[58]:


df_master_clean.info()


# **2. Drop the replies, retweets and the corresponding columns and also drop the tweets without an image or with images which don't display doggos**

# Now that this is done, we have to remove the replies, retweets and the tweets withouth an image displaying a dog, because we only want original tweets with images. Let's visualize our missing data first.

# In[59]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)


# In[60]:


pd.set_option('display.max_colwidth', 50)
df_master_clean[df_master_clean["retweeted"].isnull()]


# In[61]:


#we only want the rows without an entry in "retweeted_status_id" in our master dataframe
df_master_clean = df_master_clean[df_master_clean["retweeted_status_id"].isnull()]


# In[62]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)


# In[63]:


df_master_clean.columns


# In[64]:


#check with the column from the api table, no retweets left
df_master_clean.retweeted.value_counts()


# In[65]:


#same as for the retweets, we only want the rows without an entry in "in_reply_to_status_id"
df_master_clean = df_master_clean[df_master_clean.in_reply_to_status_id.isnull()]


# In[66]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)


# During the gathering process from the API there were some tweets, which got deleted by the account. We will also drop them out of our master dataframe.

# In[67]:


df_master_clean.dropna(subset = ["retweeted"], inplace = True)


# In[68]:


#drop the unneeded columns
df_master_clean.drop(["in_reply_to_status_id", "in_reply_to_user_id",
                      "retweeted_status_id", "retweeted_status_user_id", 
                      "retweeted_status_timestamp", "retweeted"], inplace=True, axis = 1)


# In[69]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)


# Now we want to take a look on the "jpg_url" column and drop all the rows, which are NAN - since these are the ones without an image. To check that, we could read in the image data from the gathered API data - but I will leave this for another time.

# In[70]:


df_master_clean.dropna(subset = ["jpg_url"], inplace = True)


# In[71]:


#check if there are still duplicated images after dropping the replies and the retweets
sum(df_master_clean.jpg_url.duplicated())


# In[72]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)


# The last step here is to drop the rows which contain images, that are not displaying any dogs (relying on the top three predictions of the ML algorithm).

# In[73]:


df_master_clean.drop(df_master_clean.query("p1_dog == False and p2_dog == False and p3_dog == False").index, inplace = True)


# In[74]:


df_master_clean.query("p1_dog == False and p2_dog == False and p3_dog == False")


# In[75]:


df_master_clean.info()


# **3. Clean the datatypes of the columns**

# In[76]:


df_master_clean["tweet_id"] = df_master_clean["tweet_id"].astype("str")


# In[77]:


#transform the timestamp to datetime
df_master_clean["timestamp"] = pd.to_datetime(df_master_clean.timestamp)


# In[78]:


for x in ["retweet_count", "favorite_count", "img_num"]:
    df_master_clean[x] = df_master_clean[x].astype("int64")


# In[79]:


df_master_clean.info()


# **4. Clean the wrong numerators - the floats on the one hand, the ones with multiple occurence of the pattern on the other**

# While assessing the dataset, we found out, that floating numbers got transformed into integers, which lead to loss of information.

# In[80]:


df_twitter_assess.query("check_num == False")[["rating_numerator", "num","check_num"]]


# We dropped a lot of rows, so we cannot be sure that all of these problems are still in this dataset, so we will extract it again.

# In[81]:


pattern = "(\d+\.\d+\/\d+)"

df_master_clean.text.str.extract(pattern, expand = True)[0].dropna()


# In[82]:


#get the right numerator out of the string
df_num_clean = df_master_clean.text.str.extract(pattern, expand = True)[0].dropna().str.split('/', n=1, expand=True)[0]


# In[83]:


df_num_clean


# In[84]:


#get the index of the wrong data
df_num_clean_index = df_num_clean.index
df_num_clean_values = df_num_clean.values.astype("float64")


# Now that we have our data together, we can impute these values and clean this data.

# In[85]:


#transform the datatypes to float
df_master_clean.rating_numerator = df_master_clean.rating_numerator.astype("float64")
df_master_clean.rating_denominator = df_master_clean.rating_denominator.astype("float64")
#impute the data
df_master_clean.loc[df_num_clean_index, "rating_numerator"] = df_num_clean_values
df_master_clean.loc[df_num_clean_index].rating_numerator


# We also have the problem, that there can be multiple occurrences of the pattern. The reason for this is - most of the time - the display of two or more dogs in an image. For this cases we could add the ratings up, because the author of the Twitter account did this in one case that we found. Or we could build the average rating per each picture. For now, we are going to drop them out of the dataframe.

# In[86]:


pattern = "(\d+(\.\d+)?\/\d+(\.\d+)?)"

print(df_master_clean.text.str.count(pattern)[df_master_clean.text.str.count(pattern) != 1])

#get the index of the rows which contains the pattern more than once
pattern_clean_index = df_master_clean.text.str.count(pattern)[df_master_clean.text.str.count(pattern) != 1].index


# In[87]:


df_master_clean.drop(pattern_clean_index, inplace = True)


# In[88]:


df_master_clean.info()


# In[89]:


#no more occurrences of the mentioned problem are left
print(df_master_clean.text.str.count(pattern)[df_master_clean.text.str.count(pattern) != 1])


# **5. Extract the source from html code**

# Right now the source column is not giving us any useful information while looking at it. Because the relevant information is always between two "> <", the information will be easy to extract.

# In[90]:


df_master_clean.head(2)


# In[91]:


#https://stackoverflow.com/questions/3075130/what-is-the-difference-between-and-regular-expressions
df_master_clean.source = df_master_clean.source.str.extract("\>(.*?)\<", expand = True)


# In[92]:


df_master_clean.iloc[:,:3].head(2)


# In[93]:


df_master_clean.source.value_counts()


# **6. Split the text range into two separate columns**

# In[94]:


df_master_clean[["display_text_range"]].info()


# In[95]:


df_master_clean.display_text_range[1]


# Since the display_text_range column is interpreted as list, we can simply split it by using list indexing.

# In[96]:


#get the lower text range at list index 0
df_master_clean["lower_text_range"] = df_master_clean["display_text_range"].apply(lambda x: x[0])

#get the lower text range at list index 1
df_master_clean["upper_text_range"] = df_master_clean["display_text_range"].apply(lambda x: x[1])
df_master_clean.drop("display_text_range", axis = 1, inplace = True)


# In[97]:


df_master_clean[["lower_text_range", "upper_text_range"]].head()


# **7. Remove the "None" out of the doggo, floofer, pupper and puppo column and merge them into one column**

# We want to reduce the columns into one for an easier analysis. For that we have to remove the None with "" at first to concat the columns together and afterswards with np.nan, so we could easily exclude these rows from a specific analysis.

# In[98]:


#replace "None" with "" in each column
for x in ["doggo", "floofer", "pupper", "puppo"]:
    df_master_clean[x].replace("None", "", inplace = True)

#https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-dataframe-in-pandas-python
#concat the columns together
df_master_clean['dog_class'] = df_master_clean['doggo'].map(str) + df_master_clean[
    'floofer'].map(str) + df_master_clean['pupper'].map(str) + df_master_clean['puppo'].map(str)


# In[99]:


df_master_clean.dog_class.value_counts()


# In[100]:


#replace the leftover "" with np.nan
df_master_clean["dog_class"].replace("", np.nan, inplace = True)


# In[101]:


df_master_clean.dog_class.value_counts() 


# In[102]:


#count the occurrences of the pattern and show the rows with count > 1
df_master_clean.text.str.count(r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)')[
    df_master_clean.text.str.count(r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)') > 1]


# As we can see there are cases, in which there were multiple classifications. Let's extract the classes from the text and see where the differences occur.

# In[103]:


df_master_clean["dog_class_re"] = df_master_clean.text.str.extract(
    r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)', expand = True)


# In[104]:


#http://queirozf.com/entries/visualization-options-for-jupyter-notebooks
#show the full text
pd.set_option('display.max_colwidth', -1)

#find the differences of the extract
df_master_clean[["text","dog_class", "dog_class_re"]].dropna(subset = ["dog_class_re"]).query("dog_class != dog_class_re")


# The difference occurs in 8 cases. We can read through the text and extract the correct dog_class. Afterwards we can impute the correct classes into the column - for cases, in which there are multiple dogs in it, we will impute np.nan for consistency.

# In[105]:


#191 puppo
#200 floofer
#531 two dogs
#575 pupper
#889 two dogs
#956 not classified by author
#1063 two dogs
#1113 two dogs

df_master_clean.loc[191, "dog_class"] = "puppo"
df_master_clean.loc[200, "dog_class"] = "floofer"
df_master_clean.loc[531, "dog_class"] = np.nan
df_master_clean.loc[575, "dog_class"] = "pupper"
df_master_clean.loc[889, "dog_class"] = np.nan
df_master_clean.loc[956, "dog_class"] = np.nan
df_master_clean.loc[1063, "dog_class"] = np.nan
df_master_clean.loc[1113, "dog_class"] = np.nan


# In[106]:


#find the differences of the extract - worked
df_master_clean[["text","dog_class", "dog_class_re"]].dropna(subset = ["dog_class_re"]).query("dog_class != dog_class_re")


# In[107]:


#drop the columns out
df_master_clean.drop(["doggo", "floofer", "pupper", "puppo", "dog_class_re"], inplace = True, axis = 1)


# Similar to the multiple occurence of a pattern for the numerator, we should do the same check here.

# In[108]:


#count the occurrences of the pattern and show the rows with count > 1
df_master_clean[["text", "dog_class"]].loc[
    df_master_clean.text.str.count(r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)')[
        df_master_clean.text.str.count(r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)') > 1].index]


# All of these look correct.

# **8. Remove the wrong names of name column**

# Here we will also replace the wrong names with np.nan.

# In[109]:


for x in ["None", "a", "by", "the"]:
    df_master_clean["name"].replace(x, np.nan, inplace = True)


# In[110]:


df_master_clean.name.value_counts()


# In[111]:


df_master_clean.head()


# In[112]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)


# In[113]:


df_master_clean.info()


# **9. Reduce the prediction columns into two - breed and conf**

# In the next step we want to reduce the prediction columns into two - breed and confidence. The columns are already sorted by confidence. We will take the most likely prediction for each row which is supposed to be a dog.

# In[114]:


df_master_clean.query("p2_conf > p1_conf")


# In[115]:


df_master_clean.query("p3_conf > p1_conf")


# In[116]:


df_master_clean.query("p3_conf > p2_conf")


# The order is correct.

# In[117]:


#extract the most likely prediction which is a dog
def get_attr(x):
    """
    INPUT: 
        Columns in this order: Check1, Result1, Check2, Result2, Result3
    OUTPUT:
        Results based on the check in this columns
    """
    if x[0] == True:
        return x[1]
    elif x[2] == True:
        return x[3]
    else:
        return x[4]
    
df_master_clean["breed"] = df_master_clean[["p1_dog", "p1", "p2_dog", "p2", "p3"]].apply(get_attr, axis = 1)
df_master_clean["conf"] = df_master_clean[["p1_dog", "p1_conf", "p2_dog", "p2_conf", "p3_conf"]].apply(get_attr, axis = 1)


# In[118]:


df_master_clean.iloc[:, 12:]


# In[119]:


#drop the reduced columns
df_master_clean.drop(df_master_clean.columns[12:21], inplace = True, axis = 1)


# In[120]:


df_master_clean.head()


# **10. Clean the new breed column by replacing the "_" with a whitespace and make them all lowercase**

# Now that we have our reduced column, we have to clean it for consistency.

# In[121]:


#replace "_" with " "
df_master_clean.breed = df_master_clean.breed.str.replace("_", " ")


# In[122]:


df_master_clean.breed


# In[123]:


#https://stackoverflow.com/questions/22245171/how-to-lowercase-a-python-dataframe-string-column-if-it-has-missing-values
#lower the strings
df_master_clean.breed = df_master_clean.breed.str.lower()


# In[124]:


df_master_clean.breed.value_counts().head(10)


# In[125]:


#reset index to match with the real amount of rows
df_master_clean.reset_index(drop = True, inplace = True)


# In[126]:


fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)


# In[127]:


df_master_clean.info()


# In[128]:


df_master_clean[["breed", "conf"]].head(3)


# In[129]:


#save the data to a *.csv file
df_master_clean.to_csv('twitter_archive_master.csv', index = False)


# <a id='analysis'></a>
# ## Data Analysis

# **Questions:**
# 
# <ol>
#     <li>Based on the predicted, most likely dog breed: Which breed gets retweeted and favorited the most overall?</li>
#     <li>How did the account develop (speaking about number of tweets, retweets, favorites, image number and length of the tweets)?</li> 
#     <li>Is there a pattern visible in the timing of the tweets?</li> 
# </ol>
#     

# **1. Based on the predicted, most likely dog breed: Which breed gets retweeted and favorited the most overall?**

# To answer this question we will first take a look on the frequency of the breed occurence and afterwards we will create a groupby object to sum up the favorite and retweet count of each breed in this dataset.

# In[130]:


#read in the master csv
df = pd.read_csv("twitter_archive_master.csv")


# In[131]:


df.head()


# In[132]:


df.columns


# In[133]:


#https://stackoverflow.com/questions/32891211/limit-the-number-of-groups-shown-in-seaborn-countplot
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "breed", data = df, order=df.breed.value_counts().iloc[:10].index, palette = "viridis")
ax.set_title("count of classified breeds in the dataset");

ax.set_ylim(0, 170)
#https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+2))


# The dogs displayed in the images are mostly golden retrievers with a count of 154 or labrador retrievers with a count of 105.

# In[134]:


df_breed_group = df[["retweet_count", "favorite_count", "breed"]].groupby("breed", as_index = False).sum()


# In[135]:


df_breed_group.sort_values("retweet_count", ascending = False).head(10)


# In[136]:


df_breed_group.sort_values("favorite_count", ascending = False).head(10)


# The golden retriever and the labrador retriever therefore also lead the list of most favorite and retweets.

# In[137]:


df_breed_group["sum"] = df_breed_group["retweet_count"] + df_breed_group["favorite_count"]


# In[138]:


df_breed_group.sort_values("sum", ascending = False).head(10)


# In[139]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "breed", y = "sum", data = df_breed_group.sort_values("sum", ascending=False).iloc[:10], palette = "viridis")
ax.set_title("sum of favorites and retweets per breed");


# Now let's look at the most retweetet and favorited single tweet.

# In[140]:


df[["retweet_count", "favorite_count", "breed"]].sort_values("retweet_count", ascending = False).head(5)


# In[141]:


df[["retweet_count", "favorite_count", "breed"]].sort_values("favorite_count", ascending = False).head(5)


# We can see that the most liked and retweeted tweet is in fact a labrador retriever, with golden retrievers not even being in the list. Let's see if there are big differences in the average rating.

# In[142]:


df_breed_group_mean = df[["rating_numerator", "breed"]].groupby("breed", as_index = False).mean()


# In[143]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "breed", y = "rating_numerator", data = df_breed_group_mean.sort_values("rating_numerator", ascending = False).iloc[:100], palette = "viridis")
ax.set_title("average rating_numerator per breed");
#https://code.i-harness.com/de/q/2135a8
ax.xaxis.set_ticklabels([])
plt.tight_layout()


# While the most breeds are on average on nearly the same level of rating, there is some outlier visible.

# In[144]:


df_breed_group_mean.sort_values("rating_numerator", ascending = False).head(10)


# In[145]:


df[["breed", "rating_numerator"]].sort_values("rating_numerator", ascending = False).head(5)


# In[146]:


len(df.query("breed == 'labrador retriever'"))


# In[147]:


len(df.query("breed == 'soft-coated wheaten terrier'"))


# In[148]:


df.query("breed != 'soft-coated wheaten terrier'").rating_numerator.mean()


# The soft-coated wheaten terrier	got a very high mean rating. In fact, the labrador retriever got overall the biggest rating with 165, but since there are a lot more tweets with labrador retriever than for the soft-coated wheaten terrier, the one big rating of the soft-coated wheaten terrier has a higher weight then the one of the labrador retriever (14 tweets of soft-coated wheaten terrier and 105 of labrador retriever).
# 
# Not taking the outlier into account, this leads us to an average rating of 11. Based on the number of posts, retweets, favorites and mean rating, we will give the title of "Most overall liked dog of this Twitter account and its community" to the labrador retriever.

# **2. How did the account develop (speaking about number of tweets, retweets, favorites, image number and length of the tweets)?**

# To answer this question we first have to extract time information out of the timestamp.

# In[149]:


df_time = df.copy()


# In[150]:


df_time.info()


# The timestamp is a string again, so I have to transform it again.

# In[151]:


df_time.timestamp = pd.to_datetime(df_time.timestamp)


# Let's first take a look on the day of the week.

# In[152]:


df_time["dow"] = df_time["timestamp"].apply(lambda x: x.dayofweek)


# In[153]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "dow", data = df_time, palette = "viridis")
ax.set_title("count of tweets in the dataset for each day of the week");

ax.set_ylim(0, 300)
#https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+5))


# We can see that most of the tweets are posted on monday. For tuesday to friday it is nearly the same number of posts. On the weekend the Twitter profile tweets a little bit less. 

# In[154]:


df_time.timestamp.min()


# In[155]:


df_time.timestamp.max()


# This dataset contains data from the end of 2015 to the August of 2017. Let's extract the month, year and hour information from the timestamp.

# In[156]:


#get the month out of the timestamp
df_time["month"] = df_time["timestamp"].apply(lambda x: x.month)
#get the year out of the timestamp
df_time["year"] = df_time["timestamp"].apply(lambda x: x.year)
#get the hour out of the timestamp
df_time["hour"] = df_time["timestamp"].apply(lambda x: x.hour)


# For the first graph I only want to take a look on the full year 2016.

# In[157]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "month", data = df_time.query("year == 2016"), palette = "viridis")
ax.set_title("count of tweets in the dataset for each month 2016");

ax.set_ylim(0,150)
#https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+1.5))


# Over the timeperiod of 2016 the number of post per months decreased. It went from 134 tweets in January to 52 in December. Does this mean, that the performance of this account is also decreasing?

# In[158]:


#create a timestamp containing month and year
df_time['month_year'] = pd.to_datetime(df["timestamp"]).dt.to_period('M')


# In[159]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "month_year", data = df_time.sort_values("month_year"), palette = "viridis", )
ax.set_title("count of tweets in the dataset for each year - month combination in this dataset");

ax.set_ylim(0, 300)
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+3))

plt.tight_layout()


# If we look at it over the whole timeperiod it becomes even more clear. In April 2016 the number of tweets dropped and since then it has a relatively stable level. To see if the performance of the Account decreased we will take a look on the favorites and retweets that the posts get. 

# In[160]:


df_time_groupby = df_time.groupby("month_year", as_index = False).sum()


# In[161]:


df_time_groupby.head()


# In[162]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "favorite_count", data = df_time_groupby, palette = "viridis")
ax.set_title("sum of favorites per month-year combination");
plt.tight_layout()


# In[163]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "retweet_count", data = df_time_groupby, palette = "viridis")
ax.set_title("sum of retweets per month-year combination");


# Interesting, while the number of tweets per month is decreasing, the favorites and retweets per month are increasing.

# In[164]:


df_time_groupby_mean = df_time.groupby("month_year", as_index = False).mean()


# In[165]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "favorite_count", data = df_time_groupby_mean, palette = "viridis")
ax.set_title("mean of favorites per month-year combination");
plt.tight_layout()


# In[166]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "retweet_count", data = df_time_groupby_mean, palette = "viridis")
ax.set_title("mean of retweets per month-year combination");
plt.tight_layout()


# If we look at the average number of favorites and retweets the clear uptrend gets even more clearclearer! Now let's see if the number of posted images per month or the average upper text range changed over time.

# In[167]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "img_num", data = df_time_groupby_mean, palette = "viridis")
ax.set_title("mean image number per month-year combination");
plt.tight_layout()


# For the images it seems pretty stable. There are months where are more and months where are less posted images, but overall there is no clean up- or downtrend visible.

# In[168]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "upper_text_range", data = df_time_groupby_mean, palette = "viridis")
ax.set_title("mean upper text range per month_year combination");


# In[169]:


df_time_groupby_mean.upper_text_range.mean()


# In[170]:


df_time_groupby_mean.iloc[:11].upper_text_range.mean()


# In[171]:


df_time_groupby_mean.iloc[11:].upper_text_range.mean()


# For the tweet length it seems like it increased over the second half of this dataset from an average of 106 to 113. 

# **3. Is there a pattern visible in the timing of the tweets?**

# In[172]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "hour", data = df_time.query("year == 2016"), palette = "viridis")
ax.set_title("count of tweets in the dataset for each hour in 2016");


# In[173]:


fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "hour", data = df_time.query("year == 2015"), palette = "viridis")
ax.set_title("count of tweets in the dataset for each hour in 2015");


# In years 2015 & 2016, the most posts are during the night between 0:00 - 5:00. Between 4:00 - 15:00 there is a very small amounts of tweets, no tweets between 7:00 - 12:00 and at 14:00. There are a few tweets after 14:00, but not as many as between 0:00 - 5:00.  

# <a id='#conclusion'></a>
# ## Summary and Conclusions

# Questions & **Answers:**
# 
# 1. Based on the predicted, most likely dog breed: Which breed gets retweeted and favorited the most overall?
# 
# **Labrador Retriever**.
# 
# 2. How did the account develop (number of tweets, retweets, favorites, image number and length of the tweets)?
#     
# **Number of tweets per month decreased, retweets and favorites show an uptrend. No clear trend for the image numbers. Length of the tweets with increasing trend toward to the maximum limit of 130 characters.**
#     
# 3. Is there a pattern visible in the timing of the tweets? 
#     
# **Between 5:00 - 15:00, there are nearly no tweets at all. The most tweets are between 0:00 - 4:00 and then between 15:00 - 23:00. Moreover, between 15:00 - 23:00 there are less tweets than between 0:00 - 4:00.**
#     
#     
#     
