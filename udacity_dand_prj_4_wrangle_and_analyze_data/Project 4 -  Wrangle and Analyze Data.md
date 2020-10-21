# Udacity
# Project 4: Wrangle and Analyze Data
Vasileios Garyfallos, April 2020

## Table of Contents
<ul>
    <li><a href="#intro">Introduction</a></li>
    <li><a href="#sources">Data Sources</a></li>
    <li><a href="#gathering">Data Gathering</a></li>
    <li><a href="#assessing">Data Assessing</a></li>
        <li><a href="#assessingsum">Assessing Summary</a></li>
    <li><a href="#cleaning">Data Cleaning</a></li>
    <li><a href="#analysis">Data Analysis</a></li>
    <li><a href="#conclusion">Summary and Conclusions</a></li>
</ul>

## 1. Introduction
For this project, it is asked to gather and analyze data (tweet archive) from the Twitter account <a href = "https://twitter.com/dog_rates?lang=de">"WeRateDogs"</a>. 

The data will be gathered using manual download, programmatical download and quering an API. 

After data gathering, data assessment is required. in order to define any issues regarding data cleanliness and tidiness. The next step will be to clean the data with the aim to get a clean and tidy dataset for the analysis.

<a id='sources'></a>
## Data Sources


>1. **Name:** WeRateDogs Twitter Archive (twitter-archive-enhanced.csv)</li>
><ul>   
>    <li><b>Source:</b> <a href = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/59a4e958_twitter-archive->enhanced/twitter-archive-enhanced.csv">Udacity</a></li>    

>    <li><b>Method of gathering:</b> Manual download</li>
></ul>

>2. **Name:** Tweet image predictions (image_predictions.tsv)</li>
><ul>   
>    <li><b>Source:</b> <a href="https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image->predictions.tsv">Udacity</a></li>     

>    <li><b>Method of gathering:</b> Programmatical download</li>
></ul>

>3. **Name:** Additional Twitter data (tweet_json.txt)
><ul>   
>    <li><b>Source:</b> <a href = "https://twitter.com/dog_rates">WeRateDogs</a></li>    

>    <li><b>Method of gathering:</b> API (Tweepy)</li>
></ul>


#### Import Python libraries:


```python
import requests
import numpy as np 
import pandas as pd 
import json 
import seaborn as sns
import re
import tweepy
import matplotlib.pyplot as plt
```

<a id='gathering'></a>
## Data Gathering

#### 1. WeRateDogs Twitter Archive (twitter-archive-enhanced.csv)

Downloaded the .csv file from Udacity and loaded it into a  Pandas dataframe.


```python
df_twitter = pd.read_csv("twitter-archive-enhanced.csv")

df_twitter.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892420643555336193</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 16:23:56 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Phineas. He's a mystical boy. Only eve...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892420643...</td>
      <td>13</td>
      <td>10</td>
      <td>Phineas</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 00:17:27 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Tilly. She's just checking pup on you....</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892177421...</td>
      <td>13</td>
      <td>10</td>
      <td>Tilly</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-31 00:18:03 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Archie. He is a rare Norwegian Pouncin...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891815181...</td>
      <td>12</td>
      <td>10</td>
      <td>Archie</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



#### 2. Tweet image predictions (image_predictions.tsv)

requested the file's URL and wrote the content of the response to a file.


```python
url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"

#get response
response = requests.get(url)

#write return to an image
with open("image_predictions.tsv", mode = "wb") as file:
    file.write(response.content)
```


```python
df_predict = pd.read_csv("image_predictions.tsv", sep='\t')

df_predict.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>666020888022790149</td>
      <td>https://pbs.twimg.com/media/CT4udn0WwAA0aMy.jpg</td>
      <td>1</td>
      <td>Welsh_springer_spaniel</td>
      <td>0.465074</td>
      <td>True</td>
      <td>collie</td>
      <td>0.156665</td>
      <td>True</td>
      <td>Shetland_sheepdog</td>
      <td>0.061428</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>666029285002620928</td>
      <td>https://pbs.twimg.com/media/CT42GRgUYAA5iDo.jpg</td>
      <td>1</td>
      <td>redbone</td>
      <td>0.506826</td>
      <td>True</td>
      <td>miniature_pinscher</td>
      <td>0.074192</td>
      <td>True</td>
      <td>Rhodesian_ridgeback</td>
      <td>0.072010</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>666033412701032449</td>
      <td>https://pbs.twimg.com/media/CT4521TWwAEvMyu.jpg</td>
      <td>1</td>
      <td>German_shepherd</td>
      <td>0.596461</td>
      <td>True</td>
      <td>malinois</td>
      <td>0.138584</td>
      <td>True</td>
      <td>bloodhound</td>
      <td>0.116197</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



#### 3. Additional Twitter data (tweet_json.txt)

I created a Twitter developer account, in order to gather the data from the Twitter API. Then pulled the data using tweepy. I wrote the data in a new file called "tweet_json.txt".


```python
#API keys
consumer_key = 'EMPTIED'
consumer_secret = 'EMPTIED'
access_token = 'EMPTIED'
access_secret = 'EMPTIED'

#access the API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
```

#get all the twitter ids in the df
twitter_ids = list(df_twitter.tweet_id.unique())

#https://realpython.com/python-json/
#save the gathered data to a file
with open("tweet_json.txt", "w") as file:
    for ids in twitter_ids:
        print(f"Gather id: {ids}")
        try:
            #get all the twitter status - extended mode gives us additional data
            tweet = api.get_status(ids, tweet_mode = "extended")
            #dump the json data to our file
            json.dump(tweet._json, file)
            #add a linebreak after each dump
            file.write('\n')
        except Exception as e:
            print(f"Error - id: {ids}" + str(e))
            




```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>retweeted</th>
      <th>display_text_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892420643555336193</td>
      <td>8287</td>
      <td>37931</td>
      <td>False</td>
      <td>[0, 85]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>6119</td>
      <td>32575</td>
      <td>False</td>
      <td>[0, 138]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>4054</td>
      <td>24530</td>
      <td>False</td>
      <td>[0, 121]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>891689557279858688</td>
      <td>8424</td>
      <td>41274</td>
      <td>False</td>
      <td>[0, 79]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>891327558926688256</td>
      <td>9128</td>
      <td>39464</td>
      <td>False</td>
      <td>[0, 138]</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_twitter.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892420643555336193</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 16:23:56 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Phineas. He's a mystical boy. Only eve...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892420643...</td>
      <td>13</td>
      <td>10</td>
      <td>Phineas</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_predict.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>666020888022790149</td>
      <td>https://pbs.twimg.com/media/CT4udn0WwAA0aMy.jpg</td>
      <td>1</td>
      <td>Welsh_springer_spaniel</td>
      <td>0.465074</td>
      <td>True</td>
      <td>collie</td>
      <td>0.156665</td>
      <td>True</td>
      <td>Shetland_sheepdog</td>
      <td>0.061428</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_api.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>retweeted</th>
      <th>display_text_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892420643555336193</td>
      <td>8287</td>
      <td>37931</td>
      <td>False</td>
      <td>[0, 85]</td>
    </tr>
  </tbody>
</table>
</div>



<a id='assessing'></a>
## Data Assessing


Assessing the quality/tidiness of the data.

#### df_twitter

Let's first look for missing data.


```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_twitter.isnull(), vmin=0, vmax = 1)
```


![png](output_22_0.png)


As we can see, there is a lot of missing data in the columns about the reply and the retweeted status. Since we only want original posts with images, we have to drop them later - the missing data in the "expanded_urls" column will also disappear with that cleaning operation.


```python
df_twitter.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1243</th>
      <td>711968124745228288</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-03-21 17:30:03 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>Meet Winston. He's trapped in a cup of coffee....</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/711968124...</td>
      <td>10</td>
      <td>10</td>
      <td>Winston</td>
      <td>None</td>
      <td>None</td>
      <td>pupper</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>704491224099647488</td>
      <td>7.044857e+17</td>
      <td>28785486.0</td>
      <td>2016-03-01 02:19:31 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>13/10 hero af\n@ABC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13</td>
      <td>10</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1955</th>
      <td>673636718965334016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2015-12-06 22:54:44 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is a Lofted Aphrodisiac Terrier named Kip...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/673636718...</td>
      <td>10</td>
      <td>10</td>
      <td>a</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>419</th>
      <td>822244816520155136</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-01-20 00:50:15 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>We only rate dogs. Please don't send pics of m...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/822244816...</td>
      <td>11</td>
      <td>10</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>222</th>
      <td>849668094696017920</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-04-05 17:00:34 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: This is Gidget. She's a spy pup...</td>
      <td>8.331247e+17</td>
      <td>4.196984e+09</td>
      <td>2017-02-19 01:23:00 +0000</td>
      <td>https://twitter.com/dog_rates/status/833124694...</td>
      <td>12</td>
      <td>10</td>
      <td>Gidget</td>
      <td>None</td>
      <td>None</td>
      <td>pupper</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



visual assessment:
- not all tweets could be classified as doggo, floofer, pupper or puppo and all columns contain "None"
- the source contains unnecessary HTML code
- there is the name "None" in the name column


```python
df_twitter.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2356 entries, 0 to 2355
    Data columns (total 17 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   tweet_id                    2356 non-null   int64  
     1   in_reply_to_status_id       78 non-null     float64
     2   in_reply_to_user_id         78 non-null     float64
     3   timestamp                   2356 non-null   object 
     4   source                      2356 non-null   object 
     5   text                        2356 non-null   object 
     6   retweeted_status_id         181 non-null    float64
     7   retweeted_status_user_id    181 non-null    float64
     8   retweeted_status_timestamp  181 non-null    object 
     9   expanded_urls               2297 non-null   object 
     10  rating_numerator            2356 non-null   int64  
     11  rating_denominator          2356 non-null   int64  
     12  name                        2356 non-null   object 
     13  doggo                       2356 non-null   object 
     14  floofer                     2356 non-null   object 
     15  pupper                      2356 non-null   object 
     16  puppo                       2356 non-null   object 
    dtypes: float64(4), int64(3), object(10)
    memory usage: 313.0+ KB
    

Also the datatypes are incorrect:
- tweet_id should be a str
- timestamp - columns should be datetime objects

Now let's see how many wrong names we can find.


```python
df_twitter.name.value_counts()
```




    None       745
    a           55
    Charlie     12
    Lucy        11
    Oliver      11
              ... 
    Blanket      1
    Maude        1
    Jazz         1
    Hamrick      1
    Harlso       1
    Name: name, Length: 957, dtype: int64



As we can see, the name column contains wrong names like "None", "a", "the", "an". 


```python
df_twitter[df_twitter.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



There are no duplicates in this data, so the number of unique tweet_ids should be the length of the df.


```python
df_twitter.tweet_id.nunique()
```




    2356



Proof. Now let's create a copy of the twitter dataframe for further assessing. I want to see in how many cases there were no classification of dog_size possible via text processing.


```python
df_twitter_assess = df_twitter.copy()
```


```python
#returns true if there is no dog classification in any of the columns
df_twitter_assess[["doggo","floofer","pupper","puppo"]].apply(lambda x: True if
    (x[0] == "None" and x[1] == "None" and x[2] == "None" and x[3] =="None") 
    else False, axis = 1).value_counts()
```




    True     1976
    False     380
    dtype: int64




```python
df_twitter_assess["doggo"].value_counts()
```




    None     2259
    doggo      97
    Name: doggo, dtype: int64




```python
df_twitter_assess["floofer"].value_counts()
```




    None       2346
    floofer      10
    Name: floofer, dtype: int64




```python
df_twitter_assess["pupper"].value_counts()
```




    None      2099
    pupper     257
    Name: pupper, dtype: int64




```python
df_twitter_assess["puppo"].value_counts()
```




    None     2326
    puppo      30
    Name: puppo, dtype: int64



Only for 16% of the rows the data is not missing. Now let's take a look at the ratings. By what we have seen so far, it looks like the ratings have always a format of 13/10 or 12/10 and so on. So we would expect a numerator > 10 and denominator = 10.


```python
df_twitter.rating_numerator.value_counts()
```




    12      558
    11      464
    10      461
    13      351
    9       158
    8       102
    7        55
    14       54
    5        37
    6        32
    3        19
    4        17
    1         9
    2         9
    420       2
    0         2
    15        2
    75        2
    80        1
    20        1
    24        1
    26        1
    44        1
    50        1
    60        1
    165       1
    84        1
    88        1
    144       1
    182       1
    143       1
    666       1
    960       1
    1776      1
    17        1
    27        1
    45        1
    99        1
    121       1
    204       1
    Name: rating_numerator, dtype: int64



We can observe that there is a wide range of numbers as rating_numerator, with a maximum of 1776. 


```python
print(df_twitter.query("rating_numerator == '1776'").text)
```

    979    This is Atticus. He's quite simply America af....
    Name: text, dtype: object
    

This rating is correct - so a lot higher values as 12 or 13 would be valid to this rating system. But what is with the very small ones?


```python
print(df_twitter.query("rating_numerator == '1'").text)
```

    605     RT @dog_rates: Not familiar with this breed. N...
    1446    After reading the comments I may have overesti...
    1869    What kind of person sends in a picture without...
    1940    The millennials have spoken and we've decided ...
    2038    After 22 minutes of careful deliberation this ...
    2091    Flamboyant pup here. Probably poisonous. Won't...
    2261    Never seen dog like this. Breathes heavy. Tilt...
    2335    This is an Albanian 3 1/2 legged  Episcopalian...
    2338    Not familiar with this breed. No tail (weird)....
    Name: text, dtype: object
    

The entry 605 shows, that these tweets contain pictures that don't contain any dogs. Also in the entry 2335 the rating got extracted wrongly (misinterpreted the 1/2 of 3 1/2 as the rating). 


```python
print(df_twitter.query("rating_numerator == '0'").text)
```

    315     When you're so blinded by your systematic plag...
    1016    PUPDATE: can't see any. Even if I could, I cou...
    Name: text, dtype: object
    

No clear doggo photos again. Let's make the same check for the denominator.


```python
df_twitter.rating_denominator.value_counts()
```




    10     2333
    11        3
    50        3
    80        2
    20        2
    2         1
    16        1
    40        1
    70        1
    15        1
    90        1
    110       1
    120       1
    130       1
    150       1
    170       1
    7         1
    0         1
    Name: rating_denominator, dtype: int64




```python
print(df_twitter.query("rating_denominator == '170'").text)
```

    1120    Say hello to this unbelievably well behaved sq...
    Name: text, dtype: object
    


```python
print(df_twitter.query("rating_denominator == '0'").text)
```

    313    @jonnysun @Lin_Manuel ok jomny I know you're e...
    Name: text, dtype: object
    


```python
print(df_twitter.query("rating_denominator == '7'").text)
```

    516    Meet Sam. She smiles 24/7 &amp; secretly aspir...
    Name: text, dtype: object
    

The same problems as for the numerator. Multiple Dogs or multiple occurences of the pattern \d+\/\d+. Lets extract this by our self and see what we get.


```python
pattern = "(\d+(\.\d+)?\/\d+(\.\d+)?)" #we could expect an integer rating on what we saw, but maybe some floats are the case

#https://stackoverflow.com/questions/36028932/how-to-extract-specific-content-in-a-pandas-dataframe-with-a-regex
df_twitter_assess["rating"] = df_twitter_assess.text.str.extract(pattern, expand = True)[0]

#https://stackoverflow.com/questions/14745022/how-to-split-a-column-into-two-columns
df_twitter_assess[['num', 'denom']] = df_twitter_assess['rating'].str.split('/', n=1, expand=True)
```


```python
df_twitter_assess.rating_numerator = df_twitter_assess.rating_numerator.astype("str")
df_twitter_assess.rating_denominator = df_twitter_assess.rating_denominator.astype("str")
```


```python
#look for differences in the original numerator and the newe extract
df_twitter_assess["check_num"] = df_twitter_assess[["rating_numerator", "num"]].apply(lambda x: False if (x[0] != x[1]) else True, axis = 1)
```


```python
df_twitter_assess.check_num.value_counts()
```




    True     2349
    False       7
    Name: check_num, dtype: int64




```python
df_twitter_assess.query("check_num == False")[["rating_numerator", "num","check_num"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating_numerator</th>
      <th>num</th>
      <th>check_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>5</td>
      <td>13.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>340</th>
      <td>75</td>
      <td>9.75</td>
      <td>False</td>
    </tr>
    <tr>
      <th>387</th>
      <td>7</td>
      <td>007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>695</th>
      <td>75</td>
      <td>9.75</td>
      <td>False</td>
    </tr>
    <tr>
      <th>763</th>
      <td>27</td>
      <td>11.27</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1689</th>
      <td>5</td>
      <td>9.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1712</th>
      <td>26</td>
      <td>11.26</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



These are the differences we found by extracting the first occurrence of the pattern. These ratings got transformed to integers and are therefore wrong. 


```python
df_twitter_assess[["rating_numerator", "num", "check_num"]].sample(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating_numerator</th>
      <th>num</th>
      <th>check_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2222</th>
      <td>4</td>
      <td>4</td>
      <td>True</td>
    </tr>
    <tr>
      <th>324</th>
      <td>12</td>
      <td>12</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2148</th>
      <td>8</td>
      <td>8</td>
      <td>True</td>
    </tr>
    <tr>
      <th>143</th>
      <td>13</td>
      <td>13</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1441</th>
      <td>9</td>
      <td>9</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2165</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>994</th>
      <td>12</td>
      <td>12</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>11</td>
      <td>11</td>
      <td>True</td>
    </tr>
    <tr>
      <th>796</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1611</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>933</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2123</th>
      <td>8</td>
      <td>8</td>
      <td>True</td>
    </tr>
    <tr>
      <th>393</th>
      <td>11</td>
      <td>11</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1082</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>138</th>
      <td>13</td>
      <td>13</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Let's repeat this for the denominator.


```python
df_twitter_assess["check_denom"] = df_twitter_assess[["rating_denominator", "denom"]].apply(lambda x: False if (x[0] != x[1]) else True, axis = 1)
```


```python
df_twitter_assess.check_denom.value_counts()
```




    True     2355
    False       1
    Name: check_denom, dtype: int64




```python
df_twitter_assess.query("check_denom == False")[["rating_denominator", "denom","check_denom"]] #problem with integer, maybe also floats?
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating_denominator</th>
      <th>denom</th>
      <th>check_denom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>313</th>
      <td>0</td>
      <td>00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



This seems like it is no problem that we have to worry about.


```python
df_twitter_assess[["rating_denominator", "denom", "check_denom"]].sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating_denominator</th>
      <th>denom</th>
      <th>check_denom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2260</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1442</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>253</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>738</th>
      <td>10</td>
      <td>10</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Now we should assess how often there are multiple occurences of the "rating pattern" in one tweet.


```python
df_twitter_assess["count"] = df_twitter_assess.text.str.count(pattern)
```


```python
df_twitter_assess["count"].value_counts()
```




    1    2323
    2      32
    3       1
    Name: count, dtype: int64




```python
#show the full text
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 4000)

df_twitter_assess[["text", "count"]].query("count != 1")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55</th>
      <td>@roushfenway These are good dogs but 17/10 is an emotional impulse rating. More like 13/10s</td>
      <td>2</td>
    </tr>
    <tr>
      <th>313</th>
      <td>@jonnysun @Lin_Manuel ok jomny I know you're excited but 960/00 isn't a valid rating, 13/10 is tho</td>
      <td>2</td>
    </tr>
    <tr>
      <th>561</th>
      <td>RT @dog_rates: "Yep... just as I suspected. You're not flossing." 12/10 and 11/10 for the pup not flossing https://t.co/SuXcI9B7pQ</td>
      <td>2</td>
    </tr>
    <tr>
      <th>766</th>
      <td>"Yep... just as I suspected. You're not flossing." 12/10 and 11/10 for the pup not flossing https://t.co/SuXcI9B7pQ</td>
      <td>2</td>
    </tr>
    <tr>
      <th>784</th>
      <td>RT @dog_rates: After so many requests, this is Bretagne. She was the last surviving 9/11 search dog, and our second ever 14/10. RIP https:/…</td>
      <td>2</td>
    </tr>
    <tr>
      <th>860</th>
      <td>RT @dog_rates: Meet Eve. She's a raging alcoholic 8/10 (would b 11/10 but pupper alcoholism is a tragic issue that I can't condone) https:/…</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>This is Bookstore and Seaweed. Bookstore is tired and Seaweed is an asshole. 10/10 and 7/10 respectively https://t.co/eUGjGjjFVJ</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>After so many requests, this is Bretagne. She was the last surviving 9/11 search dog, and our second ever 14/10. RIP https://t.co/XAVDNDaVgQ</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1165</th>
      <td>Happy 4/20 from the squad! 13/10 for all https://t.co/eV1diwds8a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1202</th>
      <td>This is Bluebert. He just saw that both #FinalFur match ups are split 50/50. Amazed af. 11/10 https://t.co/Kky1DPG4iq</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1222</th>
      <td>Meet Travis and Flurp. Travis is pretty chill but Flurp can't lie down properly. 10/10 &amp;amp; 8/10\nget it together Flurp https://t.co/Akzl5ynMmE</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>This is Socks. That water pup w the super legs just splashed him. Socks did not appreciate that. 9/10 and 2/10 https://t.co/8rc5I22bBf</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>This may be the greatest video I've ever been sent. 4/10 for Charles the puppy, 13/10 overall. (Vid by @stevenxx_) https://t.co/uaJmNgXR2P</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1465</th>
      <td>Meet Oliviér. He takes killer selfies. Has a dog of his own. It leaps at random &amp;amp; can't bark for shit. 10/10 &amp;amp; 5/10 https://t.co/6NgsQJuSBJ</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1508</th>
      <td>When bae says they can't go out but you see them with someone else that same night. 5/10 &amp;amp; 10/10 for heartbroken pup https://t.co/aenk0KpoWM</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1525</th>
      <td>This is Eriq. His friend just reminded him of last year's super bowl. Not cool friend\n10/10 for Eriq\n6/10 for friend https://t.co/PlEXTofdpf</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1538</th>
      <td>Meet Fynn &amp;amp; Taco. Fynn is an all-powerful leaf lord and Taco is in the wrong place at the wrong time. 11/10 &amp;amp; 10/10 https://t.co/MuqHPvtL8c</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1662</th>
      <td>This is Darrel. He just robbed a 7/11 and is in a high speed police chase. Was just spotted by the helicopter 10/10 https://t.co/7EsP8LmSp5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>Meet Tassy &amp;amp; Bee. Tassy is pretty chill, but Bee is convinced the Ruffles are haunted. 10/10 &amp;amp; 11/10 respectively https://t.co/fgORpmTN9C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1832</th>
      <td>These two pups just met and have instantly bonded. Spectacular scene. Mesmerizing af. 10/10 and 7/10 for blue dog https://t.co/gwryaJO4tC</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>Meet Rufio. He is unaware of the pink legless pupper wrapped around him. Might want to get that checked 10/10 &amp;amp; 4/10 https://t.co/KNfLnYPmYh</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>Two gorgeous dogs here. Little waddling dog is a rebel. Refuses to look at camera. Must be a preteen. 5/10 &amp;amp; 8/10 https://t.co/YPfw7oahbD</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>Meet Eve. She's a raging alcoholic 8/10 (would b 11/10 but pupper alcoholism is a tragic issue that I can't condone) https://t.co/U36HYQIijg</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>10/10 for dog. 7/10 for cat. 12/10 for human. Much skill. Would pet all https://t.co/uhx5gfpx5k</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2064</th>
      <td>Meet Holly. She's trying to teach small human-like pup about blocks but he's not paying attention smh. 11/10 &amp;amp; 8/10 https://t.co/RcksaUrGNu</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2113</th>
      <td>Meet Hank and Sully. Hank is very proud of the pumpkin they found and Sully doesn't give a shit. 11/10 and 8/10 https://t.co/cwoP1ftbrj</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2177</th>
      <td>Here we have Pancho and Peaches. Pancho is a Condoleezza Gryffindor, and Peaches is just an asshole. 10/10 &amp;amp; 7/10 https://t.co/Lh1BsJrWPp</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2216</th>
      <td>This is Spark. He's nervous. Other dog hasn't moved in a while. Won't come when called. Doesn't fetch well 8/10&amp;amp;1/10 https://t.co/stEodX9Aba</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2263</th>
      <td>This is Kial. Kial is either wearing a cape, which would be rad, or flashing us, which would be rude. 10/10 or 4/10 https://t.co/8zcwIoiuqR</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2272</th>
      <td>Two dogs in this one. Both are rare Jujitsu Pythagoreans. One slightly whiter than other. Long legs. 7/10 and 8/10 https://t.co/ITxxcc4v9y</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2298</th>
      <td>After much debate this dog is being upgraded to 10/10. I repeat 10/10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2306</th>
      <td>These are Peruvian Feldspars. Their names are Cupit and Prencer. Both resemble Rand Paul. Sick outfits 10/10 &amp;amp; 10/10 https://t.co/ZnEMHBsAs1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2335</th>
      <td>This is an Albanian 3 1/2 legged  Episcopalian. Loves well-polished hardwood flooring. Penis on the collar. 9/10 https://t.co/d9NcXFKwLv</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



We can see that:
- this data contains retweets (as mentioned before)
- sometimes there are multiple dogs/cats or else in one picture
- some of these ratings are not clear

#### df_predict

Let's also begin with the missing data first.


```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_predict.isnull(), vmin = 0, vmax = 1)
```


![png](output_74_0.png)


Looks fine. Now the visual assessment:


```python
df_predict.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2069</th>
      <td>891087950875897856</td>
      <td>https://pbs.twimg.com/media/DF3HwyEWsAABqE6.jpg</td>
      <td>1</td>
      <td>Chesapeake_Bay_retriever</td>
      <td>0.425595</td>
      <td>True</td>
      <td>Irish_terrier</td>
      <td>0.116317</td>
      <td>True</td>
      <td>Indian_elephant</td>
      <td>0.076902</td>
      <td>False</td>
    </tr>
    <tr>
      <th>322</th>
      <td>671866342182637568</td>
      <td>https://pbs.twimg.com/media/CVLy3zFWoAA93qJ.jpg</td>
      <td>1</td>
      <td>Labrador_retriever</td>
      <td>0.875614</td>
      <td>True</td>
      <td>Chihuahua</td>
      <td>0.032182</td>
      <td>True</td>
      <td>golden_retriever</td>
      <td>0.017232</td>
      <td>True</td>
    </tr>
    <tr>
      <th>493</th>
      <td>675707330206547968</td>
      <td>https://pbs.twimg.com/media/CWCYOqWUAAARmGr.jpg</td>
      <td>1</td>
      <td>bath_towel</td>
      <td>0.721933</td>
      <td>False</td>
      <td>Staffordshire_bullterrier</td>
      <td>0.059344</td>
      <td>True</td>
      <td>bagel</td>
      <td>0.035702</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>751598357617971201</td>
      <td>https://pbs.twimg.com/media/Cm42t5vXEAAv4CS.jpg</td>
      <td>1</td>
      <td>toy_poodle</td>
      <td>0.757756</td>
      <td>True</td>
      <td>miniature_poodle</td>
      <td>0.035150</td>
      <td>True</td>
      <td>Scottish_deerhound</td>
      <td>0.027698</td>
      <td>True</td>
    </tr>
    <tr>
      <th>263</th>
      <td>670792680469889025</td>
      <td>https://pbs.twimg.com/media/CU8iYi2WsAEaqQ0.jpg</td>
      <td>1</td>
      <td>brown_bear</td>
      <td>0.882426</td>
      <td>False</td>
      <td>toy_poodle</td>
      <td>0.031355</td>
      <td>True</td>
      <td>miniature_poodle</td>
      <td>0.025743</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>719551379208073216</td>
      <td>https://pbs.twimg.com/media/CfxcKU6W8AE-wEx.jpg</td>
      <td>1</td>
      <td>malamute</td>
      <td>0.873233</td>
      <td>True</td>
      <td>Siberian_husky</td>
      <td>0.076435</td>
      <td>True</td>
      <td>Eskimo_dog</td>
      <td>0.035745</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1332</th>
      <td>757725642876129280</td>
      <td>https://pbs.twimg.com/media/CoP7c4bWcAAr55g.jpg</td>
      <td>2</td>
      <td>seat_belt</td>
      <td>0.425176</td>
      <td>False</td>
      <td>Labrador_retriever</td>
      <td>0.128128</td>
      <td>True</td>
      <td>Siamese_cat</td>
      <td>0.091241</td>
      <td>False</td>
    </tr>
    <tr>
      <th>967</th>
      <td>706310011488698368</td>
      <td>https://pbs.twimg.com/media/Cc1RNHLW4AACG6H.jpg</td>
      <td>1</td>
      <td>Pembroke</td>
      <td>0.698165</td>
      <td>True</td>
      <td>Chihuahua</td>
      <td>0.105834</td>
      <td>True</td>
      <td>bloodhound</td>
      <td>0.062030</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1133</th>
      <td>728409960103686147</td>
      <td>https://pbs.twimg.com/media/ChvU_DwWMAArx5L.jpg</td>
      <td>1</td>
      <td>Siamese_cat</td>
      <td>0.478278</td>
      <td>False</td>
      <td>Saint_Bernard</td>
      <td>0.094246</td>
      <td>True</td>
      <td>king_penguin</td>
      <td>0.082157</td>
      <td>False</td>
    </tr>
    <tr>
      <th>713</th>
      <td>685325112850124800</td>
      <td>https://pbs.twimg.com/media/CYLDikFWEAAIy1y.jpg</td>
      <td>1</td>
      <td>golden_retriever</td>
      <td>0.586937</td>
      <td>True</td>
      <td>Labrador_retriever</td>
      <td>0.398260</td>
      <td>True</td>
      <td>kuvasz</td>
      <td>0.005410</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



We can see that:
- the predicitions are sometimes lowercase, sometimes uppercase
- there is an underscore instead of a whitespace between the words
- there are rows with no prediciton of a dog (neither in 1, 2 nor 3)


```python
df_predict.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2075 entries, 0 to 2074
    Data columns (total 12 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   tweet_id  2075 non-null   int64  
     1   jpg_url   2075 non-null   object 
     2   img_num   2075 non-null   int64  
     3   p1        2075 non-null   object 
     4   p1_conf   2075 non-null   float64
     5   p1_dog    2075 non-null   bool   
     6   p2        2075 non-null   object 
     7   p2_conf   2075 non-null   float64
     8   p2_dog    2075 non-null   bool   
     9   p3        2075 non-null   object 
     10  p3_conf   2075 non-null   float64
     11  p3_dog    2075 non-null   bool   
    dtypes: bool(3), float64(3), int64(2), object(4)
    memory usage: 152.1+ KB
    

- the tweet_id colum should again be string

The best way to find duplicates is to look at the jpg - url. If there are value counts > 1, then this data contains duplicates/retweets.


```python
df_predict.jpg_url.value_counts().head(10)
```




    https://pbs.twimg.com/media/CU3mITUWIAAfyQS.jpg    2
    https://pbs.twimg.com/media/CVgdFjNWEAAxmbq.jpg    2
    https://pbs.twimg.com/media/CeRoBaxWEAABi0X.jpg    2
    https://pbs.twimg.com/media/CsGnz64WYAEIDHJ.jpg    2
    https://pbs.twimg.com/media/Ck2d7tJWUAEPTL3.jpg    2
    https://pbs.twimg.com/media/CWyD2HGUYAQ1Xa7.jpg    2
    https://pbs.twimg.com/media/Cp6db4-XYAAMmqL.jpg    2
    https://pbs.twimg.com/media/CvyVxQRWEAAdSZS.jpg    2
    https://pbs.twimg.com/media/Cq9guJ5WgAADfpF.jpg    2
    https://pbs.twimg.com/media/CwJR1okWIAA6XMp.jpg    2
    Name: jpg_url, dtype: int64



As predicted - this data contains retweets. 


```python
df_predict[df_predict.tweet_id.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df_predict[df_predict.jpg_url == "https://pbs.twimg.com/media/CeRoBaxWEAABi0X.jpg"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1045</th>
      <td>712809025985978368</td>
      <td>https://pbs.twimg.com/media/CeRoBaxWEAABi0X.jpg</td>
      <td>1</td>
      <td>Labrador_retriever</td>
      <td>0.868671</td>
      <td>True</td>
      <td>carton</td>
      <td>0.095095</td>
      <td>False</td>
      <td>pug</td>
      <td>0.007651</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1594</th>
      <td>798697898615730177</td>
      <td>https://pbs.twimg.com/media/CeRoBaxWEAABi0X.jpg</td>
      <td>1</td>
      <td>Labrador_retriever</td>
      <td>0.868671</td>
      <td>True</td>
      <td>carton</td>
      <td>0.095095</td>
      <td>False</td>
      <td>pug</td>
      <td>0.007651</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_twitter[df_twitter.tweet_id == 798697898615730177]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>596</th>
      <td>798697898615730177</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-11-16 01:23:12 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" rel="nofollow"&gt;Twitter for iPhone&lt;/a&gt;</td>
      <td>RT @dog_rates: This is Stubert. He just arrived. 10/10 https://t.co/HVGs5aAKAn</td>
      <td>7.128090e+17</td>
      <td>4.196984e+09</td>
      <td>2016-03-24 01:11:29 +0000</td>
      <td>https://twitter.com/dog_rates/status/712809025985978368/photo/1,https://twitter.com/dog_rates/status/712809025985978368/photo/1</td>
      <td>10</td>
      <td>10</td>
      <td>Stubert</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



We only want Tweets with pictures which contain dogs. Let's see if there are pictures, for which the ML - Algorithm didn't predict any dogs.


```python
df_predict.query("p1_dog == False and p2_dog == False and p3_dog == False")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>666051853826850816</td>
      <td>https://pbs.twimg.com/media/CT5KoJ1WoAAJash.jpg</td>
      <td>1</td>
      <td>box_turtle</td>
      <td>0.933012</td>
      <td>False</td>
      <td>mud_turtle</td>
      <td>0.045885</td>
      <td>False</td>
      <td>terrapin</td>
      <td>0.017885</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>666104133288665088</td>
      <td>https://pbs.twimg.com/media/CT56LSZWoAAlJj2.jpg</td>
      <td>1</td>
      <td>hen</td>
      <td>0.965932</td>
      <td>False</td>
      <td>cock</td>
      <td>0.033919</td>
      <td>False</td>
      <td>partridge</td>
      <td>0.000052</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>666268910803644416</td>
      <td>https://pbs.twimg.com/media/CT8QCd1WEAADXws.jpg</td>
      <td>1</td>
      <td>desktop_computer</td>
      <td>0.086502</td>
      <td>False</td>
      <td>desk</td>
      <td>0.085547</td>
      <td>False</td>
      <td>bookcase</td>
      <td>0.079480</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>666293911632134144</td>
      <td>https://pbs.twimg.com/media/CT8mx7KW4AEQu8N.jpg</td>
      <td>1</td>
      <td>three-toed_sloth</td>
      <td>0.914671</td>
      <td>False</td>
      <td>otter</td>
      <td>0.015250</td>
      <td>False</td>
      <td>great_grey_owl</td>
      <td>0.013207</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>666362758909284353</td>
      <td>https://pbs.twimg.com/media/CT9lXGsUcAAyUFt.jpg</td>
      <td>1</td>
      <td>guinea_pig</td>
      <td>0.996496</td>
      <td>False</td>
      <td>skunk</td>
      <td>0.002402</td>
      <td>False</td>
      <td>hamster</td>
      <td>0.000461</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>880935762899988482</td>
      <td>https://pbs.twimg.com/media/DDm2Z5aXUAEDS2u.jpg</td>
      <td>1</td>
      <td>street_sign</td>
      <td>0.251801</td>
      <td>False</td>
      <td>umbrella</td>
      <td>0.115123</td>
      <td>False</td>
      <td>traffic_light</td>
      <td>0.069534</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>881268444196462592</td>
      <td>https://pbs.twimg.com/media/DDrk-f9WAAI-WQv.jpg</td>
      <td>1</td>
      <td>tusker</td>
      <td>0.473303</td>
      <td>False</td>
      <td>Indian_elephant</td>
      <td>0.245646</td>
      <td>False</td>
      <td>ibex</td>
      <td>0.055661</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2046</th>
      <td>886680336477933568</td>
      <td>https://pbs.twimg.com/media/DE4fEDzWAAAyHMM.jpg</td>
      <td>1</td>
      <td>convertible</td>
      <td>0.738995</td>
      <td>False</td>
      <td>sports_car</td>
      <td>0.139952</td>
      <td>False</td>
      <td>car_wheel</td>
      <td>0.044173</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2052</th>
      <td>887517139158093824</td>
      <td>https://pbs.twimg.com/ext_tw_video_thumb/887517108413886465/pu/img/WanJKwssZj4VJvL9.jpg</td>
      <td>1</td>
      <td>limousine</td>
      <td>0.130432</td>
      <td>False</td>
      <td>tow_truck</td>
      <td>0.029175</td>
      <td>False</td>
      <td>shopping_cart</td>
      <td>0.026321</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2074</th>
      <td>892420643555336193</td>
      <td>https://pbs.twimg.com/media/DGKD1-bXoAAIAUK.jpg</td>
      <td>1</td>
      <td>orange</td>
      <td>0.097049</td>
      <td>False</td>
      <td>bagel</td>
      <td>0.085851</td>
      <td>False</td>
      <td>banana</td>
      <td>0.076110</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>324 rows × 12 columns</p>
</div>




```python
df_predict.query("p1_dog == False and (p2_dog == True or p3_dog == True)")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>666057090499244032</td>
      <td>https://pbs.twimg.com/media/CT5PY90WoAAQGLo.jpg</td>
      <td>1</td>
      <td>shopping_cart</td>
      <td>0.962465</td>
      <td>False</td>
      <td>shopping_basket</td>
      <td>0.014594</td>
      <td>False</td>
      <td>golden_retriever</td>
      <td>0.007959</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>666337882303524864</td>
      <td>https://pbs.twimg.com/media/CT9OwFIWEAMuRje.jpg</td>
      <td>1</td>
      <td>ox</td>
      <td>0.416669</td>
      <td>False</td>
      <td>Newfoundland</td>
      <td>0.278407</td>
      <td>True</td>
      <td>groenendael</td>
      <td>0.102643</td>
      <td>True</td>
    </tr>
    <tr>
      <th>33</th>
      <td>666430724426358785</td>
      <td>https://pbs.twimg.com/media/CT-jNYqW4AAPi2M.jpg</td>
      <td>1</td>
      <td>llama</td>
      <td>0.505184</td>
      <td>False</td>
      <td>Irish_terrier</td>
      <td>0.104109</td>
      <td>True</td>
      <td>dingo</td>
      <td>0.062071</td>
      <td>False</td>
    </tr>
    <tr>
      <th>43</th>
      <td>666776908487630848</td>
      <td>https://pbs.twimg.com/media/CUDeDoWUYAAD-EM.jpg</td>
      <td>1</td>
      <td>seat_belt</td>
      <td>0.375057</td>
      <td>False</td>
      <td>miniature_pinscher</td>
      <td>0.167175</td>
      <td>True</td>
      <td>Chihuahua</td>
      <td>0.086951</td>
      <td>True</td>
    </tr>
    <tr>
      <th>52</th>
      <td>666996132027977728</td>
      <td>https://pbs.twimg.com/media/CUGlb6iUwAITEbW.jpg</td>
      <td>1</td>
      <td>hay</td>
      <td>0.507637</td>
      <td>False</td>
      <td>Rottweiler</td>
      <td>0.062490</td>
      <td>True</td>
      <td>water_buffalo</td>
      <td>0.048425</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>872122724285648897</td>
      <td>https://pbs.twimg.com/media/DBpm-5UXcAUeCru.jpg</td>
      <td>1</td>
      <td>basketball</td>
      <td>0.808396</td>
      <td>False</td>
      <td>pug</td>
      <td>0.066736</td>
      <td>True</td>
      <td>dalmatian</td>
      <td>0.054570</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>873697596434513921</td>
      <td>https://pbs.twimg.com/media/DA7iHL5U0AA1OQo.jpg</td>
      <td>1</td>
      <td>laptop</td>
      <td>0.153718</td>
      <td>False</td>
      <td>French_bulldog</td>
      <td>0.099984</td>
      <td>True</td>
      <td>printer</td>
      <td>0.077130</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>879376492567855104</td>
      <td>https://pbs.twimg.com/media/DDQsQGFV0AAw6u9.jpg</td>
      <td>1</td>
      <td>tricycle</td>
      <td>0.663601</td>
      <td>False</td>
      <td>Labrador_retriever</td>
      <td>0.033496</td>
      <td>True</td>
      <td>Pembroke</td>
      <td>0.018827</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>882045870035918850</td>
      <td>https://pbs.twimg.com/media/DD2oCl2WAAEI_4a.jpg</td>
      <td>1</td>
      <td>web_site</td>
      <td>0.949591</td>
      <td>False</td>
      <td>dhole</td>
      <td>0.017326</td>
      <td>False</td>
      <td>golden_retriever</td>
      <td>0.006941</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2071</th>
      <td>891689557279858688</td>
      <td>https://pbs.twimg.com/media/DF_q7IAWsAEuuN8.jpg</td>
      <td>1</td>
      <td>paper_towel</td>
      <td>0.170278</td>
      <td>False</td>
      <td>Labrador_retriever</td>
      <td>0.168086</td>
      <td>True</td>
      <td>spatula</td>
      <td>0.040836</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>219 rows × 12 columns</p>
</div>



After checking some of these pictures it gets clear, that sometimes the doggos are in the background or the pictures doesn't contain any dogs at all.

**df_api**

Lets repeat the same pattern for this dataset: missing data → visual assessment.


```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_api.isnull(), vmin = 0, vmax = 1)
```


![png](output_92_0.png)



```python
df_api.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>retweeted</th>
      <th>display_text_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>906</th>
      <td>756303284449767430</td>
      <td>1163</td>
      <td>4159</td>
      <td>False</td>
      <td>[0, 46]</td>
    </tr>
    <tr>
      <th>883</th>
      <td>759047813560868866</td>
      <td>2175</td>
      <td>6865</td>
      <td>False</td>
      <td>[0, 84]</td>
    </tr>
    <tr>
      <th>1592</th>
      <td>685663452032069632</td>
      <td>1560</td>
      <td>3405</td>
      <td>False</td>
      <td>[0, 109]</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>710833117892898816</td>
      <td>569</td>
      <td>2802</td>
      <td>False</td>
      <td>[0, 120]</td>
    </tr>
    <tr>
      <th>182</th>
      <td>855862651834028034</td>
      <td>27</td>
      <td>348</td>
      <td>False</td>
      <td>[14, 86]</td>
    </tr>
    <tr>
      <th>1809</th>
      <td>676470639084101634</td>
      <td>4898</td>
      <td>11956</td>
      <td>False</td>
      <td>[0, 66]</td>
    </tr>
    <tr>
      <th>2151</th>
      <td>669359674819481600</td>
      <td>127</td>
      <td>373</td>
      <td>False</td>
      <td>[0, 139]</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>699691744225525762</td>
      <td>4879</td>
      <td>10699</td>
      <td>False</td>
      <td>[0, 140]</td>
    </tr>
    <tr>
      <th>535</th>
      <td>805823200554876929</td>
      <td>8792</td>
      <td>0</td>
      <td>False</td>
      <td>[0, 114]</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>672264251789176834</td>
      <td>346</td>
      <td>1160</td>
      <td>False</td>
      <td>[0, 123]</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_api.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2340 entries, 0 to 2339
    Data columns (total 5 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   tweet_id            2340 non-null   int64 
     1   retweet_count       2340 non-null   int64 
     2   favorite_count      2340 non-null   int64 
     3   retweeted           2340 non-null   bool  
     4   display_text_range  2340 non-null   object
    dtypes: bool(1), int64(3), object(1)
    memory usage: 75.5+ KB
    

Overall it looks good for this dataset. There is no missing data - only the datatype of the tweet_id should again be a string.

<a id='assessingsum'></a>
### Assessing Summary

#### Quality
##### `df_twitter` table
- the datatype of the id - columns is integer and should be str
- the datatype of the timestamp - column is object and should be datetime
- some of the dogs are not classified as one of "doggo", "floofer", "pupper" or "puppo" and contain all "None" instead
- some of the dog names are not correct (None, an, by, a, ...)
- contains retweets
- some of the ratings are not correctly extracted (mostly if there are >1 entries with the pattern "(\d+(\.\d+)?\/\d+(\.\d+)?)"
- also transforming the ratings to integer created some mistakes (there are also floats)
- the source column contains html code

##### `df_predict` table
- the datatype of the id - columns is integer and should be str
- contains retweets (duplicated rows in column `jpg_url`)
- there are pictures in this table that are not dogs
- the predictions are sometimes uppercase, sometimes lowercase
- also there is a "_" instead of a whitespace in the predictions

##### `df_api` table
- the datatype of the id - columns is integer and should be str

#### Tidiness
##### `df_twitter` table
- the columns `doggo`, `floofer`,`pupper` and `puppo` are not easy to analyze and should be in one column

##### `df_predict` table
- the prediction and confidence columns should be reduced to two columns - one for the prediction with the highest confidence (dog)

##### `df_api` table
- display_text_range contains 2 variables

##### `all` tables

- All three tables share the column `tweet_id` and should be merged together.

<a id='cleaning'></a>
## Data Cleaning

Cleaning steps:

<ol>
    <li>Merge the tables together</li>
    <li>Drop the replies, retweets and the corresponding columns and also drop the tweets without an image or with images which don't display doggos</li>
    <li>Clean the datatypes of the columns</li>
    <li>Clean the wrong numerators - the floats on the one hand (replacement), the ones with multiple occurence of the pattern on the other (drop)</li>
    <li>Extract the source from html code</li>
    <li>Split the text range into two separate columns</li>
    <li>Remove the "None" out of the doggo, floofer, pupper and puppo column and merge them into one column</li>
    <li>Remove the wrong names of name column</li>
    <li>Reduce the prediction columns into two - breed and conf</li>
    <li>Clean the new breed column by replacing the "_" with a whitespace and make them all lowercase</li>
</ol>

**1. Merge the tables together**

I could clean all the tables one by one, but all of them share some cleaning needs or are dependent on each other to do so (for example removing of retweets or pictures not containig dogs). By merging them all together as first step, I can save some coding time and avoid repetition. 


```python
#outer join to not loose rows at first
df_master = pd.merge(df_twitter, df_api, on = "tweet_id", how = "outer")
```


```python
df_master = pd.merge(df_master, df_predict, on = "tweet_id", how = "outer")
```


```python
df_master_clean = df_master.copy()
```


```python
df_master_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2356 entries, 0 to 2355
    Data columns (total 32 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   tweet_id                    2356 non-null   int64  
     1   in_reply_to_status_id       78 non-null     float64
     2   in_reply_to_user_id         78 non-null     float64
     3   timestamp                   2356 non-null   object 
     4   source                      2356 non-null   object 
     5   text                        2356 non-null   object 
     6   retweeted_status_id         181 non-null    float64
     7   retweeted_status_user_id    181 non-null    float64
     8   retweeted_status_timestamp  181 non-null    object 
     9   expanded_urls               2297 non-null   object 
     10  rating_numerator            2356 non-null   int64  
     11  rating_denominator          2356 non-null   int64  
     12  name                        2356 non-null   object 
     13  doggo                       2356 non-null   object 
     14  floofer                     2356 non-null   object 
     15  pupper                      2356 non-null   object 
     16  puppo                       2356 non-null   object 
     17  retweet_count               2340 non-null   float64
     18  favorite_count              2340 non-null   float64
     19  retweeted                   2340 non-null   object 
     20  display_text_range          2340 non-null   object 
     21  jpg_url                     2075 non-null   object 
     22  img_num                     2075 non-null   float64
     23  p1                          2075 non-null   object 
     24  p1_conf                     2075 non-null   float64
     25  p1_dog                      2075 non-null   object 
     26  p2                          2075 non-null   object 
     27  p2_conf                     2075 non-null   float64
     28  p2_dog                      2075 non-null   object 
     29  p3                          2075 non-null   object 
     30  p3_conf                     2075 non-null   float64
     31  p3_dog                      2075 non-null   object 
    dtypes: float64(10), int64(3), object(19)
    memory usage: 607.4+ KB
    

**2. Drop the replies, retweets and the corresponding columns and also drop the tweets without an image or with images which don't display doggos**

Now that this is done, we have to remove the replies, retweets and the tweets withouth an image displaying a dog, because we only want original tweets with images. Let's visualize our missing data first.


```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)
```


![png](output_109_0.png)



```python
pd.set_option('display.max_colwidth', 50)
df_master_clean[df_master_clean["retweeted"].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>retweeted</th>
      <th>display_text_range</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>888202515573088257</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-21 01:02:36 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: This is Canela. She attempted s...</td>
      <td>8.874740e+17</td>
      <td>4.196984e+09</td>
      <td>2017-07-19 00:47:34 +0000</td>
      <td>https://twitter.com/dog_rates/status/887473957...</td>
      <td>13</td>
      <td>10</td>
      <td>Canela</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/media/DFDw2tyUQAAAFke.jpg</td>
      <td>2.0</td>
      <td>Pembroke</td>
      <td>0.809197</td>
      <td>True</td>
      <td>Rhodesian_ridgeback</td>
      <td>0.054950</td>
      <td>True</td>
      <td>beagle</td>
      <td>0.038915</td>
      <td>True</td>
    </tr>
    <tr>
      <th>95</th>
      <td>873697596434513921</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-06-11 00:25:14 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: This is Walter. He won't start ...</td>
      <td>8.688804e+17</td>
      <td>4.196984e+09</td>
      <td>2017-05-28 17:23:24 +0000</td>
      <td>https://twitter.com/dog_rates/status/868880397...</td>
      <td>14</td>
      <td>10</td>
      <td>Walter</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/media/DA7iHL5U0AA1OQo.jpg</td>
      <td>1.0</td>
      <td>laptop</td>
      <td>0.153718</td>
      <td>False</td>
      <td>French_bulldog</td>
      <td>0.099984</td>
      <td>True</td>
      <td>printer</td>
      <td>0.077130</td>
      <td>False</td>
    </tr>
    <tr>
      <th>101</th>
      <td>872668790621863937</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-06-08 04:17:07 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @loganamnosis: Penelope here is doing me qu...</td>
      <td>8.726576e+17</td>
      <td>1.547674e+08</td>
      <td>2017-06-08 03:32:35 +0000</td>
      <td>https://twitter.com/loganamnosis/status/872657...</td>
      <td>14</td>
      <td>10</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>118</th>
      <td>869988702071779329</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-05-31 18:47:24 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: We only rate dogs. This is quit...</td>
      <td>8.591970e+17</td>
      <td>4.196984e+09</td>
      <td>2017-05-02 00:04:57 +0000</td>
      <td>https://twitter.com/dog_rates/status/859196978...</td>
      <td>12</td>
      <td>10</td>
      <td>quite</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>132</th>
      <td>866816280283807744</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-05-23 00:41:20 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: This is Jamesy. He gives a kiss...</td>
      <td>8.664507e+17</td>
      <td>4.196984e+09</td>
      <td>2017-05-22 00:28:40 +0000</td>
      <td>https://twitter.com/dog_rates/status/866450705...</td>
      <td>13</td>
      <td>10</td>
      <td>Jamesy</td>
      <td>None</td>
      <td>None</td>
      <td>pupper</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>155</th>
      <td>861769973181624320</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-05-09 02:29:07 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: "Good afternoon class today we'...</td>
      <td>8.066291e+17</td>
      <td>4.196984e+09</td>
      <td>2016-12-07 22:38:52 +0000</td>
      <td>https://twitter.com/dog_rates/status/806629075...</td>
      <td>13</td>
      <td>10</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/media/CzG425nWgAAnP7P.jpg</td>
      <td>2.0</td>
      <td>Arabian_camel</td>
      <td>0.366248</td>
      <td>False</td>
      <td>house_finch</td>
      <td>0.209852</td>
      <td>False</td>
      <td>cocker_spaniel</td>
      <td>0.046403</td>
      <td>True</td>
    </tr>
    <tr>
      <th>247</th>
      <td>845459076796616705</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-03-25 02:15:26 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: Here's a heartwarming scene of ...</td>
      <td>7.562885e+17</td>
      <td>4.196984e+09</td>
      <td>2016-07-22 00:43:32 +0000</td>
      <td>https://twitter.com/dog_rates/status/756288534...</td>
      <td>12</td>
      <td>10</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>260</th>
      <td>842892208864923648</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-03-18 00:15:37 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: This is Stephan. He just wants ...</td>
      <td>8.071068e+17</td>
      <td>4.196984e+09</td>
      <td>2016-12-09 06:17:20 +0000</td>
      <td>https://twitter.com/dog_rates/status/807106840...</td>
      <td>13</td>
      <td>10</td>
      <td>Stephan</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/ext_tw_video_thumb/80710...</td>
      <td>1.0</td>
      <td>Chihuahua</td>
      <td>0.505370</td>
      <td>True</td>
      <td>Pomeranian</td>
      <td>0.120358</td>
      <td>True</td>
      <td>toy_terrier</td>
      <td>0.077008</td>
      <td>True</td>
    </tr>
    <tr>
      <th>298</th>
      <td>837012587749474308</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-03-01 18:52:06 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @KennyFromDaBlok: 14/10 h*ckin good hats. w...</td>
      <td>8.370113e+17</td>
      <td>7.266347e+08</td>
      <td>2017-03-01 18:47:10 +0000</td>
      <td>https://twitter.com/KennyFromDaBlok/status/837...</td>
      <td>14</td>
      <td>10</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/media/C52pYJXWgAA2BEf.jpg</td>
      <td>1.0</td>
      <td>toilet_tissue</td>
      <td>0.186387</td>
      <td>False</td>
      <td>cowboy_hat</td>
      <td>0.158555</td>
      <td>False</td>
      <td>sombrero</td>
      <td>0.149470</td>
      <td>False</td>
    </tr>
    <tr>
      <th>382</th>
      <td>827228250799742977</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-02-02 18:52:38 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: This is Phil. He's an important...</td>
      <td>6.946697e+17</td>
      <td>4.196984e+09</td>
      <td>2016-02-02 23:52:22 +0000</td>
      <td>https://twitter.com/dog_rates/status/694669722...</td>
      <td>12</td>
      <td>10</td>
      <td>Phil</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>506</th>
      <td>812747805718642688</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-12-24 19:52:31 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: Meet Sammy. At first I was like...</td>
      <td>6.800555e+17</td>
      <td>4.196984e+09</td>
      <td>2015-12-24 16:00:30 +0000</td>
      <td>https://twitter.com/dog_rates/status/680055455...</td>
      <td>10</td>
      <td>10</td>
      <td>Sammy</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>566</th>
      <td>802247111496568832</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-11-25 20:26:31 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: Everybody drop what you're doin...</td>
      <td>7.790561e+17</td>
      <td>4.196984e+09</td>
      <td>2016-09-22 20:33:42 +0000</td>
      <td>https://twitter.com/dog_rates/status/779056095...</td>
      <td>13</td>
      <td>10</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/media/Cs_DYr1XEAA54Pu.jpg</td>
      <td>1.0</td>
      <td>Chihuahua</td>
      <td>0.721188</td>
      <td>True</td>
      <td>toy_terrier</td>
      <td>0.112943</td>
      <td>True</td>
      <td>kelpie</td>
      <td>0.053365</td>
      <td>True</td>
    </tr>
    <tr>
      <th>784</th>
      <td>775096608509886464</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-09-11 22:20:06 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: After so many requests, this is...</td>
      <td>7.403732e+17</td>
      <td>4.196984e+09</td>
      <td>2016-06-08 02:41:38 +0000</td>
      <td>https://twitter.com/dog_rates/status/740373189...</td>
      <td>9</td>
      <td>11</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>818</th>
      <td>770743923962707968</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-08-30 22:04:05 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @dog_rates: Here's a doggo blowing bubbles....</td>
      <td>7.392382e+17</td>
      <td>4.196984e+09</td>
      <td>2016-06-04 23:31:25 +0000</td>
      <td>https://twitter.com/dog_rates/status/739238157...</td>
      <td>13</td>
      <td>10</td>
      <td>None</td>
      <td>doggo</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>932</th>
      <td>754011816964026368</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-07-15 17:56:40 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Charlie. He pouts until he gets to go ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/754011816...</td>
      <td>12</td>
      <td>10</td>
      <td>Charlie</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/media/CnbJuPoXEAAjcVF.jpg</td>
      <td>1.0</td>
      <td>French_bulldog</td>
      <td>0.600985</td>
      <td>True</td>
      <td>Boston_bull</td>
      <td>0.273176</td>
      <td>True</td>
      <td>boxer</td>
      <td>0.056772</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1726</th>
      <td>680055455951884288</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2015-12-24 16:00:30 +0000</td>
      <td>&lt;a href="https://about.twitter.com/products/tw...</td>
      <td>Meet Sammy. At first I was like "that's a snow...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/680055455...</td>
      <td>10</td>
      <td>10</td>
      <td>Sammy</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/media/CW-ZRC_WQAAyFrL.jpg</td>
      <td>1.0</td>
      <td>Samoyed</td>
      <td>0.995466</td>
      <td>True</td>
      <td>Great_Pyrenees</td>
      <td>0.001834</td>
      <td>True</td>
      <td>Pomeranian</td>
      <td>0.000667</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
#we only want the rows without an entry in "retweeted_status_id" in our master dataframe
df_master_clean = df_master_clean[df_master_clean["retweeted_status_id"].isnull()]
```


```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)
```


![png](output_112_0.png)



```python
df_master_clean.columns
```




    Index(['tweet_id', 'in_reply_to_status_id', 'in_reply_to_user_id', 'timestamp',
           'source', 'text', 'retweeted_status_id', 'retweeted_status_user_id',
           'retweeted_status_timestamp', 'expanded_urls', 'rating_numerator',
           'rating_denominator', 'name', 'doggo', 'floofer', 'pupper', 'puppo',
           'retweet_count', 'favorite_count', 'retweeted', 'display_text_range',
           'jpg_url', 'img_num', 'p1', 'p1_conf', 'p1_dog', 'p2', 'p2_conf',
           'p2_dog', 'p3', 'p3_conf', 'p3_dog'],
          dtype='object')




```python
#check with the column from the api table, no retweets left
df_master_clean.retweeted.value_counts()
```




    False    2173
    Name: retweeted, dtype: int64




```python
#same as for the retweets, we only want the rows without an entry in "in_reply_to_status_id"
df_master_clean = df_master_clean[df_master_clean.in_reply_to_status_id.isnull()]
```


```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)
```


![png](output_116_0.png)


During the gathering process from the API there were some tweets, which got deleted by the account. We will also drop them out of our master dataframe.


```python
df_master_clean.dropna(subset = ["retweeted"], inplace = True)
```


```python
#drop the unneeded columns
df_master_clean.drop(["in_reply_to_status_id", "in_reply_to_user_id",
                      "retweeted_status_id", "retweeted_status_user_id", 
                      "retweeted_status_timestamp", "retweeted"], inplace=True, axis = 1)
```


```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)
```


![png](output_120_0.png)


Now we want to take a look on the "jpg_url" column and drop all the rows, which are NAN - since these are the ones without an image. To check that, we could read in the image data from the gathered API data - but I will leave this for another time.


```python
df_master_clean.dropna(subset = ["jpg_url"], inplace = True)
```


```python
#check if there are still duplicated images after dropping the replies and the retweets
sum(df_master_clean.jpg_url.duplicated())
```




    0




```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)
```


![png](output_124_0.png)


The last step here is to drop the rows which contain images, that are not displaying any dogs (relying on the top three predictions of the ML algorithm).


```python
df_master_clean.drop(df_master_clean.query("p1_dog == False and p2_dog == False and p3_dog == False").index, inplace = True)
```


```python
df_master_clean.query("p1_dog == False and p2_dog == False and p3_dog == False")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>display_text_range</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df_master_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1664 entries, 1 to 2355
    Data columns (total 26 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   tweet_id            1664 non-null   int64  
     1   timestamp           1664 non-null   object 
     2   source              1664 non-null   object 
     3   text                1664 non-null   object 
     4   expanded_urls       1664 non-null   object 
     5   rating_numerator    1664 non-null   int64  
     6   rating_denominator  1664 non-null   int64  
     7   name                1664 non-null   object 
     8   doggo               1664 non-null   object 
     9   floofer             1664 non-null   object 
     10  pupper              1664 non-null   object 
     11  puppo               1664 non-null   object 
     12  retweet_count       1664 non-null   float64
     13  favorite_count      1664 non-null   float64
     14  display_text_range  1664 non-null   object 
     15  jpg_url             1664 non-null   object 
     16  img_num             1664 non-null   float64
     17  p1                  1664 non-null   object 
     18  p1_conf             1664 non-null   float64
     19  p1_dog              1664 non-null   object 
     20  p2                  1664 non-null   object 
     21  p2_conf             1664 non-null   float64
     22  p2_dog              1664 non-null   object 
     23  p3                  1664 non-null   object 
     24  p3_conf             1664 non-null   float64
     25  p3_dog              1664 non-null   object 
    dtypes: float64(6), int64(3), object(17)
    memory usage: 351.0+ KB
    

**3. Clean the datatypes of the columns**


```python
df_master_clean["tweet_id"] = df_master_clean["tweet_id"].astype("str")
```


```python
#transform the timestamp to datetime
df_master_clean["timestamp"] = pd.to_datetime(df_master_clean.timestamp)
```


```python
for x in ["retweet_count", "favorite_count", "img_num"]:
    df_master_clean[x] = df_master_clean[x].astype("int64")
```


```python
df_master_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1664 entries, 1 to 2355
    Data columns (total 26 columns):
     #   Column              Non-Null Count  Dtype              
    ---  ------              --------------  -----              
     0   tweet_id            1664 non-null   object             
     1   timestamp           1664 non-null   datetime64[ns, UTC]
     2   source              1664 non-null   object             
     3   text                1664 non-null   object             
     4   expanded_urls       1664 non-null   object             
     5   rating_numerator    1664 non-null   int64              
     6   rating_denominator  1664 non-null   int64              
     7   name                1664 non-null   object             
     8   doggo               1664 non-null   object             
     9   floofer             1664 non-null   object             
     10  pupper              1664 non-null   object             
     11  puppo               1664 non-null   object             
     12  retweet_count       1664 non-null   int64              
     13  favorite_count      1664 non-null   int64              
     14  display_text_range  1664 non-null   object             
     15  jpg_url             1664 non-null   object             
     16  img_num             1664 non-null   int64              
     17  p1                  1664 non-null   object             
     18  p1_conf             1664 non-null   float64            
     19  p1_dog              1664 non-null   object             
     20  p2                  1664 non-null   object             
     21  p2_conf             1664 non-null   float64            
     22  p2_dog              1664 non-null   object             
     23  p3                  1664 non-null   object             
     24  p3_conf             1664 non-null   float64            
     25  p3_dog              1664 non-null   object             
    dtypes: datetime64[ns, UTC](1), float64(3), int64(5), object(17)
    memory usage: 351.0+ KB
    

**4. Clean the wrong numerators - the floats on the one hand, the ones with multiple occurence of the pattern on the other**

While assessing the dataset, we found out, that floating numbers got transformed into integers, which lead to loss of information.


```python
df_twitter_assess.query("check_num == False")[["rating_numerator", "num","check_num"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating_numerator</th>
      <th>num</th>
      <th>check_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>5</td>
      <td>13.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>340</th>
      <td>75</td>
      <td>9.75</td>
      <td>False</td>
    </tr>
    <tr>
      <th>387</th>
      <td>7</td>
      <td>007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>695</th>
      <td>75</td>
      <td>9.75</td>
      <td>False</td>
    </tr>
    <tr>
      <th>763</th>
      <td>27</td>
      <td>11.27</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1689</th>
      <td>5</td>
      <td>9.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1712</th>
      <td>26</td>
      <td>11.26</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



We dropped a lot of rows, so we cannot be sure that all of these problems are still in this dataset, so we will extract it again.


```python
pattern = "(\d+\.\d+\/\d+)"

df_master_clean.text.str.extract(pattern, expand = True)[0].dropna()
```




    45       13.5/10
    695      9.75/10
    763     11.27/10
    1712    11.26/10
    Name: 0, dtype: object




```python
#get the right numerator out of the string
df_num_clean = df_master_clean.text.str.extract(pattern, expand = True)[0].dropna().str.split('/', n=1, expand=True)[0]
```


```python
df_num_clean
```




    45       13.5
    695      9.75
    763     11.27
    1712    11.26
    Name: 0, dtype: object




```python
#get the index of the wrong data
df_num_clean_index = df_num_clean.index
df_num_clean_values = df_num_clean.values.astype("float64")
```

Now that we have our data together, we can impute these values and clean this data.


```python
#transform the datatypes to float
df_master_clean.rating_numerator = df_master_clean.rating_numerator.astype("float64")
df_master_clean.rating_denominator = df_master_clean.rating_denominator.astype("float64")
#impute the data
df_master_clean.loc[df_num_clean_index, "rating_numerator"] = df_num_clean_values
df_master_clean.loc[df_num_clean_index].rating_numerator
```




    45      13.50
    695      9.75
    763     11.27
    1712    11.26
    Name: rating_numerator, dtype: float64



We also have the problem, that there can be multiple occurrences of the pattern. The reason for this is - most of the time - the display of two or more dogs in an image. For this cases we could add the ratings up, because the author of the Twitter account did this in one case that we found. Or we could build the average rating per each picture. For now, we are going to drop them out of the dataframe.


```python
pattern = "(\d+(\.\d+)?\/\d+(\.\d+)?)"

print(df_master_clean.text.str.count(pattern)[df_master_clean.text.str.count(pattern) != 1])

#get the index of the rows which contains the pattern more than once
pattern_clean_index = df_master_clean.text.str.count(pattern)[df_master_clean.text.str.count(pattern) != 1].index
```

    766     2
    1007    2
    1068    2
    1165    2
    1202    2
    1222    2
    1359    2
    1465    2
    1508    2
    1525    2
    1538    2
    1662    2
    1795    2
    1832    2
    1897    2
    1901    2
    1970    2
    2010    3
    2064    2
    2113    2
    2177    2
    2216    2
    2263    2
    2272    2
    2306    2
    2335    2
    Name: text, dtype: int64
    


```python
df_master_clean.drop(pattern_clean_index, inplace = True)
```


```python
df_master_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1638 entries, 1 to 2355
    Data columns (total 26 columns):
     #   Column              Non-Null Count  Dtype              
    ---  ------              --------------  -----              
     0   tweet_id            1638 non-null   object             
     1   timestamp           1638 non-null   datetime64[ns, UTC]
     2   source              1638 non-null   object             
     3   text                1638 non-null   object             
     4   expanded_urls       1638 non-null   object             
     5   rating_numerator    1638 non-null   float64            
     6   rating_denominator  1638 non-null   float64            
     7   name                1638 non-null   object             
     8   doggo               1638 non-null   object             
     9   floofer             1638 non-null   object             
     10  pupper              1638 non-null   object             
     11  puppo               1638 non-null   object             
     12  retweet_count       1638 non-null   int64              
     13  favorite_count      1638 non-null   int64              
     14  display_text_range  1638 non-null   object             
     15  jpg_url             1638 non-null   object             
     16  img_num             1638 non-null   int64              
     17  p1                  1638 non-null   object             
     18  p1_conf             1638 non-null   float64            
     19  p1_dog              1638 non-null   object             
     20  p2                  1638 non-null   object             
     21  p2_conf             1638 non-null   float64            
     22  p2_dog              1638 non-null   object             
     23  p3                  1638 non-null   object             
     24  p3_conf             1638 non-null   float64            
     25  p3_dog              1638 non-null   object             
    dtypes: datetime64[ns, UTC](1), float64(5), int64(3), object(17)
    memory usage: 345.5+ KB
    


```python
#no more occurrences of the mentioned problem are left
print(df_master_clean.text.str.count(pattern)[df_master_clean.text.str.count(pattern) != 1])
```

    Series([], Name: text, dtype: int64)
    

**5. Extract the source from html code**

Right now the source column is not giving us any useful information while looking at it. Because the relevant information is always between two "> <", the information will be easy to extract.


```python
df_master_clean.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>display_text_range</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>2017-08-01 00:17:27+00:00</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Tilly. She's just checking pup on you....</td>
      <td>https://twitter.com/dog_rates/status/892177421...</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>Tilly</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>6119</td>
      <td>32575</td>
      <td>[0, 138]</td>
      <td>https://pbs.twimg.com/media/DGGmoV4XsAAUL6n.jpg</td>
      <td>1</td>
      <td>Chihuahua</td>
      <td>0.323581</td>
      <td>True</td>
      <td>Pekinese</td>
      <td>0.090647</td>
      <td>True</td>
      <td>papillon</td>
      <td>0.068957</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>2017-07-31 00:18:03+00:00</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Archie. He is a rare Norwegian Pouncin...</td>
      <td>https://twitter.com/dog_rates/status/891815181...</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>Archie</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>4054</td>
      <td>24530</td>
      <td>[0, 121]</td>
      <td>https://pbs.twimg.com/media/DGBdLU1WsAANxJ9.jpg</td>
      <td>1</td>
      <td>Chihuahua</td>
      <td>0.716012</td>
      <td>True</td>
      <td>malamute</td>
      <td>0.078253</td>
      <td>True</td>
      <td>kelpie</td>
      <td>0.031379</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
#https://stackoverflow.com/questions/3075130/what-is-the-difference-between-and-regular-expressions
df_master_clean.source = df_master_clean.source.str.extract("\>(.*?)\<", expand = True)
```


```python
df_master_clean.iloc[:,:3].head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>2017-08-01 00:17:27+00:00</td>
      <td>Twitter for iPhone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>2017-07-31 00:18:03+00:00</td>
      <td>Twitter for iPhone</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_master_clean.source.value_counts()
```




    Twitter for iPhone    1610
    Twitter Web Client      20
    TweetDeck                8
    Name: source, dtype: int64



**6. Split the text range into two separate columns**


```python
df_master_clean[["display_text_range"]].info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1638 entries, 1 to 2355
    Data columns (total 1 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   display_text_range  1638 non-null   object
    dtypes: object(1)
    memory usage: 25.6+ KB
    


```python
df_master_clean.display_text_range[1]
```




    [0, 138]



Since the display_text_range column is interpreted as list, we can simply split it by using list indexing.


```python
#get the lower text range at list index 0
df_master_clean["lower_text_range"] = df_master_clean["display_text_range"].apply(lambda x: x[0])

#get the lower text range at list index 1
df_master_clean["upper_text_range"] = df_master_clean["display_text_range"].apply(lambda x: x[1])
df_master_clean.drop("display_text_range", axis = 1, inplace = True)
```


```python
df_master_clean[["lower_text_range", "upper_text_range"]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>121</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>138</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>138</td>
    </tr>
  </tbody>
</table>
</div>



**7. Remove the "None" out of the doggo, floofer, pupper and puppo column and merge them into one column**

We want to reduce the columns into one for an easier analysis. For that we have to remove the None with "" at first to concat the columns together and afterswards with np.nan, so we could easily exclude these rows from a specific analysis.


```python
#replace "None" with "" in each column
for x in ["doggo", "floofer", "pupper", "puppo"]:
    df_master_clean[x].replace("None", "", inplace = True)

#https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-dataframe-in-pandas-python
#concat the columns together
df_master_clean['dog_class'] = df_master_clean['doggo'].map(str) + df_master_clean[
    'floofer'].map(str) + df_master_clean['pupper'].map(str) + df_master_clean['puppo'].map(str)
```


```python
df_master_clean.dog_class.value_counts()
```




                    1383
    pupper           164
    doggo             54
    puppo             21
    doggopupper        7
    floofer            7
    doggofloofer       1
    doggopuppo         1
    Name: dog_class, dtype: int64




```python
#replace the leftover "" with np.nan
df_master_clean["dog_class"].replace("", np.nan, inplace = True)
```


```python
df_master_clean.dog_class.value_counts() 
```




    pupper          164
    doggo            54
    puppo            21
    doggopupper       7
    floofer           7
    doggofloofer      1
    doggopuppo        1
    Name: dog_class, dtype: int64




```python
#count the occurrences of the pattern and show the rows with count > 1
df_master_clean.text.str.count(r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)')[
    df_master_clean.text.str.count(r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)') > 1]
```




    191     2
    531     3
    575     2
    889     2
    956     2
    1063    2
    1113    2
    1304    2
    1340    2
    1367    2
    1653    2
    1788    2
    1828    2
    1907    3
    Name: text, dtype: int64



As we can see there are cases, in which there were multiple classifications. Let's extract the classes from the text and see where the differences occur.


```python
df_master_clean["dog_class_re"] = df_master_clean.text.str.extract(
    r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)', expand = True)
```


```python
#http://queirozf.com/entries/visualization-options-for-jupyter-notebooks
#show the full text
pd.set_option('display.max_colwidth', -1)

#find the differences of the extract
df_master_clean[["text","dog_class", "dog_class_re"]].dropna(subset = ["dog_class_re"]).query("dog_class != dog_class_re")
```

    C:\Users\Vasileios Garyfallos\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      This is separate from the ipykernel package so we can avoid doing imports until
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>dog_class</th>
      <th>dog_class_re</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>191</th>
      <td>Here's a puppo participating in the #ScienceMarch. Cleverly disguising her own doggo agenda. 13/10 would keep the planet habitable for https://t.co/cMhq16isel</td>
      <td>doggopuppo</td>
      <td>puppo</td>
    </tr>
    <tr>
      <th>200</th>
      <td>At first I thought this was a shy doggo, but it's actually a Rare Canadian Floofer Owl. Amateurs would confuse the two. 11/10 only send dogs https://t.co/TXdT3tmuYk</td>
      <td>doggofloofer</td>
      <td>doggo</td>
    </tr>
    <tr>
      <th>531</th>
      <td>Here we have Burke (pupper) and Dexter (doggo). Pupper wants to be exactly like doggo. Both 12/10 would pet at same time https://t.co/ANBpEYHaho</td>
      <td>doggopupper</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>575</th>
      <td>This is Bones. He's being haunted by another doggo of roughly the same size. 12/10 deep breaths pupper everything's fine https://t.co/55Dqe0SJNj</td>
      <td>doggopupper</td>
      <td>doggo</td>
    </tr>
    <tr>
      <th>889</th>
      <td>Meet Maggie &amp;amp; Lila. Maggie is the doggo, Lila is the pupper. They are sisters. Both 12/10 would pet at the same time https://t.co/MYwR4DQKll</td>
      <td>doggopupper</td>
      <td>doggo</td>
    </tr>
    <tr>
      <th>956</th>
      <td>Please stop sending it pictures that don't even have a doggo or pupper in them. Churlish af. 5/10 neat couch tho https://t.co/u2c9c7qSg8</td>
      <td>doggopupper</td>
      <td>doggo</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>This is just downright precious af. 12/10 for both pupper and doggo https://t.co/o5J479bZUC</td>
      <td>doggopupper</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>1113</th>
      <td>Like father (doggo), like son (pupper). Both 12/10 https://t.co/pG2inLaOda</td>
      <td>doggopupper</td>
      <td>doggo</td>
    </tr>
  </tbody>
</table>
</div>



The difference occurs in 8 cases. We can read through the text and extract the correct dog_class. Afterwards we can impute the correct classes into the column - for cases, in which there are multiple dogs in it, we will impute np.nan for consistency.


```python
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
```


```python
#find the differences of the extract - worked
df_master_clean[["text","dog_class", "dog_class_re"]].dropna(subset = ["dog_class_re"]).query("dog_class != dog_class_re")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>dog_class</th>
      <th>dog_class_re</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>200</th>
      <td>At first I thought this was a shy doggo, but it's actually a Rare Canadian Floofer Owl. Amateurs would confuse the two. 11/10 only send dogs https://t.co/TXdT3tmuYk</td>
      <td>floofer</td>
      <td>doggo</td>
    </tr>
    <tr>
      <th>531</th>
      <td>Here we have Burke (pupper) and Dexter (doggo). Pupper wants to be exactly like doggo. Both 12/10 would pet at same time https://t.co/ANBpEYHaho</td>
      <td>NaN</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>575</th>
      <td>This is Bones. He's being haunted by another doggo of roughly the same size. 12/10 deep breaths pupper everything's fine https://t.co/55Dqe0SJNj</td>
      <td>pupper</td>
      <td>doggo</td>
    </tr>
    <tr>
      <th>889</th>
      <td>Meet Maggie &amp;amp; Lila. Maggie is the doggo, Lila is the pupper. They are sisters. Both 12/10 would pet at the same time https://t.co/MYwR4DQKll</td>
      <td>NaN</td>
      <td>doggo</td>
    </tr>
    <tr>
      <th>956</th>
      <td>Please stop sending it pictures that don't even have a doggo or pupper in them. Churlish af. 5/10 neat couch tho https://t.co/u2c9c7qSg8</td>
      <td>NaN</td>
      <td>doggo</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>This is just downright precious af. 12/10 for both pupper and doggo https://t.co/o5J479bZUC</td>
      <td>NaN</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>1113</th>
      <td>Like father (doggo), like son (pupper). Both 12/10 https://t.co/pG2inLaOda</td>
      <td>NaN</td>
      <td>doggo</td>
    </tr>
  </tbody>
</table>
</div>




```python
#drop the columns out
df_master_clean.drop(["doggo", "floofer", "pupper", "puppo", "dog_class_re"], inplace = True, axis = 1)
```

Similar to the multiple occurence of a pattern for the numerator, we should do the same check here.


```python
#count the occurrences of the pattern and show the rows with count > 1
df_master_clean[["text", "dog_class"]].loc[
    df_master_clean.text.str.count(r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)')[
        df_master_clean.text.str.count(r'(\bpuppo\b|\bdoggo\b|\bfloofer\b|\bpupper\b)') > 1].index]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>dog_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>191</th>
      <td>Here's a puppo participating in the #ScienceMarch. Cleverly disguising her own doggo agenda. 13/10 would keep the planet habitable for https://t.co/cMhq16isel</td>
      <td>puppo</td>
    </tr>
    <tr>
      <th>531</th>
      <td>Here we have Burke (pupper) and Dexter (doggo). Pupper wants to be exactly like doggo. Both 12/10 would pet at same time https://t.co/ANBpEYHaho</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>575</th>
      <td>This is Bones. He's being haunted by another doggo of roughly the same size. 12/10 deep breaths pupper everything's fine https://t.co/55Dqe0SJNj</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>889</th>
      <td>Meet Maggie &amp;amp; Lila. Maggie is the doggo, Lila is the pupper. They are sisters. Both 12/10 would pet at the same time https://t.co/MYwR4DQKll</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>956</th>
      <td>Please stop sending it pictures that don't even have a doggo or pupper in them. Churlish af. 5/10 neat couch tho https://t.co/u2c9c7qSg8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>This is just downright precious af. 12/10 for both pupper and doggo https://t.co/o5J479bZUC</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1113</th>
      <td>Like father (doggo), like son (pupper). Both 12/10 https://t.co/pG2inLaOda</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>"I shall trip the big pupper with leash. Big pupper will never see it coming. I am a genius." Both 11/10 https://t.co/uQsCJ8pf51</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>Here is a heartbreaking scene of an incredible pupper being laid to rest. 10/10 RIP pupper https://t.co/81mvJ0rGRu</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>1367</th>
      <td>This is Sansa. She's gotten too big for her chair. Not so smol anymore. 11/10 once a pupper, always a pupper https://t.co/IpAoztle2s</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>1653</th>
      <td>"Hello forest pupper I am house pupper welcome to my abode" (8/10 for both) https://t.co/qFD8217fUT</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>1788</th>
      <td>Reckless pupper here. Not even looking at road. Absolute menace. No regard for fellow pupper lives. 10/10 still cute https://t.co/96IBkOYB7j</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>1828</th>
      <td>All this pupper wanted to do was go skiing. No one told him about the El Niño. Poor pupper. 10/10 maybe next year https://t.co/fTgbq1UBR9</td>
      <td>pupper</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>This pupper just wants a belly rub. This pupper has nothing to do w the tree being sideways now. 10/10 good pupper https://t.co/AyJ7Ohk71f</td>
      <td>pupper</td>
    </tr>
  </tbody>
</table>
</div>



All of these look correct.

**8. Remove the wrong names of name column**

Here we will also replace the wrong names with np.nan.


```python
for x in ["None", "a", "by", "the"]:
    df_master_clean["name"].replace(x, np.nan, inplace = True)
```


```python
df_master_clean.name.value_counts()
```




    Lucy        10
    Cooper      10
    Tucker      9 
    Charlie     9 
    Oliver      9 
               .. 
    Einstein    1 
    Atticus     1 
    Crumpet     1 
    Blanket     1 
    Harlso      1 
    Name: name, Length: 835, dtype: int64




```python
df_master_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
      <th>dog_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>2017-08-01 00:17:27+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Tilly. She's just checking pup on you. Hopes you're doing ok. If not, she's available for pats, snugs, boops, the whole bit. 13/10 https://t.co/0Xxu71qeIV</td>
      <td>https://twitter.com/dog_rates/status/892177421306343426/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>Tilly</td>
      <td>6119</td>
      <td>32575</td>
      <td>https://pbs.twimg.com/media/DGGmoV4XsAAUL6n.jpg</td>
      <td>1</td>
      <td>Chihuahua</td>
      <td>0.323581</td>
      <td>True</td>
      <td>Pekinese</td>
      <td>0.090647</td>
      <td>True</td>
      <td>papillon</td>
      <td>0.068957</td>
      <td>True</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>2017-07-31 00:18:03+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Archie. He is a rare Norwegian Pouncing Corgo. Lives in the tall grass. You never know when one may strike. 12/10 https://t.co/wUnZnhtVJB</td>
      <td>https://twitter.com/dog_rates/status/891815181378084864/photo/1</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>Archie</td>
      <td>4054</td>
      <td>24530</td>
      <td>https://pbs.twimg.com/media/DGBdLU1WsAANxJ9.jpg</td>
      <td>1</td>
      <td>Chihuahua</td>
      <td>0.716012</td>
      <td>True</td>
      <td>malamute</td>
      <td>0.078253</td>
      <td>True</td>
      <td>kelpie</td>
      <td>0.031379</td>
      <td>True</td>
      <td>0</td>
      <td>121</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>891689557279858688</td>
      <td>2017-07-30 15:58:51+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Darla. She commenced a snooze mid meal. 13/10 happens to the best of us https://t.co/tD36da7qLQ</td>
      <td>https://twitter.com/dog_rates/status/891689557279858688/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>Darla</td>
      <td>8424</td>
      <td>41274</td>
      <td>https://pbs.twimg.com/media/DF_q7IAWsAEuuN8.jpg</td>
      <td>1</td>
      <td>paper_towel</td>
      <td>0.170278</td>
      <td>False</td>
      <td>Labrador_retriever</td>
      <td>0.168086</td>
      <td>True</td>
      <td>spatula</td>
      <td>0.040836</td>
      <td>False</td>
      <td>0</td>
      <td>79</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>891327558926688256</td>
      <td>2017-07-29 16:00:24+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Franklin. He would like you to stop calling him "cute." He is a very fierce shark and should be respected as such. 12/10 #BarkWeek https://t.co/AtUZn91f7f</td>
      <td>https://twitter.com/dog_rates/status/891327558926688256/photo/1,https://twitter.com/dog_rates/status/891327558926688256/photo/1</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>Franklin</td>
      <td>9128</td>
      <td>39464</td>
      <td>https://pbs.twimg.com/media/DF6hr6BUMAAzZgT.jpg</td>
      <td>2</td>
      <td>basset</td>
      <td>0.555712</td>
      <td>True</td>
      <td>English_springer</td>
      <td>0.225770</td>
      <td>True</td>
      <td>German_short-haired_pointer</td>
      <td>0.175219</td>
      <td>True</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>891087950875897856</td>
      <td>2017-07-29 00:08:17+00:00</td>
      <td>Twitter for iPhone</td>
      <td>Here we have a majestic great white breaching off South Africa's coast. Absolutely h*ckin breathtaking. 13/10 (IG: tucker_marlo) #BarkWeek https://t.co/kQ04fDDRmh</td>
      <td>https://twitter.com/dog_rates/status/891087950875897856/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>3038</td>
      <td>19834</td>
      <td>https://pbs.twimg.com/media/DF3HwyEWsAABqE6.jpg</td>
      <td>1</td>
      <td>Chesapeake_Bay_retriever</td>
      <td>0.425595</td>
      <td>True</td>
      <td>Irish_terrier</td>
      <td>0.116317</td>
      <td>True</td>
      <td>Indian_elephant</td>
      <td>0.076902</td>
      <td>False</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)
```


![png](output_183_0.png)



```python
df_master_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1638 entries, 1 to 2355
    Data columns (total 24 columns):
     #   Column              Non-Null Count  Dtype              
    ---  ------              --------------  -----              
     0   tweet_id            1638 non-null   object             
     1   timestamp           1638 non-null   datetime64[ns, UTC]
     2   source              1638 non-null   object             
     3   text                1638 non-null   object             
     4   expanded_urls       1638 non-null   object             
     5   rating_numerator    1638 non-null   float64            
     6   rating_denominator  1638 non-null   float64            
     7   name                1195 non-null   object             
     8   retweet_count       1638 non-null   int64              
     9   favorite_count      1638 non-null   int64              
     10  jpg_url             1638 non-null   object             
     11  img_num             1638 non-null   int64              
     12  p1                  1638 non-null   object             
     13  p1_conf             1638 non-null   float64            
     14  p1_dog              1638 non-null   object             
     15  p2                  1638 non-null   object             
     16  p2_conf             1638 non-null   float64            
     17  p2_dog              1638 non-null   object             
     18  p3                  1638 non-null   object             
     19  p3_conf             1638 non-null   float64            
     20  p3_dog              1638 non-null   object             
     21  lower_text_range    1638 non-null   int64              
     22  upper_text_range    1638 non-null   int64              
     23  dog_class           250 non-null    object             
    dtypes: datetime64[ns, UTC](1), float64(5), int64(5), object(13)
    memory usage: 399.9+ KB
    

**9. Reduce the prediction columns into two - breed and conf**

In the next step we want to reduce the prediction columns into two - breed and confidence. The columns are already sorted by confidence. We will take the most likely prediction for each row which is supposed to be a dog.


```python
df_master_clean.query("p2_conf > p1_conf")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
      <th>dog_class</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df_master_clean.query("p3_conf > p1_conf")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
      <th>dog_class</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df_master_clean.query("p3_conf > p2_conf")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
      <th>dog_class</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



The order is correct.


```python
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
```


```python
df_master_clean.iloc[:, 12:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
      <th>dog_class</th>
      <th>breed</th>
      <th>conf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Chihuahua</td>
      <td>0.323581</td>
      <td>True</td>
      <td>Pekinese</td>
      <td>0.090647</td>
      <td>True</td>
      <td>papillon</td>
      <td>0.068957</td>
      <td>True</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>Chihuahua</td>
      <td>0.323581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chihuahua</td>
      <td>0.716012</td>
      <td>True</td>
      <td>malamute</td>
      <td>0.078253</td>
      <td>True</td>
      <td>kelpie</td>
      <td>0.031379</td>
      <td>True</td>
      <td>0</td>
      <td>121</td>
      <td>NaN</td>
      <td>Chihuahua</td>
      <td>0.716012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>paper_towel</td>
      <td>0.170278</td>
      <td>False</td>
      <td>Labrador_retriever</td>
      <td>0.168086</td>
      <td>True</td>
      <td>spatula</td>
      <td>0.040836</td>
      <td>False</td>
      <td>0</td>
      <td>79</td>
      <td>NaN</td>
      <td>Labrador_retriever</td>
      <td>0.168086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>basset</td>
      <td>0.555712</td>
      <td>True</td>
      <td>English_springer</td>
      <td>0.225770</td>
      <td>True</td>
      <td>German_short-haired_pointer</td>
      <td>0.175219</td>
      <td>True</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>basset</td>
      <td>0.555712</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chesapeake_Bay_retriever</td>
      <td>0.425595</td>
      <td>True</td>
      <td>Irish_terrier</td>
      <td>0.116317</td>
      <td>True</td>
      <td>Indian_elephant</td>
      <td>0.076902</td>
      <td>False</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>Chesapeake_Bay_retriever</td>
      <td>0.425595</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2351</th>
      <td>miniature_pinscher</td>
      <td>0.560311</td>
      <td>True</td>
      <td>Rottweiler</td>
      <td>0.243682</td>
      <td>True</td>
      <td>Doberman</td>
      <td>0.154629</td>
      <td>True</td>
      <td>0</td>
      <td>120</td>
      <td>NaN</td>
      <td>miniature_pinscher</td>
      <td>0.560311</td>
    </tr>
    <tr>
      <th>2352</th>
      <td>Rhodesian_ridgeback</td>
      <td>0.408143</td>
      <td>True</td>
      <td>redbone</td>
      <td>0.360687</td>
      <td>True</td>
      <td>miniature_pinscher</td>
      <td>0.222752</td>
      <td>True</td>
      <td>0</td>
      <td>137</td>
      <td>NaN</td>
      <td>Rhodesian_ridgeback</td>
      <td>0.408143</td>
    </tr>
    <tr>
      <th>2353</th>
      <td>German_shepherd</td>
      <td>0.596461</td>
      <td>True</td>
      <td>malinois</td>
      <td>0.138584</td>
      <td>True</td>
      <td>bloodhound</td>
      <td>0.116197</td>
      <td>True</td>
      <td>0</td>
      <td>130</td>
      <td>NaN</td>
      <td>German_shepherd</td>
      <td>0.596461</td>
    </tr>
    <tr>
      <th>2354</th>
      <td>redbone</td>
      <td>0.506826</td>
      <td>True</td>
      <td>miniature_pinscher</td>
      <td>0.074192</td>
      <td>True</td>
      <td>Rhodesian_ridgeback</td>
      <td>0.072010</td>
      <td>True</td>
      <td>0</td>
      <td>139</td>
      <td>NaN</td>
      <td>redbone</td>
      <td>0.506826</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>Welsh_springer_spaniel</td>
      <td>0.465074</td>
      <td>True</td>
      <td>collie</td>
      <td>0.156665</td>
      <td>True</td>
      <td>Shetland_sheepdog</td>
      <td>0.061428</td>
      <td>True</td>
      <td>0</td>
      <td>131</td>
      <td>NaN</td>
      <td>Welsh_springer_spaniel</td>
      <td>0.465074</td>
    </tr>
  </tbody>
</table>
<p>1638 rows × 14 columns</p>
</div>




```python
#drop the reduced columns
df_master_clean.drop(df_master_clean.columns[12:21], inplace = True, axis = 1)
```


```python
df_master_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
      <th>dog_class</th>
      <th>breed</th>
      <th>conf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>2017-08-01 00:17:27+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Tilly. She's just checking pup on you. Hopes you're doing ok. If not, she's available for pats, snugs, boops, the whole bit. 13/10 https://t.co/0Xxu71qeIV</td>
      <td>https://twitter.com/dog_rates/status/892177421306343426/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>Tilly</td>
      <td>6119</td>
      <td>32575</td>
      <td>https://pbs.twimg.com/media/DGGmoV4XsAAUL6n.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>Chihuahua</td>
      <td>0.323581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>2017-07-31 00:18:03+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Archie. He is a rare Norwegian Pouncing Corgo. Lives in the tall grass. You never know when one may strike. 12/10 https://t.co/wUnZnhtVJB</td>
      <td>https://twitter.com/dog_rates/status/891815181378084864/photo/1</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>Archie</td>
      <td>4054</td>
      <td>24530</td>
      <td>https://pbs.twimg.com/media/DGBdLU1WsAANxJ9.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>121</td>
      <td>NaN</td>
      <td>Chihuahua</td>
      <td>0.716012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>891689557279858688</td>
      <td>2017-07-30 15:58:51+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Darla. She commenced a snooze mid meal. 13/10 happens to the best of us https://t.co/tD36da7qLQ</td>
      <td>https://twitter.com/dog_rates/status/891689557279858688/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>Darla</td>
      <td>8424</td>
      <td>41274</td>
      <td>https://pbs.twimg.com/media/DF_q7IAWsAEuuN8.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>79</td>
      <td>NaN</td>
      <td>Labrador_retriever</td>
      <td>0.168086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>891327558926688256</td>
      <td>2017-07-29 16:00:24+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Franklin. He would like you to stop calling him "cute." He is a very fierce shark and should be respected as such. 12/10 #BarkWeek https://t.co/AtUZn91f7f</td>
      <td>https://twitter.com/dog_rates/status/891327558926688256/photo/1,https://twitter.com/dog_rates/status/891327558926688256/photo/1</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>Franklin</td>
      <td>9128</td>
      <td>39464</td>
      <td>https://pbs.twimg.com/media/DF6hr6BUMAAzZgT.jpg</td>
      <td>2</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>basset</td>
      <td>0.555712</td>
    </tr>
    <tr>
      <th>5</th>
      <td>891087950875897856</td>
      <td>2017-07-29 00:08:17+00:00</td>
      <td>Twitter for iPhone</td>
      <td>Here we have a majestic great white breaching off South Africa's coast. Absolutely h*ckin breathtaking. 13/10 (IG: tucker_marlo) #BarkWeek https://t.co/kQ04fDDRmh</td>
      <td>https://twitter.com/dog_rates/status/891087950875897856/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>3038</td>
      <td>19834</td>
      <td>https://pbs.twimg.com/media/DF3HwyEWsAABqE6.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>Chesapeake_Bay_retriever</td>
      <td>0.425595</td>
    </tr>
  </tbody>
</table>
</div>



**10. Clean the new breed column by replacing the "_" with a whitespace and make them all lowercase**

Now that we have our reduced column, we have to clean it for consistency.


```python
#replace "_" with " "
df_master_clean.breed = df_master_clean.breed.str.replace("_", " ")
```


```python
df_master_clean.breed
```




    1       Chihuahua               
    2       Chihuahua               
    3       Labrador retriever      
    4       basset                  
    5       Chesapeake Bay retriever
                      ...           
    2351    miniature pinscher      
    2352    Rhodesian ridgeback     
    2353    German shepherd         
    2354    redbone                 
    2355    Welsh springer spaniel  
    Name: breed, Length: 1638, dtype: object




```python
#https://stackoverflow.com/questions/22245171/how-to-lowercase-a-python-dataframe-string-column-if-it-has-missing-values
#lower the strings
df_master_clean.breed = df_master_clean.breed.str.lower()
```


```python
df_master_clean.breed.value_counts().head(10)
```




    golden retriever      154
    labrador retriever    105
    pembroke              93 
    chihuahua             87 
    pug                   62 
    toy poodle            50 
    chow                  48 
    samoyed               41 
    pomeranian            39 
    malamute              33 
    Name: breed, dtype: int64




```python
#reset index to match with the real amount of rows
df_master_clean.reset_index(drop = True, inplace = True)
```


```python
fig, ax = plt.subplots(figsize = (20,7))
ax = sns.heatmap(df_master_clean.isnull(), vmin = 0, vmax = 1)
```


![png](output_202_0.png)



```python
df_master_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1638 entries, 0 to 1637
    Data columns (total 17 columns):
     #   Column              Non-Null Count  Dtype              
    ---  ------              --------------  -----              
     0   tweet_id            1638 non-null   object             
     1   timestamp           1638 non-null   datetime64[ns, UTC]
     2   source              1638 non-null   object             
     3   text                1638 non-null   object             
     4   expanded_urls       1638 non-null   object             
     5   rating_numerator    1638 non-null   float64            
     6   rating_denominator  1638 non-null   float64            
     7   name                1195 non-null   object             
     8   retweet_count       1638 non-null   int64              
     9   favorite_count      1638 non-null   int64              
     10  jpg_url             1638 non-null   object             
     11  img_num             1638 non-null   int64              
     12  lower_text_range    1638 non-null   int64              
     13  upper_text_range    1638 non-null   int64              
     14  dog_class           250 non-null    object             
     15  breed               1638 non-null   object             
     16  conf                1638 non-null   float64            
    dtypes: datetime64[ns, UTC](1), float64(3), int64(5), object(8)
    memory usage: 217.7+ KB
    


```python
df_master_clean[["breed", "conf"]].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>breed</th>
      <th>conf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chihuahua</td>
      <td>0.323581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chihuahua</td>
      <td>0.716012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>labrador retriever</td>
      <td>0.168086</td>
    </tr>
  </tbody>
</table>
</div>




```python
#save the data to a *.csv file
df_master_clean.to_csv('twitter_archive_master.csv', index = False)
```

<a id='analysis'></a>
## Data Analysis

**Questions:**

<ol>
    <li>Based on the predicted, most likely dog breed: Which breed gets retweeted and favorited the most overall?</li>
    <li>How did the account develop (speaking about number of tweets, retweets, favorites, image number and length of the tweets)?</li> 
    <li>Is there a pattern visible in the timing of the tweets?</li> 
</ol>
    

**1. Based on the predicted, most likely dog breed: Which breed gets retweeted and favorited the most overall?**

To answer this question we will first take a look on the frequency of the breed occurence and afterwards we will create a groupby object to sum up the favorite and retweet count of each breed in this dataset.


```python
#read in the master csv
df = pd.read_csv("twitter_archive_master.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
      <th>dog_class</th>
      <th>breed</th>
      <th>conf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892177421306343426</td>
      <td>2017-08-01 00:17:27+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Tilly. She's just checking pup on you. Hopes you're doing ok. If not, she's available for pats, snugs, boops, the whole bit. 13/10 https://t.co/0Xxu71qeIV</td>
      <td>https://twitter.com/dog_rates/status/892177421306343426/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>Tilly</td>
      <td>6119</td>
      <td>32575</td>
      <td>https://pbs.twimg.com/media/DGGmoV4XsAAUL6n.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>chihuahua</td>
      <td>0.323581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>891815181378084864</td>
      <td>2017-07-31 00:18:03+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Archie. He is a rare Norwegian Pouncing Corgo. Lives in the tall grass. You never know when one may strike. 12/10 https://t.co/wUnZnhtVJB</td>
      <td>https://twitter.com/dog_rates/status/891815181378084864/photo/1</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>Archie</td>
      <td>4054</td>
      <td>24530</td>
      <td>https://pbs.twimg.com/media/DGBdLU1WsAANxJ9.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>121</td>
      <td>NaN</td>
      <td>chihuahua</td>
      <td>0.716012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891689557279858688</td>
      <td>2017-07-30 15:58:51+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Darla. She commenced a snooze mid meal. 13/10 happens to the best of us https://t.co/tD36da7qLQ</td>
      <td>https://twitter.com/dog_rates/status/891689557279858688/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>Darla</td>
      <td>8424</td>
      <td>41274</td>
      <td>https://pbs.twimg.com/media/DF_q7IAWsAEuuN8.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>79</td>
      <td>NaN</td>
      <td>labrador retriever</td>
      <td>0.168086</td>
    </tr>
    <tr>
      <th>3</th>
      <td>891327558926688256</td>
      <td>2017-07-29 16:00:24+00:00</td>
      <td>Twitter for iPhone</td>
      <td>This is Franklin. He would like you to stop calling him "cute." He is a very fierce shark and should be respected as such. 12/10 #BarkWeek https://t.co/AtUZn91f7f</td>
      <td>https://twitter.com/dog_rates/status/891327558926688256/photo/1,https://twitter.com/dog_rates/status/891327558926688256/photo/1</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>Franklin</td>
      <td>9128</td>
      <td>39464</td>
      <td>https://pbs.twimg.com/media/DF6hr6BUMAAzZgT.jpg</td>
      <td>2</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>basset</td>
      <td>0.555712</td>
    </tr>
    <tr>
      <th>4</th>
      <td>891087950875897856</td>
      <td>2017-07-29 00:08:17+00:00</td>
      <td>Twitter for iPhone</td>
      <td>Here we have a majestic great white breaching off South Africa's coast. Absolutely h*ckin breathtaking. 13/10 (IG: tucker_marlo) #BarkWeek https://t.co/kQ04fDDRmh</td>
      <td>https://twitter.com/dog_rates/status/891087950875897856/photo/1</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>3038</td>
      <td>19834</td>
      <td>https://pbs.twimg.com/media/DF3HwyEWsAABqE6.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>138</td>
      <td>NaN</td>
      <td>chesapeake bay retriever</td>
      <td>0.425595</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['tweet_id', 'timestamp', 'source', 'text', 'expanded_urls',
           'rating_numerator', 'rating_denominator', 'name', 'retweet_count',
           'favorite_count', 'jpg_url', 'img_num', 'lower_text_range',
           'upper_text_range', 'dog_class', 'breed', 'conf'],
          dtype='object')




```python
#https://stackoverflow.com/questions/32891211/limit-the-number-of-groups-shown-in-seaborn-countplot
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "breed", data = df, order=df.breed.value_counts().iloc[:10].index, palette = "viridis")
ax.set_title("count of classified breeds in the dataset");

ax.set_ylim(0, 170)
#https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+2))

```


![png](output_213_0.png)


The dogs displayed in the images are mostly golden retrievers with a count of 154 or labrador retrievers with a count of 105.


```python
df_breed_group = df[["retweet_count", "favorite_count", "breed"]].groupby("breed", as_index = False).sum()
```


```python
df_breed_group.sort_values("retweet_count", ascending = False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>breed</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>golden retriever</td>
      <td>540306</td>
      <td>1843530</td>
    </tr>
    <tr>
      <th>62</th>
      <td>labrador retriever</td>
      <td>388340</td>
      <td>1246987</td>
    </tr>
    <tr>
      <th>80</th>
      <td>pembroke</td>
      <td>276036</td>
      <td>1011313</td>
    </tr>
    <tr>
      <th>27</th>
      <td>chihuahua</td>
      <td>227718</td>
      <td>710271</td>
    </tr>
    <tr>
      <th>88</th>
      <td>samoyed</td>
      <td>182659</td>
      <td>540911</td>
    </tr>
    <tr>
      <th>41</th>
      <td>french bulldog</td>
      <td>154310</td>
      <td>587398</td>
    </tr>
    <tr>
      <th>30</th>
      <td>cocker spaniel</td>
      <td>136214</td>
      <td>385193</td>
    </tr>
    <tr>
      <th>28</th>
      <td>chow</td>
      <td>125706</td>
      <td>438103</td>
    </tr>
    <tr>
      <th>82</th>
      <td>pug</td>
      <td>111130</td>
      <td>366684</td>
    </tr>
    <tr>
      <th>103</th>
      <td>toy poodle</td>
      <td>108459</td>
      <td>322009</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_breed_group.sort_values("favorite_count", ascending = False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>breed</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>golden retriever</td>
      <td>540306</td>
      <td>1843530</td>
    </tr>
    <tr>
      <th>62</th>
      <td>labrador retriever</td>
      <td>388340</td>
      <td>1246987</td>
    </tr>
    <tr>
      <th>80</th>
      <td>pembroke</td>
      <td>276036</td>
      <td>1011313</td>
    </tr>
    <tr>
      <th>27</th>
      <td>chihuahua</td>
      <td>227718</td>
      <td>710271</td>
    </tr>
    <tr>
      <th>41</th>
      <td>french bulldog</td>
      <td>154310</td>
      <td>587398</td>
    </tr>
    <tr>
      <th>88</th>
      <td>samoyed</td>
      <td>182659</td>
      <td>540911</td>
    </tr>
    <tr>
      <th>28</th>
      <td>chow</td>
      <td>125706</td>
      <td>438103</td>
    </tr>
    <tr>
      <th>30</th>
      <td>cocker spaniel</td>
      <td>136214</td>
      <td>385193</td>
    </tr>
    <tr>
      <th>82</th>
      <td>pug</td>
      <td>111130</td>
      <td>366684</td>
    </tr>
    <tr>
      <th>66</th>
      <td>malamute</td>
      <td>103625</td>
      <td>342690</td>
    </tr>
  </tbody>
</table>
</div>



The golden retriever and the labrador retriever therefore also lead the list of most favorite and retweets.


```python
df_breed_group["sum"] = df_breed_group["retweet_count"] + df_breed_group["favorite_count"]
```


```python
df_breed_group.sort_values("sum", ascending = False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>breed</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>golden retriever</td>
      <td>540306</td>
      <td>1843530</td>
      <td>2383836</td>
    </tr>
    <tr>
      <th>62</th>
      <td>labrador retriever</td>
      <td>388340</td>
      <td>1246987</td>
      <td>1635327</td>
    </tr>
    <tr>
      <th>80</th>
      <td>pembroke</td>
      <td>276036</td>
      <td>1011313</td>
      <td>1287349</td>
    </tr>
    <tr>
      <th>27</th>
      <td>chihuahua</td>
      <td>227718</td>
      <td>710271</td>
      <td>937989</td>
    </tr>
    <tr>
      <th>41</th>
      <td>french bulldog</td>
      <td>154310</td>
      <td>587398</td>
      <td>741708</td>
    </tr>
    <tr>
      <th>88</th>
      <td>samoyed</td>
      <td>182659</td>
      <td>540911</td>
      <td>723570</td>
    </tr>
    <tr>
      <th>28</th>
      <td>chow</td>
      <td>125706</td>
      <td>438103</td>
      <td>563809</td>
    </tr>
    <tr>
      <th>30</th>
      <td>cocker spaniel</td>
      <td>136214</td>
      <td>385193</td>
      <td>521407</td>
    </tr>
    <tr>
      <th>82</th>
      <td>pug</td>
      <td>111130</td>
      <td>366684</td>
      <td>477814</td>
    </tr>
    <tr>
      <th>66</th>
      <td>malamute</td>
      <td>103625</td>
      <td>342690</td>
      <td>446315</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "breed", y = "sum", data = df_breed_group.sort_values("sum", ascending=False).iloc[:10], palette = "viridis")
ax.set_title("sum of favorites and retweets per breed");
```


![png](output_221_0.png)


Now let's look at the most retweetet and favorited single tweet.


```python
df[["retweet_count", "favorite_count", "breed"]].sort_values("retweet_count", ascending = False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>breed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>683</th>
      <td>83310</td>
      <td>163821</td>
      <td>labrador retriever</td>
    </tr>
    <tr>
      <th>710</th>
      <td>61680</td>
      <td>121016</td>
      <td>eskimo dog</td>
    </tr>
    <tr>
      <th>360</th>
      <td>60760</td>
      <td>126645</td>
      <td>chihuahua</td>
    </tr>
    <tr>
      <th>275</th>
      <td>47503</td>
      <td>139998</td>
      <td>lakeland terrier</td>
    </tr>
    <tr>
      <th>53</th>
      <td>43282</td>
      <td>103726</td>
      <td>english springer</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[["retweet_count", "favorite_count", "breed"]].sort_values("favorite_count", ascending = False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>breed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>683</th>
      <td>83310</td>
      <td>163821</td>
      <td>labrador retriever</td>
    </tr>
    <tr>
      <th>275</th>
      <td>47503</td>
      <td>139998</td>
      <td>lakeland terrier</td>
    </tr>
    <tr>
      <th>360</th>
      <td>60760</td>
      <td>126645</td>
      <td>chihuahua</td>
    </tr>
    <tr>
      <th>100</th>
      <td>35307</td>
      <td>121592</td>
      <td>french bulldog</td>
    </tr>
    <tr>
      <th>710</th>
      <td>61680</td>
      <td>121016</td>
      <td>eskimo dog</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the most liked and retweeted tweet is in fact a labrador retriever, with golden retrievers not even being in the list. Let's see if there are big differences in the average rating.


```python
df_breed_group_mean = df[["rating_numerator", "breed"]].groupby("breed", as_index = False).mean()
```


```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "breed", y = "rating_numerator", data = df_breed_group_mean.sort_values("rating_numerator", ascending = False).iloc[:100], palette = "viridis")
ax.set_title("average rating_numerator per breed");
#https://code.i-harness.com/de/q/2135a8
ax.xaxis.set_ticklabels([])
plt.tight_layout()
```


![png](output_227_0.png)


While the most breeds are on average on nearly the same level of rating, there is some outlier visible.


```python
df_breed_group_mean.sort_values("rating_numerator", ascending = False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>breed</th>
      <th>rating_numerator</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>96</th>
      <td>soft-coated wheaten terrier</td>
      <td>21.357143</td>
    </tr>
    <tr>
      <th>109</th>
      <td>west highland white terrier</td>
      <td>14.687500</td>
    </tr>
    <tr>
      <th>48</th>
      <td>great pyrenees</td>
      <td>14.666667</td>
    </tr>
    <tr>
      <th>16</th>
      <td>borzoi</td>
      <td>14.333333</td>
    </tr>
    <tr>
      <th>28</th>
      <td>chow</td>
      <td>14.166667</td>
    </tr>
    <tr>
      <th>62</th>
      <td>labrador retriever</td>
      <td>13.714286</td>
    </tr>
    <tr>
      <th>4</th>
      <td>australian terrier</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>bouvier des flandres</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>87</th>
      <td>saluki</td>
      <td>12.500000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>briard</td>
      <td>12.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[["breed", "rating_numerator"]].sort_values("rating_numerator", ascending = False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>breed</th>
      <th>rating_numerator</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>588</th>
      <td>labrador retriever</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>1227</th>
      <td>chow</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>1121</th>
      <td>soft-coated wheaten terrier</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>828</th>
      <td>golden retriever</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>1274</th>
      <td>labrador retriever</td>
      <td>88.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df.query("breed == 'labrador retriever'"))
```




    105




```python
len(df.query("breed == 'soft-coated wheaten terrier'"))
```




    14




```python
df.query("breed != 'soft-coated wheaten terrier'").rating_numerator.mean()
```




    11.26649014778325



The soft-coated wheaten terrier	got a very high mean rating. In fact, the labrador retriever got overall the biggest rating with 165, but since there are a lot more tweets with labrador retriever than for the soft-coated wheaten terrier, the one big rating of the soft-coated wheaten terrier has a higher weight then the one of the labrador retriever (14 tweets of soft-coated wheaten terrier and 105 of labrador retriever).

Not taking the outlier into account, this leads us to an average rating of 11. Based on the number of posts, retweets, favorites and mean rating, we will give the title of "Most overall liked dog of this Twitter account and its community" to the labrador retriever.

**2. How did the account develop (speaking about number of tweets, retweets, favorites, image number and length of the tweets)?**

To answer this question we first have to extract time information out of the timestamp.


```python
df_time = df.copy()
```


```python
df_time.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1638 entries, 0 to 1637
    Data columns (total 17 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   tweet_id            1638 non-null   int64  
     1   timestamp           1638 non-null   object 
     2   source              1638 non-null   object 
     3   text                1638 non-null   object 
     4   expanded_urls       1638 non-null   object 
     5   rating_numerator    1638 non-null   float64
     6   rating_denominator  1638 non-null   float64
     7   name                1195 non-null   object 
     8   retweet_count       1638 non-null   int64  
     9   favorite_count      1638 non-null   int64  
     10  jpg_url             1638 non-null   object 
     11  img_num             1638 non-null   int64  
     12  lower_text_range    1638 non-null   int64  
     13  upper_text_range    1638 non-null   int64  
     14  dog_class           250 non-null    object 
     15  breed               1638 non-null   object 
     16  conf                1638 non-null   float64
    dtypes: float64(3), int64(6), object(8)
    memory usage: 217.7+ KB
    

The timestamp is a string again, so I have to transform it again.


```python
df_time.timestamp = pd.to_datetime(df_time.timestamp)
```

Let's first take a look on the day of the week.


```python
df_time["dow"] = df_time["timestamp"].apply(lambda x: x.dayofweek)
```


```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "dow", data = df_time, palette = "viridis")
ax.set_title("count of tweets in the dataset for each day of the week");

ax.set_ylim(0, 300)
#https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+5))

```


![png](output_243_0.png)


We can see that most of the tweets are posted on monday. For tuesday to friday it is nearly the same number of posts. On the weekend the Twitter profile tweets a little bit less. 


```python
df_time.timestamp.min()
```




    Timestamp('2015-11-15 22:32:08+0000', tz='UTC')




```python
df_time.timestamp.max()
```




    Timestamp('2017-08-01 00:17:27+0000', tz='UTC')



This dataset contains data from the end of 2015 to the August of 2017. Let's extract the month, year and hour information from the timestamp.


```python
#get the month out of the timestamp
df_time["month"] = df_time["timestamp"].apply(lambda x: x.month)
#get the year out of the timestamp
df_time["year"] = df_time["timestamp"].apply(lambda x: x.year)
#get the hour out of the timestamp
df_time["hour"] = df_time["timestamp"].apply(lambda x: x.hour)
```

For the first graph I only want to take a look on the full year 2016.


```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "month", data = df_time.query("year == 2016"), palette = "viridis")
ax.set_title("count of tweets in the dataset for each month 2016");

ax.set_ylim(0,150)
#https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+1.5))

```


![png](output_250_0.png)


Over the timeperiod of 2016 the number of post per months decreased. It went from 134 tweets in January to 52 in December. Does this mean, that the performance of this account is also decreasing?


```python
#create a timestamp containing month and year
df_time['month_year'] = pd.to_datetime(df["timestamp"]).dt.to_period('M')
```

    C:\Users\Vasileios Garyfallos\Anaconda3\lib\site-packages\pandas\core\arrays\datetimes.py:1104: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
      UserWarning,
    


```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "month_year", data = df_time.sort_values("month_year"), palette = "viridis", )
ax.set_title("count of tweets in the dataset for each year - month combination in this dataset");

ax.set_ylim(0, 300)
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+3))

plt.tight_layout()
```


![png](output_253_0.png)


If we look at it over the whole timeperiod it becomes even more clear. In April 2016 the number of tweets dropped and since then it has a relatively stable level. To see if the performance of the Account decreased we will take a look on the favorites and retweets that the posts get. 


```python
df_time_groupby = df_time.groupby("month_year", as_index = False).sum()
```


```python
df_time_groupby.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month_year</th>
      <th>tweet_id</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>img_num</th>
      <th>lower_text_range</th>
      <th>upper_text_range</th>
      <th>conf</th>
      <th>dow</th>
      <th>month</th>
      <th>year</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-11</td>
      <td>1.450925e+20</td>
      <td>2049.00</td>
      <td>2170.0</td>
      <td>99409.0</td>
      <td>249184.0</td>
      <td>220.0</td>
      <td>0.0</td>
      <td>26103.0</td>
      <td>105.165214</td>
      <td>635.0</td>
      <td>2387.0</td>
      <td>437255.0</td>
      <td>1833.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-12</td>
      <td>1.866097e+20</td>
      <td>2986.26</td>
      <td>2940.0</td>
      <td>419666.0</td>
      <td>1008254.0</td>
      <td>299.0</td>
      <td>0.0</td>
      <td>31103.0</td>
      <td>143.884391</td>
      <td>783.0</td>
      <td>3312.0</td>
      <td>556140.0</td>
      <td>2514.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01</td>
      <td>9.218872e+19</td>
      <td>1505.00</td>
      <td>1440.0</td>
      <td>166875.0</td>
      <td>468049.0</td>
      <td>152.0</td>
      <td>0.0</td>
      <td>15547.0</td>
      <td>71.353070</td>
      <td>418.0</td>
      <td>134.0</td>
      <td>270144.0</td>
      <td>1082.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-02</td>
      <td>6.362919e+19</td>
      <td>1033.00</td>
      <td>980.0</td>
      <td>121656.0</td>
      <td>347210.0</td>
      <td>107.0</td>
      <td>0.0</td>
      <td>10390.0</td>
      <td>49.415261</td>
      <td>256.0</td>
      <td>182.0</td>
      <td>183456.0</td>
      <td>737.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03</td>
      <td>7.376839e+19</td>
      <td>1298.00</td>
      <td>1230.0</td>
      <td>150242.0</td>
      <td>442948.0</td>
      <td>123.0</td>
      <td>0.0</td>
      <td>12282.0</td>
      <td>62.738141</td>
      <td>310.0</td>
      <td>312.0</td>
      <td>209664.0</td>
      <td>889.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "favorite_count", data = df_time_groupby, palette = "viridis")
ax.set_title("sum of favorites per month-year combination");
plt.tight_layout()
```


![png](output_257_0.png)



```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "retweet_count", data = df_time_groupby, palette = "viridis")
ax.set_title("sum of retweets per month-year combination");
```


![png](output_258_0.png)


Interesting, while the number of tweets per month is decreasing, the favorites and retweets per month are increasing.


```python
df_time_groupby_mean = df_time.groupby("month_year", as_index = False).mean()
```


```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "favorite_count", data = df_time_groupby_mean, palette = "viridis")
ax.set_title("mean of favorites per month-year combination");
plt.tight_layout()
```


![png](output_261_0.png)



```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "retweet_count", data = df_time_groupby_mean, palette = "viridis")
ax.set_title("mean of retweets per month-year combination");
plt.tight_layout()
```


![png](output_262_0.png)


If we look at the average number of favorites and retweets the clear uptrend gets even more clearclearer! Now let's see if the number of posted images per month or the average upper text range changed over time.


```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "img_num", data = df_time_groupby_mean, palette = "viridis")
ax.set_title("mean image number per month-year combination");
plt.tight_layout()
```


![png](output_264_0.png)


For the images it seems pretty stable. There are months where are more and months where are less posted images, but overall there is no clean up- or downtrend visible.


```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.barplot(x = "month_year", y = "upper_text_range", data = df_time_groupby_mean, palette = "viridis")
ax.set_title("mean upper text range per month_year combination");
```


![png](output_266_0.png)



```python
df_time_groupby_mean.upper_text_range.mean()
```




    109.83472369338006




```python
df_time_groupby_mean.iloc[:11].upper_text_range.mean()
```




    106.13287257908605




```python
df_time_groupby_mean.iloc[11:].upper_text_range.mean()
```




    113.5365748076741



For the tweet length it seems like it increased over the second half of this dataset from an average of 106 to 113. 

**3. Is there a pattern visible in the timing of the tweets?**


```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "hour", data = df_time.query("year == 2016"), palette = "viridis")
ax.set_title("count of tweets in the dataset for each hour in 2016");
```


![png](output_272_0.png)



```python
fig, ax = plt.subplots(figsize = (16,5))
ax = sns.countplot(x = "hour", data = df_time.query("year == 2015"), palette = "viridis")
ax.set_title("count of tweets in the dataset for each hour in 2015");
```


![png](output_273_0.png)


In years 2015 & 2016, the most posts are during the night between 0:00 - 5:00. Between 4:00 - 15:00 there is a very small amounts of tweets, no tweets between 7:00 - 12:00 and at 14:00. There are a few tweets after 14:00, but not as many as between 0:00 - 5:00.  

<a id='#conclusion'></a>
## Summary and Conclusions

Questions & **Answers:**

1. Based on the predicted, most likely dog breed: Which breed gets retweeted and favorited the most overall?

**Labrador Retriever**.

2. How did the account develop (number of tweets, retweets, favorites, image number and length of the tweets)?
    
**Number of tweets per month decreased, retweets and favorites show an uptrend. No clear trend for the image numbers. Length of the tweets with increasing trend toward to the maximum limit of 130 characters.**
    
3. Is there a pattern visible in the timing of the tweets? 
    
**Between 5:00 - 15:00, there are nearly no tweets at all. The most tweets are between 0:00 - 4:00 and then between 15:00 - 23:00. Moreover, between 15:00 - 23:00 there are less tweets than between 0:00 - 4:00.**
    
    
    
