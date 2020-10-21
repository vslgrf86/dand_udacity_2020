#!/usr/bin/env python
# coding: utf-8

# # <center>Udacity Data Analyst Nanodegree </center>
# 
# # <center>Project: Investigate a Dataset - Analyze TMDb Movie Data</center>
# #### Vasileios Garyfallos, March 2020
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## <center>Introduction</center>
# 
# ### Dataset chosen for analysis: TMDb Movie Data
# 
# The dataset contains information about movies collected from The Movies Database, including revenue, budget, ratings etc.
# 
# ### Questions posed:
#   * How budgets, revenues, runtimes, costs per minute evolved over the years? What is the correlation between those metrics?
# 

# The following Python libraries were imported, in order to conduct the analysis and answer the questions:

# In[1]:


import seaborn as sns; sns.set() #plot data
import numpy as np #create arrays
import pandas as pd #handle and wrangle data
import matplotlib.pyplot as plt #plot data


get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## <center>Data Wrangling</center>
# 
# 
# 
# ### Dataset General Properties - Structure

# After loading the dataset, I displayed 5 rows to get a little more detailed view about the columns, rows and  overall structure of the dataset:

# In[2]:


# load tmdb-movies.csv file and show headers of the dataframe

df = pd.read_csv("tmdb-movies.csv")
df.head()


# Looking at the shape of the dataset:
# 
# Output = 10866 rows, 21 columns

# In[3]:


df.shape


# In order to understand the data a little bit more, I performed some basic summary statistics:

# In[4]:


df.describe()


# ## <center>Data Cleaning</center>

# I examined whether the dataset has missing values, looked for any wrong datatypes and reduntant columns:

# In[5]:


df.info()


# Output: 
# 
# The dataset has some missing values, which are non-numerical data (object) and need to be replaced before starting our analysis. Also 8 columns are irrelevant and will be dropped.

# I decided to drop 8 columns, which are irrelevant for the intented analysis. This will make the dataset more consistent and its handling easier:

# In[6]:


df = df.drop(['cast', 'homepage', 'tagline', 'keywords', 'overview', 'imdb_id', 'budget_adj', 'revenue_adj'], axis=1)


# Columns dropped:
# * cast 
# * homepage 
# * tagline
# * keywords
# * overview
# * imdb id
# * budget_adj
# * revenue_adj

# Checked results: The unnecessary 8 columns were dropped.

# In[7]:


df.info()


# I trimmed the dataset, replacing the empty values in 3 columns with the word 'nodata':

# In[8]:


df['director'] = df['director'].fillna('nodata')
df['production_companies'] = df['production_companies'].fillna('nodata')
df['genres'] = df['genres'].fillna('nodata')


# Checked results: The empty values were replaced with the word 'nodata'

# In[9]:


df.query('director == "nodata"').head(5)
df.query('production_companies == "nodata"').head(5)
df.query('genres == "nodata"').head(5)


# In[10]:


df.info()


# <a id='eda'></a>
# ## <center>Exploratory Data Analysis</center>
# 
# 
# ### Question 1: How budgets, revenues, runtimes, costs per minute evolved over the years? What is the correlation between those metrics?
# 
# 
# To answer this question for each metric, I will first group and plot the data , in order to gain insight and understand it better.

# In[11]:


# Grouping the data by release year and summing its attributes
df.groupby('release_year').sum()


# A first glimpse gives the intuition that the numbers have changed a lot over time with an upward trend. Is this the case? 
# 
# I will create charts with the two variables (revenue, budget), in order to have a view of the two trends and examine the correlation.

# In[12]:


ax = plt.subplots(figsize=(30, 10))
sns.barplot(x='release_year', y='revenue', data=df);
plt.title('Revenue generated from films per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('Revenue', fontsize=18)


# In[13]:


ax = plt.subplots(figsize=(30, 10))
sns.barplot(x='release_year', y='budget', data=df);
plt.title('Budget invested in films per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('Budget', fontsize=18)


# In[14]:


ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='release_year', y='revenue', data=df);
sns.lineplot(x='release_year', y='budget', data=df);
plt.title('Chart line of film budget and revenue per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('Revenue', fontsize=18)


# It seems that the two variables follow identical trends, especially after 1990. The higher the budget for a movie is, the higher are the revenues that generates.
# 
# I also examined the correlation coefficient R between these two variables:

# In[15]:


df['revenue'].corr(df['budget'])


# With a value of 0,74 there is a strong correlation between the two variables, which confirms my first conclusion.

# What about the runtime of the movies over the years? 
# 
# This will give helpful insight regarding whether there's a correlation between the budget of a movie and its duration:

# In[16]:


df.groupby('release_year').sum()


# In[17]:


ax = plt.subplots(figsize=(30, 10))
sns.lineplot(y="runtime", x="release_year", data=df, color='b');
plt.title('Runtime of films per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('Runtime', fontsize=18)


# In[ ]:





# The plot shows a decreasing tendency for the decade 1960 - 1970. 
# 
# From that point and until the mid 90's we can see an increase of the movies' runtimes, only to reach mid 60's levels and, since then, the tendency is steadily decreasing.
# 
# This is opposite to the tendency of the movies' budgets over the years, which have seen a steady increase since the beginning of the examined timeline.
# 
# I also examined the correlation coefficient R between these two variables:

# In[18]:


df['runtime'].corr(df['budget'])


# With a value of 0,19 there is a weak or no correlation between the two variables, which confirms my first conclusion.

# There's another one metric, which needs to be examined: The distribution (N) of the films' runtimes. This will give useful info about the standard trend of filmaking regarding the runtime attribute. I created a histogram to examine it:

# In[19]:


ax = plt.subplots(figsize=(30, 10))
sns.distplot(df['runtime'], bins=200)
plt.title('Distribution of runtime of films', fontsize=28)
plt.xlabel('Runtime in Minutes',fontsize=18)
plt.ylabel('N',fontsize=18)
plt.xlim(xmin=0, xmax = 200)


# The histogram shows that the films with a runtime of 95 minutes are the most frequent in the population.

# I also examined the tendency for another important metric: Cost per minute (cpm). To do so, I created a new, derived column, which gives the result of budget/runtime:

# In[20]:


df['cpm'] = df['budget']/df['runtime']
df


# In[21]:


ax = plt.subplots(figsize=(30, 10))

sns.barplot(x='release_year', y='cpm', data=df);
plt.title('Cost per minute of films per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('CPM', fontsize=18)


# The bar plot shows increasing tendency for the costs per minute from 1960 until 1999. From that point the CPM of the movies are steadily decreasing.

# <a id='conclusions'></a>
# ## <center>Conclusions</center>
# 

# 1. The research question "How budgets, revenues, runtimes, costs per minute evolved over the years? What is the correlation between those metrics?" has shown as results that **budgets and revenues have steadily increased over time** with a strong correlation. It seems that nowadays we have overall more costly movies that generate higher revenues.
# 
# 2. The runtime of the movies has seen a **steady decrease over the examined timeline**. In the recent decades the movies have been shorter with a tendency for a runtime of 95 minutes.
# 
# 3. The metric **Costs per minute has seen a steady increase until the year 1999** and since then the costs are decreasing. This is in accordance with the other two metrics (budget, runtime) and all three variables have identical tendencies.

# **All results are limited to the underlying dataset and since no advanced statistics were performed, the results can only be treated as indicators and are not generalizable. Furthermore, one has to consider that many entries in the dataset have been removed due to missing data**
