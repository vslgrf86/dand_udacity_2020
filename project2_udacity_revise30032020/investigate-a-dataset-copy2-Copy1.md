# <center>Udacity Data Analyst Nanodegree </center>

# <center>Project: Investigate a Dataset - Analyze TMDb Movie Data</center>
#### Vasileios Garyfallos, March 2020

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## <center>Introduction</center>

### Dataset chosen for analysis: TMDb Movie Data

The dataset contains information about movies collected from The Movies Database, including revenue, budget, ratings etc.

### Questions posed:
  * How budgets, revenues, runtimes, costs per minute evolved over the years? What is the correlation between those metrics?


The following Python libraries were imported, in order to conduct the analysis and answer the questions:


```python
import seaborn as sns; sns.set() #plot data
import numpy as np #create arrays
import pandas as pd #handle and wrangle data
import matplotlib.pyplot as plt #plot data


%matplotlib inline
```

<a id='wrangling'></a>
## <center>Data Wrangling</center>



### Dataset General Properties - Structure

After loading the dataset, I displayed 5 rows to get a little more detailed view about the columns, rows and  overall structure of the dataset:


```python
# load tmdb-movies.csv file and show headers of the dataframe

df = pd.read_csv("tmdb-movies.csv")
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
      <th>id</th>
      <th>imdb_id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>cast</th>
      <th>homepage</th>
      <th>director</th>
      <th>tagline</th>
      <th>...</th>
      <th>overview</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135397</td>
      <td>tt0369610</td>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>
      <td>http://www.jurassicworld.com/</td>
      <td>Colin Trevorrow</td>
      <td>The park is open.</td>
      <td>...</td>
      <td>Twenty-two years after the events of Jurassic ...</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>6/9/15</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76341</td>
      <td>tt1392190</td>
      <td>28.419936</td>
      <td>150000000</td>
      <td>378436354</td>
      <td>Mad Max: Fury Road</td>
      <td>Tom Hardy|Charlize Theron|Hugh Keays-Byrne|Nic...</td>
      <td>http://www.madmaxmovie.com/</td>
      <td>George Miller</td>
      <td>What a Lovely Day.</td>
      <td>...</td>
      <td>An apocalyptic story set in the furthest reach...</td>
      <td>120</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Village Roadshow Pictures|Kennedy Miller Produ...</td>
      <td>5/13/15</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>3.481613e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>262500</td>
      <td>tt2908446</td>
      <td>13.112507</td>
      <td>110000000</td>
      <td>295238201</td>
      <td>Insurgent</td>
      <td>Shailene Woodley|Theo James|Kate Winslet|Ansel...</td>
      <td>http://www.thedivergentseries.movie/#insurgent</td>
      <td>Robert Schwentke</td>
      <td>One Choice Can Destroy You</td>
      <td>...</td>
      <td>Beatrice Prior must confront her inner demons ...</td>
      <td>119</td>
      <td>Adventure|Science Fiction|Thriller</td>
      <td>Summit Entertainment|Mandeville Films|Red Wago...</td>
      <td>3/18/15</td>
      <td>2480</td>
      <td>6.3</td>
      <td>2015</td>
      <td>1.012000e+08</td>
      <td>2.716190e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140607</td>
      <td>tt2488496</td>
      <td>11.173104</td>
      <td>200000000</td>
      <td>2068178225</td>
      <td>Star Wars: The Force Awakens</td>
      <td>Harrison Ford|Mark Hamill|Carrie Fisher|Adam D...</td>
      <td>http://www.starwars.com/films/star-wars-episod...</td>
      <td>J.J. Abrams</td>
      <td>Every generation has a story.</td>
      <td>...</td>
      <td>Thirty years after defeating the Galactic Empi...</td>
      <td>136</td>
      <td>Action|Adventure|Science Fiction|Fantasy</td>
      <td>Lucasfilm|Truenorth Productions|Bad Robot</td>
      <td>12/15/15</td>
      <td>5292</td>
      <td>7.5</td>
      <td>2015</td>
      <td>1.839999e+08</td>
      <td>1.902723e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>168259</td>
      <td>tt2820852</td>
      <td>9.335014</td>
      <td>190000000</td>
      <td>1506249360</td>
      <td>Furious 7</td>
      <td>Vin Diesel|Paul Walker|Jason Statham|Michelle ...</td>
      <td>http://www.furious7.com/</td>
      <td>James Wan</td>
      <td>Vengeance Hits Home</td>
      <td>...</td>
      <td>Deckard Shaw seeks revenge against Dominic Tor...</td>
      <td>137</td>
      <td>Action|Crime|Thriller</td>
      <td>Universal Pictures|Original Film|Media Rights ...</td>
      <td>4/1/15</td>
      <td>2947</td>
      <td>7.3</td>
      <td>2015</td>
      <td>1.747999e+08</td>
      <td>1.385749e+09</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Looking at the shape of the dataset:

Output = 10866 rows, 21 columns


```python
df.shape
```




    (10866, 21)



In order to understand the data a little bit more, I performed some basic summary statistics:


```python
df.describe()
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
      <th>id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>1.086600e+04</td>
      <td>1.086600e+04</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>1.086600e+04</td>
      <td>1.086600e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>66064.177434</td>
      <td>0.646441</td>
      <td>1.462570e+07</td>
      <td>3.982332e+07</td>
      <td>102.070863</td>
      <td>217.389748</td>
      <td>5.974922</td>
      <td>2001.322658</td>
      <td>1.755104e+07</td>
      <td>5.136436e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>92130.136561</td>
      <td>1.000185</td>
      <td>3.091321e+07</td>
      <td>1.170035e+08</td>
      <td>31.381405</td>
      <td>575.619058</td>
      <td>0.935142</td>
      <td>12.812941</td>
      <td>3.430616e+07</td>
      <td>1.446325e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>0.000065</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>1.500000</td>
      <td>1960.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10596.250000</td>
      <td>0.207583</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>90.000000</td>
      <td>17.000000</td>
      <td>5.400000</td>
      <td>1995.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20669.000000</td>
      <td>0.383856</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>99.000000</td>
      <td>38.000000</td>
      <td>6.000000</td>
      <td>2006.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75610.000000</td>
      <td>0.713817</td>
      <td>1.500000e+07</td>
      <td>2.400000e+07</td>
      <td>111.000000</td>
      <td>145.750000</td>
      <td>6.600000</td>
      <td>2011.000000</td>
      <td>2.085325e+07</td>
      <td>3.369710e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>417859.000000</td>
      <td>32.985763</td>
      <td>4.250000e+08</td>
      <td>2.781506e+09</td>
      <td>900.000000</td>
      <td>9767.000000</td>
      <td>9.200000</td>
      <td>2015.000000</td>
      <td>4.250000e+08</td>
      <td>2.827124e+09</td>
    </tr>
  </tbody>
</table>
</div>



## <center>Data Cleaning</center>

I examined whether the dataset has missing values, looked for any wrong datatypes and reduntant columns:


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 21 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   id                    10866 non-null  int64  
     1   imdb_id               10856 non-null  object 
     2   popularity            10866 non-null  float64
     3   budget                10866 non-null  int64  
     4   revenue               10866 non-null  int64  
     5   original_title        10866 non-null  object 
     6   cast                  10790 non-null  object 
     7   homepage              2936 non-null   object 
     8   director              10822 non-null  object 
     9   tagline               8042 non-null   object 
     10  keywords              9373 non-null   object 
     11  overview              10862 non-null  object 
     12  runtime               10866 non-null  int64  
     13  genres                10843 non-null  object 
     14  production_companies  9836 non-null   object 
     15  release_date          10866 non-null  object 
     16  vote_count            10866 non-null  int64  
     17  vote_average          10866 non-null  float64
     18  release_year          10866 non-null  int64  
     19  budget_adj            10866 non-null  float64
     20  revenue_adj           10866 non-null  float64
    dtypes: float64(4), int64(6), object(11)
    memory usage: 1.7+ MB
    

Output: 

The dataset has some missing values, which are non-numerical data (object) and need to be replaced before starting our analysis. Also 8 columns are irrelevant and will be dropped.

I decided to drop 8 columns, which are irrelevant for the intented analysis. This will make the dataset more consistent and its handling easier:


```python
df = df.drop(['cast', 'homepage', 'tagline', 'keywords', 'overview', 'imdb_id', 'budget_adj', 'revenue_adj'], axis=1)
```

Columns dropped:
* cast 
* homepage 
* tagline
* keywords
* overview
* imdb id
* budget_adj
* revenue_adj

Checked results: The unnecessary 8 columns were dropped.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   id                    10866 non-null  int64  
     1   popularity            10866 non-null  float64
     2   budget                10866 non-null  int64  
     3   revenue               10866 non-null  int64  
     4   original_title        10866 non-null  object 
     5   director              10822 non-null  object 
     6   runtime               10866 non-null  int64  
     7   genres                10843 non-null  object 
     8   production_companies  9836 non-null   object 
     9   release_date          10866 non-null  object 
     10  vote_count            10866 non-null  int64  
     11  vote_average          10866 non-null  float64
     12  release_year          10866 non-null  int64  
    dtypes: float64(2), int64(6), object(5)
    memory usage: 1.1+ MB
    

I trimmed the dataset, replacing the empty values in 3 columns with the word 'nodata':


```python
df['director'] = df['director'].fillna('nodata')
df['production_companies'] = df['production_companies'].fillna('nodata')
df['genres'] = df['genres'].fillna('nodata')
```

Checked results: The empty values were replaced with the word 'nodata'


```python
df.query('director == "nodata"').head(5)
df.query('production_companies == "nodata"').head(5)
df.query('genres == "nodata"').head(5)
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
      <th>id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>424</th>
      <td>363869</td>
      <td>0.244648</td>
      <td>0</td>
      <td>0</td>
      <td>Belli di papÃ</td>
      <td>Guido Chiesa</td>
      <td>100</td>
      <td>nodata</td>
      <td>nodata</td>
      <td>10/29/15</td>
      <td>21</td>
      <td>6.1</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>620</th>
      <td>361043</td>
      <td>0.129696</td>
      <td>0</td>
      <td>0</td>
      <td>All Hallows' Eve 2</td>
      <td>Antonio Padovan|Bryan Norton|Marc Roussel|Ryan...</td>
      <td>90</td>
      <td>nodata</td>
      <td>Ruthless Pictures|Hollywood Shorts</td>
      <td>10/6/15</td>
      <td>13</td>
      <td>5.0</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>997</th>
      <td>287663</td>
      <td>0.330431</td>
      <td>0</td>
      <td>0</td>
      <td>Star Wars Rebels: Spark of Rebellion</td>
      <td>Steward Lee|Steven G. Lee</td>
      <td>44</td>
      <td>nodata</td>
      <td>nodata</td>
      <td>10/3/14</td>
      <td>13</td>
      <td>6.8</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>1712</th>
      <td>21634</td>
      <td>0.302095</td>
      <td>0</td>
      <td>0</td>
      <td>Prayers for Bobby</td>
      <td>Russell Mulcahy</td>
      <td>88</td>
      <td>nodata</td>
      <td>Daniel Sladek Entertainment</td>
      <td>2/27/09</td>
      <td>57</td>
      <td>7.4</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>40534</td>
      <td>0.020701</td>
      <td>0</td>
      <td>0</td>
      <td>Jonas Brothers: The Concert Experience</td>
      <td>Bruce Hendricks</td>
      <td>76</td>
      <td>nodata</td>
      <td>nodata</td>
      <td>2/27/09</td>
      <td>11</td>
      <td>7.0</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   id                    10866 non-null  int64  
     1   popularity            10866 non-null  float64
     2   budget                10866 non-null  int64  
     3   revenue               10866 non-null  int64  
     4   original_title        10866 non-null  object 
     5   director              10866 non-null  object 
     6   runtime               10866 non-null  int64  
     7   genres                10866 non-null  object 
     8   production_companies  10866 non-null  object 
     9   release_date          10866 non-null  object 
     10  vote_count            10866 non-null  int64  
     11  vote_average          10866 non-null  float64
     12  release_year          10866 non-null  int64  
    dtypes: float64(2), int64(6), object(5)
    memory usage: 1.1+ MB
    

<a id='eda'></a>
## <center>Exploratory Data Analysis</center>


### Question 1: How budgets, revenues, runtimes, costs per minute evolved over the years? What is the correlation between those metrics?


To answer this question for each metric, I will first group and plot the data , in order to gain insight and understand it better.


```python
# Grouping the data by release year and summing its attributes
df.groupby('release_year').sum()
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
      <th>id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_count</th>
      <th>vote_average</th>
    </tr>
    <tr>
      <th>release_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>502889</td>
      <td>14.685834</td>
      <td>22056948</td>
      <td>145005000</td>
      <td>3541</td>
      <td>2481</td>
      <td>202.4</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>578367</td>
      <td>13.107641</td>
      <td>46137000</td>
      <td>337720188</td>
      <td>3702</td>
      <td>2405</td>
      <td>197.6</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>544034</td>
      <td>14.553069</td>
      <td>54722126</td>
      <td>215579846</td>
      <td>3979</td>
      <td>2392</td>
      <td>203.0</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>562904</td>
      <td>17.092019</td>
      <td>73331500</td>
      <td>187404989</td>
      <td>3785</td>
      <td>2816</td>
      <td>215.2</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>729942</td>
      <td>17.321989</td>
      <td>39483161</td>
      <td>340981782</td>
      <td>4587</td>
      <td>3137</td>
      <td>260.9</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>614765</td>
      <td>11.990529</td>
      <td>70205115</td>
      <td>458081854</td>
      <td>4136</td>
      <td>1820</td>
      <td>216.8</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>759644</td>
      <td>13.989152</td>
      <td>57554800</td>
      <td>84736689</td>
      <td>4917</td>
      <td>1460</td>
      <td>281.9</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>697188</td>
      <td>18.846147</td>
      <td>100652200</td>
      <td>737834637</td>
      <td>4198</td>
      <td>3102</td>
      <td>249.7</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>726515</td>
      <td>18.883866</td>
      <td>71939000</td>
      <td>264732980</td>
      <td>4184</td>
      <td>4217</td>
      <td>248.8</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>591357</td>
      <td>13.106127</td>
      <td>42129087</td>
      <td>243957076</td>
      <td>3304</td>
      <td>1733</td>
      <td>184.8</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>751907</td>
      <td>14.009686</td>
      <td>126966946</td>
      <td>560221969</td>
      <td>4594</td>
      <td>2011</td>
      <td>263.1</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>1137584</td>
      <td>24.646153</td>
      <td>75997000</td>
      <td>404910610</td>
      <td>5925</td>
      <td>5081</td>
      <td>353.1</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>704684</td>
      <td>17.962162</td>
      <td>36279254</td>
      <td>494730171</td>
      <td>4078</td>
      <td>5433</td>
      <td>261.4</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>997893</td>
      <td>27.195419</td>
      <td>65190783</td>
      <td>1223981102</td>
      <td>5694</td>
      <td>5173</td>
      <td>368.7</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>910835</td>
      <td>21.047978</td>
      <td>76970000</td>
      <td>812539818</td>
      <td>4964</td>
      <td>5207</td>
      <td>300.6</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>760791</td>
      <td>22.855561</td>
      <td>56279000</td>
      <td>957489966</td>
      <td>4724</td>
      <td>6088</td>
      <td>281.2</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>744888</td>
      <td>22.302266</td>
      <td>122150000</td>
      <td>801005600</td>
      <td>5138</td>
      <td>4720</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>1420531</td>
      <td>35.704048</td>
      <td>161580000</td>
      <td>2180583159</td>
      <td>6105</td>
      <td>8031</td>
      <td>350.4</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>1411200</td>
      <td>26.865433</td>
      <td>208997011</td>
      <td>1369779659</td>
      <td>7155</td>
      <td>4898</td>
      <td>398.5</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>1086718</td>
      <td>33.590762</td>
      <td>254814000</td>
      <td>1684794913</td>
      <td>6385</td>
      <td>8830</td>
      <td>359.6</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>1639820</td>
      <td>38.660886</td>
      <td>362500000</td>
      <td>1768662387</td>
      <td>8385</td>
      <td>10898</td>
      <td>480.5</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>1560246</td>
      <td>36.277875</td>
      <td>373757786</td>
      <td>1774606236</td>
      <td>8681</td>
      <td>7480</td>
      <td>505.6</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>1403773</td>
      <td>44.011589</td>
      <td>437795002</td>
      <td>2458443852</td>
      <td>8418</td>
      <td>11385</td>
      <td>505.6</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>1400668</td>
      <td>43.315574</td>
      <td>519107412</td>
      <td>2307529320</td>
      <td>8240</td>
      <td>9886</td>
      <td>477.7</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>1611152</td>
      <td>62.038521</td>
      <td>729211964</td>
      <td>2635524418</td>
      <td>10864</td>
      <td>15371</td>
      <td>630.4</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>1629950</td>
      <td>63.662354</td>
      <td>748720637</td>
      <td>2875772392</td>
      <td>12463</td>
      <td>13761</td>
      <td>673.3</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>1886604</td>
      <td>61.655002</td>
      <td>704533613</td>
      <td>3002778281</td>
      <td>12092</td>
      <td>13675</td>
      <td>726.6</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>2048079</td>
      <td>63.079564</td>
      <td>709455811</td>
      <td>3462104847</td>
      <td>12646</td>
      <td>15064</td>
      <td>766.1</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>2707730</td>
      <td>67.430667</td>
      <td>925348000</td>
      <td>3739550845</td>
      <td>14760</td>
      <td>14576</td>
      <td>865.0</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>2629704</td>
      <td>77.221155</td>
      <td>1079656360</td>
      <td>5164923718</td>
      <td>14355</td>
      <td>18657</td>
      <td>831.6</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>1857851</td>
      <td>70.716996</td>
      <td>1289922066</td>
      <td>5315166660</td>
      <td>13946</td>
      <td>21575</td>
      <td>791.2</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>1905608</td>
      <td>66.644959</td>
      <td>1466233000</td>
      <td>4706599796</td>
      <td>13980</td>
      <td>17006</td>
      <td>799.9</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>1794978</td>
      <td>77.527364</td>
      <td>1441765370</td>
      <td>6078153217</td>
      <td>14235</td>
      <td>19447</td>
      <td>808.8</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>2788827</td>
      <td>97.375953</td>
      <td>1779628653</td>
      <td>6955151167</td>
      <td>19068</td>
      <td>24284</td>
      <td>1076.9</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>3231011</td>
      <td>123.063733</td>
      <td>2229207032</td>
      <td>7095429177</td>
      <td>19672</td>
      <td>37591</td>
      <td>1093.4</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>2920928</td>
      <td>124.376366</td>
      <td>2865884377</td>
      <td>9156341160</td>
      <td>18821</td>
      <td>35810</td>
      <td>1059.8</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>3066028</td>
      <td>123.372747</td>
      <td>3687042051</td>
      <td>8311492279</td>
      <td>21564</td>
      <td>26031</td>
      <td>1203.8</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>2925074</td>
      <td>136.704630</td>
      <td>4751086675</td>
      <td>10655173234</td>
      <td>20449</td>
      <td>40985</td>
      <td>1149.8</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>3092866</td>
      <td>131.494588</td>
      <td>4499660000</td>
      <td>9493174938</td>
      <td>22063</td>
      <td>40042</td>
      <td>1253.8</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>4150239</td>
      <td>144.658971</td>
      <td>5765235106</td>
      <td>11355712579</td>
      <td>24385</td>
      <td>53447</td>
      <td>1351.3</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>3611297</td>
      <td>124.123452</td>
      <td>5752700000</td>
      <td>10978701012</td>
      <td>23558</td>
      <td>46206</td>
      <td>1335.3</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>4144137</td>
      <td>170.043163</td>
      <td>5641944000</td>
      <td>13410083139</td>
      <td>26144</td>
      <td>63058</td>
      <td>1426.9</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>3607313</td>
      <td>186.586849</td>
      <td>5894640255</td>
      <td>14643618528</td>
      <td>28426</td>
      <td>62904</td>
      <td>1588.9</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>5527642</td>
      <td>202.062290</td>
      <td>6239857694</td>
      <td>15138243542</td>
      <td>28291</td>
      <td>68425</td>
      <td>1666.6</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>4468149</td>
      <td>221.788592</td>
      <td>7170340222</td>
      <td>16793822618</td>
      <td>32347</td>
      <td>79200</td>
      <td>1838.5</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>5140666</td>
      <td>228.833453</td>
      <td>7343284349</td>
      <td>16516835108</td>
      <td>37378</td>
      <td>73336</td>
      <td>2135.4</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>5906687</td>
      <td>247.399250</td>
      <td>7306185300</td>
      <td>16275739385</td>
      <td>41487</td>
      <td>76083</td>
      <td>2424.1</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>7005822</td>
      <td>259.804380</td>
      <td>7635569004</td>
      <td>19411668670</td>
      <td>43980</td>
      <td>90391</td>
      <td>2612.4</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>9101362</td>
      <td>290.069191</td>
      <td>7781262597</td>
      <td>19431695138</td>
      <td>49739</td>
      <td>102562</td>
      <td>2941.7</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>16241077</td>
      <td>319.895074</td>
      <td>8594084056</td>
      <td>22180170559</td>
      <td>52261</td>
      <td>119689</td>
      <td>3121.6</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>23087632</td>
      <td>316.029586</td>
      <td>9385001006</td>
      <td>21959998545</td>
      <td>48079</td>
      <td>130149</td>
      <td>2935.5</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>37302676</td>
      <td>364.537424</td>
      <td>9018153652</td>
      <td>23695591578</td>
      <td>52878</td>
      <td>135439</td>
      <td>3217.6</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>61545950</td>
      <td>357.031964</td>
      <td>8274084052</td>
      <td>24668428824</td>
      <td>57607</td>
      <td>183539</td>
      <td>3410.3</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>111910769</td>
      <td>413.606395</td>
      <td>9236038361</td>
      <td>24703633017</td>
      <td>63293</td>
      <td>214486</td>
      <td>3875.3</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>170103095</td>
      <td>621.087887</td>
      <td>7923990138</td>
      <td>24331150183</td>
      <td>68832</td>
      <td>206262</td>
      <td>4144.5</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>186663306</td>
      <td>648.283099</td>
      <td>7596547557</td>
      <td>26762450518</td>
      <td>60620</td>
      <td>182422</td>
      <td>3702.1</td>
    </tr>
  </tbody>
</table>
</div>



A first glimpse gives the intuition that the numbers have changed a lot over time with an upward trend. Is this the case? 

I will create charts with the two variables (revenue, budget), in order to have a view of the two trends and examine the correlation.


```python
ax = plt.subplots(figsize=(30, 10))
sns.barplot(x='release_year', y='revenue', data=df);
plt.title('Revenue generated from films per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('Revenue', fontsize=18)
```




    Text(0, 0.5, 'Revenue')




![png](output_28_1.png)



```python
ax = plt.subplots(figsize=(30, 10))
sns.barplot(x='release_year', y='budget', data=df);
plt.title('Budget invested in films per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('Budget', fontsize=18)
```




    Text(0, 0.5, 'Budget')




![png](output_29_1.png)



```python
ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='release_year', y='revenue', data=df);
sns.lineplot(x='release_year', y='budget', data=df);
plt.title('Chart line of film budget and revenue per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('Revenue', fontsize=18)
```




    Text(0, 0.5, 'Revenue')




![png](output_30_1.png)


It seems that the two variables follow identical trends, especially after 1990. The higher the budget for a movie is, the higher are the revenues that generates.

I also examined the correlation coefficient R between these two variables:


```python
df['revenue'].corr(df['budget'])
```




    0.7349006819076118



With a value of 0,74 there is a strong correlation between the two variables, which confirms my first conclusion.

What about the runtime of the movies over the years? 

This will give helpful insight regarding whether there's a correlation between the budget of a movie and its duration:


```python
df.groupby('release_year').sum()
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
      <th>id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_count</th>
      <th>vote_average</th>
    </tr>
    <tr>
      <th>release_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>502889</td>
      <td>14.685834</td>
      <td>22056948</td>
      <td>145005000</td>
      <td>3541</td>
      <td>2481</td>
      <td>202.4</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>578367</td>
      <td>13.107641</td>
      <td>46137000</td>
      <td>337720188</td>
      <td>3702</td>
      <td>2405</td>
      <td>197.6</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>544034</td>
      <td>14.553069</td>
      <td>54722126</td>
      <td>215579846</td>
      <td>3979</td>
      <td>2392</td>
      <td>203.0</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>562904</td>
      <td>17.092019</td>
      <td>73331500</td>
      <td>187404989</td>
      <td>3785</td>
      <td>2816</td>
      <td>215.2</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>729942</td>
      <td>17.321989</td>
      <td>39483161</td>
      <td>340981782</td>
      <td>4587</td>
      <td>3137</td>
      <td>260.9</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>614765</td>
      <td>11.990529</td>
      <td>70205115</td>
      <td>458081854</td>
      <td>4136</td>
      <td>1820</td>
      <td>216.8</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>759644</td>
      <td>13.989152</td>
      <td>57554800</td>
      <td>84736689</td>
      <td>4917</td>
      <td>1460</td>
      <td>281.9</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>697188</td>
      <td>18.846147</td>
      <td>100652200</td>
      <td>737834637</td>
      <td>4198</td>
      <td>3102</td>
      <td>249.7</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>726515</td>
      <td>18.883866</td>
      <td>71939000</td>
      <td>264732980</td>
      <td>4184</td>
      <td>4217</td>
      <td>248.8</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>591357</td>
      <td>13.106127</td>
      <td>42129087</td>
      <td>243957076</td>
      <td>3304</td>
      <td>1733</td>
      <td>184.8</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>751907</td>
      <td>14.009686</td>
      <td>126966946</td>
      <td>560221969</td>
      <td>4594</td>
      <td>2011</td>
      <td>263.1</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>1137584</td>
      <td>24.646153</td>
      <td>75997000</td>
      <td>404910610</td>
      <td>5925</td>
      <td>5081</td>
      <td>353.1</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>704684</td>
      <td>17.962162</td>
      <td>36279254</td>
      <td>494730171</td>
      <td>4078</td>
      <td>5433</td>
      <td>261.4</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>997893</td>
      <td>27.195419</td>
      <td>65190783</td>
      <td>1223981102</td>
      <td>5694</td>
      <td>5173</td>
      <td>368.7</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>910835</td>
      <td>21.047978</td>
      <td>76970000</td>
      <td>812539818</td>
      <td>4964</td>
      <td>5207</td>
      <td>300.6</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>760791</td>
      <td>22.855561</td>
      <td>56279000</td>
      <td>957489966</td>
      <td>4724</td>
      <td>6088</td>
      <td>281.2</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>744888</td>
      <td>22.302266</td>
      <td>122150000</td>
      <td>801005600</td>
      <td>5138</td>
      <td>4720</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>1420531</td>
      <td>35.704048</td>
      <td>161580000</td>
      <td>2180583159</td>
      <td>6105</td>
      <td>8031</td>
      <td>350.4</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>1411200</td>
      <td>26.865433</td>
      <td>208997011</td>
      <td>1369779659</td>
      <td>7155</td>
      <td>4898</td>
      <td>398.5</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>1086718</td>
      <td>33.590762</td>
      <td>254814000</td>
      <td>1684794913</td>
      <td>6385</td>
      <td>8830</td>
      <td>359.6</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>1639820</td>
      <td>38.660886</td>
      <td>362500000</td>
      <td>1768662387</td>
      <td>8385</td>
      <td>10898</td>
      <td>480.5</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>1560246</td>
      <td>36.277875</td>
      <td>373757786</td>
      <td>1774606236</td>
      <td>8681</td>
      <td>7480</td>
      <td>505.6</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>1403773</td>
      <td>44.011589</td>
      <td>437795002</td>
      <td>2458443852</td>
      <td>8418</td>
      <td>11385</td>
      <td>505.6</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>1400668</td>
      <td>43.315574</td>
      <td>519107412</td>
      <td>2307529320</td>
      <td>8240</td>
      <td>9886</td>
      <td>477.7</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>1611152</td>
      <td>62.038521</td>
      <td>729211964</td>
      <td>2635524418</td>
      <td>10864</td>
      <td>15371</td>
      <td>630.4</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>1629950</td>
      <td>63.662354</td>
      <td>748720637</td>
      <td>2875772392</td>
      <td>12463</td>
      <td>13761</td>
      <td>673.3</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>1886604</td>
      <td>61.655002</td>
      <td>704533613</td>
      <td>3002778281</td>
      <td>12092</td>
      <td>13675</td>
      <td>726.6</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>2048079</td>
      <td>63.079564</td>
      <td>709455811</td>
      <td>3462104847</td>
      <td>12646</td>
      <td>15064</td>
      <td>766.1</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>2707730</td>
      <td>67.430667</td>
      <td>925348000</td>
      <td>3739550845</td>
      <td>14760</td>
      <td>14576</td>
      <td>865.0</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>2629704</td>
      <td>77.221155</td>
      <td>1079656360</td>
      <td>5164923718</td>
      <td>14355</td>
      <td>18657</td>
      <td>831.6</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>1857851</td>
      <td>70.716996</td>
      <td>1289922066</td>
      <td>5315166660</td>
      <td>13946</td>
      <td>21575</td>
      <td>791.2</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>1905608</td>
      <td>66.644959</td>
      <td>1466233000</td>
      <td>4706599796</td>
      <td>13980</td>
      <td>17006</td>
      <td>799.9</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>1794978</td>
      <td>77.527364</td>
      <td>1441765370</td>
      <td>6078153217</td>
      <td>14235</td>
      <td>19447</td>
      <td>808.8</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>2788827</td>
      <td>97.375953</td>
      <td>1779628653</td>
      <td>6955151167</td>
      <td>19068</td>
      <td>24284</td>
      <td>1076.9</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>3231011</td>
      <td>123.063733</td>
      <td>2229207032</td>
      <td>7095429177</td>
      <td>19672</td>
      <td>37591</td>
      <td>1093.4</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>2920928</td>
      <td>124.376366</td>
      <td>2865884377</td>
      <td>9156341160</td>
      <td>18821</td>
      <td>35810</td>
      <td>1059.8</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>3066028</td>
      <td>123.372747</td>
      <td>3687042051</td>
      <td>8311492279</td>
      <td>21564</td>
      <td>26031</td>
      <td>1203.8</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>2925074</td>
      <td>136.704630</td>
      <td>4751086675</td>
      <td>10655173234</td>
      <td>20449</td>
      <td>40985</td>
      <td>1149.8</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>3092866</td>
      <td>131.494588</td>
      <td>4499660000</td>
      <td>9493174938</td>
      <td>22063</td>
      <td>40042</td>
      <td>1253.8</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>4150239</td>
      <td>144.658971</td>
      <td>5765235106</td>
      <td>11355712579</td>
      <td>24385</td>
      <td>53447</td>
      <td>1351.3</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>3611297</td>
      <td>124.123452</td>
      <td>5752700000</td>
      <td>10978701012</td>
      <td>23558</td>
      <td>46206</td>
      <td>1335.3</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>4144137</td>
      <td>170.043163</td>
      <td>5641944000</td>
      <td>13410083139</td>
      <td>26144</td>
      <td>63058</td>
      <td>1426.9</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>3607313</td>
      <td>186.586849</td>
      <td>5894640255</td>
      <td>14643618528</td>
      <td>28426</td>
      <td>62904</td>
      <td>1588.9</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>5527642</td>
      <td>202.062290</td>
      <td>6239857694</td>
      <td>15138243542</td>
      <td>28291</td>
      <td>68425</td>
      <td>1666.6</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>4468149</td>
      <td>221.788592</td>
      <td>7170340222</td>
      <td>16793822618</td>
      <td>32347</td>
      <td>79200</td>
      <td>1838.5</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>5140666</td>
      <td>228.833453</td>
      <td>7343284349</td>
      <td>16516835108</td>
      <td>37378</td>
      <td>73336</td>
      <td>2135.4</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>5906687</td>
      <td>247.399250</td>
      <td>7306185300</td>
      <td>16275739385</td>
      <td>41487</td>
      <td>76083</td>
      <td>2424.1</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>7005822</td>
      <td>259.804380</td>
      <td>7635569004</td>
      <td>19411668670</td>
      <td>43980</td>
      <td>90391</td>
      <td>2612.4</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>9101362</td>
      <td>290.069191</td>
      <td>7781262597</td>
      <td>19431695138</td>
      <td>49739</td>
      <td>102562</td>
      <td>2941.7</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>16241077</td>
      <td>319.895074</td>
      <td>8594084056</td>
      <td>22180170559</td>
      <td>52261</td>
      <td>119689</td>
      <td>3121.6</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>23087632</td>
      <td>316.029586</td>
      <td>9385001006</td>
      <td>21959998545</td>
      <td>48079</td>
      <td>130149</td>
      <td>2935.5</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>37302676</td>
      <td>364.537424</td>
      <td>9018153652</td>
      <td>23695591578</td>
      <td>52878</td>
      <td>135439</td>
      <td>3217.6</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>61545950</td>
      <td>357.031964</td>
      <td>8274084052</td>
      <td>24668428824</td>
      <td>57607</td>
      <td>183539</td>
      <td>3410.3</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>111910769</td>
      <td>413.606395</td>
      <td>9236038361</td>
      <td>24703633017</td>
      <td>63293</td>
      <td>214486</td>
      <td>3875.3</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>170103095</td>
      <td>621.087887</td>
      <td>7923990138</td>
      <td>24331150183</td>
      <td>68832</td>
      <td>206262</td>
      <td>4144.5</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>186663306</td>
      <td>648.283099</td>
      <td>7596547557</td>
      <td>26762450518</td>
      <td>60620</td>
      <td>182422</td>
      <td>3702.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = plt.subplots(figsize=(30, 10))
sns.lineplot(y="runtime", x="release_year", data=df, color='b');
plt.title('Runtime of films per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('Runtime', fontsize=18)
```




    Text(0, 0.5, 'Runtime')




![png](output_36_1.png)



```python

```

The plot shows a decreasing tendency for the decade 1960 - 1970. 

From that point and until the mid 90's we can see an increase of the movies' runtimes, only to reach mid 60's levels and, since then, the tendency is steadily decreasing.

This is opposite to the tendency of the movies' budgets over the years, which have seen a steady increase since the beginning of the examined timeline.

I also examined the correlation coefficient R between these two variables:


```python
df['runtime'].corr(df['budget'])
```




    0.19128265656578047



With a value of 0,19 there is a weak or no correlation between the two variables, which confirms my first conclusion.

There's another one metric, which needs to be examined: The distribution (N) of the films' runtimes. This will give useful info about the standard trend of filmaking regarding the runtime attribute. I created a histogram to examine it:


```python
ax = plt.subplots(figsize=(30, 10))
sns.distplot(df['runtime'], bins=200)
plt.title('Distribution of runtime of films', fontsize=28)
plt.xlabel('Runtime in Minutes',fontsize=18)
plt.ylabel('N',fontsize=18)
plt.xlim(xmin=0, xmax = 200)
```




    (0, 200)




![png](output_42_1.png)


The histogram shows that the films with a runtime of 95 minutes are the most frequent in the population.

I also examined the tendency for another important metric: Cost per minute (cpm). To do so, I created a new, derived column, which gives the result of budget/runtime:


```python
df['cpm'] = df['budget']/df['runtime']
df

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
      <th>id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>cpm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135397</td>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>6/9/15</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.209677e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76341</td>
      <td>28.419936</td>
      <td>150000000</td>
      <td>378436354</td>
      <td>Mad Max: Fury Road</td>
      <td>George Miller</td>
      <td>120</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Village Roadshow Pictures|Kennedy Miller Produ...</td>
      <td>5/13/15</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>1.250000e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>262500</td>
      <td>13.112507</td>
      <td>110000000</td>
      <td>295238201</td>
      <td>Insurgent</td>
      <td>Robert Schwentke</td>
      <td>119</td>
      <td>Adventure|Science Fiction|Thriller</td>
      <td>Summit Entertainment|Mandeville Films|Red Wago...</td>
      <td>3/18/15</td>
      <td>2480</td>
      <td>6.3</td>
      <td>2015</td>
      <td>9.243697e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140607</td>
      <td>11.173104</td>
      <td>200000000</td>
      <td>2068178225</td>
      <td>Star Wars: The Force Awakens</td>
      <td>J.J. Abrams</td>
      <td>136</td>
      <td>Action|Adventure|Science Fiction|Fantasy</td>
      <td>Lucasfilm|Truenorth Productions|Bad Robot</td>
      <td>12/15/15</td>
      <td>5292</td>
      <td>7.5</td>
      <td>2015</td>
      <td>1.470588e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>168259</td>
      <td>9.335014</td>
      <td>190000000</td>
      <td>1506249360</td>
      <td>Furious 7</td>
      <td>James Wan</td>
      <td>137</td>
      <td>Action|Crime|Thriller</td>
      <td>Universal Pictures|Original Film|Media Rights ...</td>
      <td>4/1/15</td>
      <td>2947</td>
      <td>7.3</td>
      <td>2015</td>
      <td>1.386861e+06</td>
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
      <th>10861</th>
      <td>21</td>
      <td>0.080598</td>
      <td>0</td>
      <td>0</td>
      <td>The Endless Summer</td>
      <td>Bruce Brown</td>
      <td>95</td>
      <td>Documentary</td>
      <td>Bruce Brown Films</td>
      <td>6/15/66</td>
      <td>11</td>
      <td>7.4</td>
      <td>1966</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>10862</th>
      <td>20379</td>
      <td>0.065543</td>
      <td>0</td>
      <td>0</td>
      <td>Grand Prix</td>
      <td>John Frankenheimer</td>
      <td>176</td>
      <td>Action|Adventure|Drama</td>
      <td>Cherokee Productions|Joel Productions|Douglas ...</td>
      <td>12/21/66</td>
      <td>20</td>
      <td>5.7</td>
      <td>1966</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>10863</th>
      <td>39768</td>
      <td>0.065141</td>
      <td>0</td>
      <td>0</td>
      <td>Beregis Avtomobilya</td>
      <td>Eldar Ryazanov</td>
      <td>94</td>
      <td>Mystery|Comedy</td>
      <td>Mosfilm</td>
      <td>1/1/66</td>
      <td>11</td>
      <td>6.5</td>
      <td>1966</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>10864</th>
      <td>21449</td>
      <td>0.064317</td>
      <td>0</td>
      <td>0</td>
      <td>What's Up, Tiger Lily?</td>
      <td>Woody Allen</td>
      <td>80</td>
      <td>Action|Comedy</td>
      <td>Benedict Pictures Corp.</td>
      <td>11/2/66</td>
      <td>22</td>
      <td>5.4</td>
      <td>1966</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>10865</th>
      <td>22293</td>
      <td>0.035919</td>
      <td>19000</td>
      <td>0</td>
      <td>Manos: The Hands of Fate</td>
      <td>Harold P. Warren</td>
      <td>74</td>
      <td>Horror</td>
      <td>Norm-Iris</td>
      <td>11/15/66</td>
      <td>15</td>
      <td>1.5</td>
      <td>1966</td>
      <td>2.567568e+02</td>
    </tr>
  </tbody>
</table>
<p>10866 rows × 14 columns</p>
</div>




```python
ax = plt.subplots(figsize=(30, 10))

sns.barplot(x='release_year', y='cpm', data=df);
plt.title('Cost per minute of films per release year', fontsize=28)
plt.xlabel('Release Year', fontsize=18)
plt.ylabel('CPM', fontsize=18)
```




    Text(0, 0.5, 'CPM')




![png](output_46_1.png)


The bar plot shows increasing tendency for the costs per minute from 1960 until 1999. From that point the CPM of the movies are steadily decreasing.

<a id='conclusions'></a>
## <center>Conclusions</center>


1. The research question "How budgets, revenues, runtimes, costs per minute evolved over the years? What is the correlation between those metrics?" has shown as results that **budgets and revenues have steadily increased over time** with a strong correlation. It seems that nowadays we have overall more costly movies that generate higher revenues.

2. The runtime of the movies has seen a **steady decrease over the examined timeline**. In the recent decades the movies have been shorter with a tendency for a runtime of 95 minutes.

3. The metric **Costs per minute has seen a steady increase until the year 1999** and since then the costs are decreasing. This is in accordance with the other two metrics (budget, runtime) and all three variables have identical tendencies.

**All results are limited to the underlying dataset and since no advanced statistics were performed, the results can only be treated as indicators and are not generalizable. Furthermore, one has to consider that many entries in the dataset have been removed due to missing data**
