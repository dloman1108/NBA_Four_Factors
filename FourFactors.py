
# coding: utf-8

# In[3]:

#Import packages
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import urllib2
import re
import sklearn


# In[5]:

filepath = '/Users/DanLo1108/Documents/Projects/NBA Analysis/'


# In[3]:

url_list=[]
for i in range(1,21):
    url = "http://www.nbaminer.com/nbaminer_nbaminer/four_factors.php?operation=eexcel&partitionpage="+str(i)+"&partition2page=1"  
    url_list.append(url)


# In[4]:

#Gets contents of url web page
request=urllib2.Request(url)
page = urllib2.urlopen(request)


# In[5]:

#Reads contents of page
content=page.read()
soup=BeautifulSoup(content,'lxml')


# In[6]:

table=soup.find_all('table')


# In[7]:

results=table[0].find_all('td')


# In[8]:

#Creates dictionary of data
ff_dict = {'Team': [],
           'Off_Eff_FGperc': [],
           'Def_Eff_FGperc': [],
           'Off_FTRate': [],
           'Def_FTRate': [],
           'Off_TORate': [],
           'Def_TORate': [],
           'Off_ORRate': [],
           'Def_ORRate': [],
           'Wins': [],
           'Year': [],
           'Games': []}


# In[9]:

#Appends data to dictionary
count = 0
for item in results[14:]:
    count += 1
    if np.mod(count,14) == 1:
        ff_dict['Team'].append(item.string)
    if np.mod(count,14) == 2:
        ff_dict['Off_Eff_FGperc'].append(float(item.string))
    if np.mod(count,14) == 3:
        ff_dict['Def_Eff_FGperc'].append(float(item.string))
    if np.mod(count,14) == 4:
        ff_dict['Off_FTRate'].append(float(item.string))
    if np.mod(count,14) == 5:
        ff_dict['Def_FTRate'].append(float(item.string))
    if np.mod(count,14) == 6:
        ff_dict['Off_TORate'].append(float(item.string))
    if np.mod(count,14) == 7:
        ff_dict['Def_TORate'].append(float(item.string))
    if np.mod(count,14) == 8:
        ff_dict['Off_ORRate'].append(float(item.string))
    if np.mod(count,14) == 9:
        ff_dict['Def_ORRate'].append(float(item.string))
    if np.mod(count,14) == 10:
        ff_dict['Wins'].append(float(item.string))
    if np.mod(count,14) == 12:
        ff_dict['Year'].append(item.string)
    if np.mod(count,14) == 0:
        ff_dict['Games'].append(float(item.string))


# In[10]:

#Creates dataframe
Four_Factors = pd.DataFrame()
for key in ff_dict:
    Four_Factors[key] = ff_dict[key]


# In[11]:

def get_wins_82(x):
    return x.Wins/x.Games*82

Four_Factors['wins_82'] = Four_Factors.apply(lambda x: get_wins_82(x), axis=1)


# In[12]:

Four_Factors = Four_Factors[['Team','Year','Off_Eff_FGperc','Off_ORRate','Off_TORate',
                            'Off_FTRate','Def_Eff_FGperc','Def_ORRate','Def_TORate',
                            'Def_FTRate','wins_82']]


# In[14]:

Four_Factors.to_csv(filepath+'Four_Factors_data.csv',index=False)


# In[6]:

Four_Factors = pd.read_csv(filepath+'Four_Factors_data.csv')


# In[4]:

#Creates X and y
X = Four_Factors[['Off_Eff_FGperc','Off_ORRate','Off_TORate',
                 'Off_FTRate','Def_Eff_FGperc','Def_ORRate','Def_TORate',
                 'Def_FTRate']]

y = Four_Factors['wins_82']


# In[16]:

#Runs linear regression and finds MAE (2.54)

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

maes=[]
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    linreg = LinearRegression()
    linreg.fit(X_train,y_train)
    pred = linreg.predict(X_test)

    maes.append(np.mean(abs(pred-y_test)))
    
print np.mean(maes)


# In[17]:

#Finds mean and standard deviation of 4 factors
ave_Off_Eff_FGperc = np.mean(X.Off_Eff_FGperc)
ave_Off_ORRate = np.mean(X.Off_ORRate)
ave_Off_TORate = np.mean(X.Off_TORate)
ave_Off_FTRate = np.mean(X.Off_FTRate)
ave_Def_Eff_FGperc = np.mean(X.Def_Eff_FGperc)
ave_Def_ORRate = np.mean(X.Def_ORRate)
ave_Def_TORate = np.mean(X.Def_TORate)
ave_Def_FTRate = np.mean(X.Def_FTRate)

std_Off_Eff_FGperc = np.std(X.Off_Eff_FGperc)
std_Off_ORRate = np.std(X.Off_ORRate)
std_Off_TORate = np.std(X.Off_TORate)
std_Off_FTRate = np.std(X.Off_FTRate)
std_Def_Eff_FGperc = np.std(X.Def_Eff_FGperc)
std_Def_ORRate = np.std(X.Def_ORRate)
std_Def_TORate = np.std(X.Def_TORate)
std_Def_FTRate = np.std(X.Def_FTRate)


#Predicts 41 wins with average stats
linreg.predict(np.array([ave_Off_Eff_FGperc,ave_Off_ORRate,ave_Off_TORate,
                         ave_Off_FTRate,ave_Def_Eff_FGperc,ave_Def_ORRate,
                         ave_Def_TORate,ave_Def_FTRate]))


# In[18]:

#Gets data from this year
this_year=Four_Factors[Four_Factors.Year=='2014-2015']


# In[19]:

#Predicts wins based on Four Factors stats
teams=this_year.Team
this_year_X=this_year[['Off_Eff_FGperc','Off_ORRate','Off_TORate','Off_FTRate',
                       'Def_Eff_FGperc','Def_ORRate','Def_TORate','Def_FTRate']]
wins=this_year.wins_82
wins_pred=linreg.predict(this_year_X)


# In[20]:

#Normalize so that sum of predicted wins = 1230 
#(avg of 41 wins for 30 teams) and round

wins_pred=(wins_pred/sum(wins_pred))*1230
wins_pred=np.array(map(lambda x: round(x,2),wins_pred))
wins_diff=wins-wins_pred
wins_diff=np.array(map(lambda x: round(x,2),wins_diff))


# In[21]:

teams_analysis=pd.DataFrame({'Team': teams,'Predicted wins': wins_pred,'Actual wins': wins,
                             'Difference': wins_diff})

teams_analysis=teams_analysis[['Team','Predicted wins','Actual wins','Difference']]


# In[22]:

teams_analysis


# In[181]:

teams_analysis_sort=teams_analysis.sort('Difference',ascending=False)


# In[183]:

teams_analysis_sort.to_csv(filepath+'teams_analysis.csv',index=False)


# In[38]:

#Gets importance of variables

coef_adj=abs(linreg.coef_/sum(abs(linreg.coef_)))

EFG_imp=coef_adj[0]+coef_adj[4]
ORR_imp=coef_adj[1]+coef_adj[5]
TOR_imp=coef_adj[2]+coef_adj[6]
FTR_imp=coef_adj[3]+coef_adj[7]


# In[40]:

EFG_imp,ORR_imp,TOR_imp,FTR_imp


# In[28]:

#Correlations
np.corrcoef(Four_Factors.Off_ORRate,Four_Factors.wins_82)[1][0]


# In[27]:

np.corrcoef(Four_Factors.Def_Eff_FGperc,Four_Factors.wins_82)[1][0]


# In[30]:

np.corrcoef(Four_Factors.Def_TORate,Four_Factors.wins_82)[1][0]


# In[32]:

np.corrcoef(Four_Factors.Def_FTRate,Four_Factors.wins_82)[1][0]


# In[ ]:



