
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# ## 데이터 전처리 (data_preprocessing.ipynb)
# * Date Split
# * Weekday
# * Lunar Date
# * Date Normalization
# * 식사명 -> one-hot
# * 식사내용 -> bag-of-word

# In[3]:


# Read Data
train_df = pd.read_excel("data/train.xlsx")
test_df = pd.read_excel("data/test.xlsx")


# In[4]:


train_df.head()


# In[5]:


test_df.head()


# In[6]:


df = pd.concat([train_df, test_df])
df = df.sort_values(by=['일자']).reset_index(drop=True)
print(df[:3])
print(df[-3:])


# In[7]:


# 식사명 변환 (one-hot)
def convert_ont_hot(df):
    df = df.join(pd.get_dummies(df['식사명'], prefix='식사명'))
    df.drop(['식사명'], axis=1, inplace=True)
    return df


# In[8]:


def moving_average(df, window_size):
    morning = df['수량'][df['식사명_아침']==1].rolling(window_size, min_periods=1).mean().shift(3)
    lunch = df['수량'][df['식사명_점심(일반)']==1].rolling(window_size, min_periods=1).mean().shift(3)
    lunch_west = df['수량'][df['식사명_점심(양식)']==1].rolling(window_size, min_periods=1).mean().shift(3)
    dinner = df['수량'][df['식사명_저녁']==1].rolling(window_size, min_periods=1).mean().shift(3)
    
    return pd.concat([morning, lunch, lunch_west, dinner]).sort_index()


# In[9]:


# 식사내용 변환 (Bag-of-Word)
from sklearn.feature_extraction.text import CountVectorizer
def tokenize(text):
        return text.split(',')
def convert_bow(df):
    vectorizer = CountVectorizer(tokenizer=tokenize)
    bow = vectorizer.fit_transform(df['식사내용']).toarray()
    df = df.join(pd.DataFrame(bow, columns=vectorizer.get_feature_names()))
    df.drop(['식사내용'], axis=1, inplace=True)
    return df


# In[10]:


# 년/월/일 분리(split) + 요일(Weekday) 추가
def split_date(df):
    # Normalize Date
    df['year'] = (df['일자'] / 10000).astype(int)
    df['month'] = (df['일자'] % 10000 / 100).astype(int)
    df['day'] = (df['일자'] % 100).astype(int)
    df['weekday'] = pd.to_datetime(df['일자'], format = '%Y%m%d').dt.dayofweek
    # df.drop(['일자'], axis=1, inplace=True)


# In[11]:


# 음력 추가
from korean_lunar_calendar import KoreanLunarCalendar
from datetime import datetime

def add_lunar_date(df):
    calendar = KoreanLunarCalendar()
    
    lunar_y = []
    lunar_m = []
    lunar_d = []
    for y, m, d in zip (df['year'], df['month'], df['day']):
        calendar.setSolarDate(y, m, d)
        lunar_date = calendar.LunarIsoFormat()
        lunar_y.append(int(lunar_date[:4]))
        lunar_m.append(int(lunar_date[5:7]))
        lunar_d.append(int(lunar_date[8:10]))
        
    df['lunar_year'], df['lunar_month'], df['lunar_day'] = lunar_y, lunar_m, lunar_d


# In[12]:


# 년/월/일 변환
def year_norm(df):
    df['year'] = (df['year']-min(df['year'])) / (max(df['year'])-min(df['year']))
    df['lunar_year'] = (df['lunar_year']-min(df['lunar_year'])) / (max(df['lunar_year'])-min(df['lunar_year']))
def month_norm(df):
    df['month_sin'] = [np.sin(x*2*np.pi/12) for x in df['month']]
    df['month_cos'] = [np.cos(x*2*np.pi/12) for x in df['month']]
    df['lunar_month_sin'] = [np.sin(x*2*np.pi/12) for x in df['lunar_month']]
    df['lunar_month_cos'] = [np.cos(x*2*np.pi/12) for x in df['lunar_month']]
    df.drop(['month', 'lunar_month'], axis=1, inplace=True)
def day_norm(df):
    df['day_sin'] = [np.sin(x*2*np.pi/31) for x in df['day']]
    df['day_cos'] = [np.cos(x*2*np.pi/31) for x in df['day']]
    df['lunar_ay_sin'] = [np.sin(x*2*np.pi/31) for x in df['lunar_day']]
    df['lunar_day_cos'] = [np.cos(x*2*np.pi/31) for x in df['lunar_day']]
    df.drop(['day', 'lunar_day'], axis=1, inplace=True)
def weekday_norm(df):
    df['weekday_sin'] = [np.sin(x*2*np.pi/7) for x in df['weekday']]
    df['weekday_cos'] = [np.cos(x*2*np.pi/7) for x in df['weekday']]


# In[13]:


# convert 식사명 to one-hot
df = convert_ont_hot(df)

# Moving Average of 수량
df['MA_week'] = moving_average(df, 7)
df['MA_month'] = moving_average(df, 30)
df['MA_half_year'] = moving_average(df, 180)
df['MA_year'] = moving_average(df, 365)
df.drop(df[df.일자 < 20040326].index, inplace=True)
df.drop(df[(df.일자 > 20050109) & (df.일자 < 20060331) & (df['식사명_점심(양식)']==1)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

# convert 식사내용 to Bag-of-Word Vector
df = convert_bow(df)

# Date
split_date(df)
add_lunar_date(df)

# Date Normalization
year_norm(df)
month_norm(df)
day_norm(df)
weekday_norm(df)


# In[14]:


print("Number of Columns =", len(df.columns))
df.head()


# # Modeling

# 1. Random Forest
# 2. XGBoost

# ## 0. Prepare train & test

# #### 1) Split X and Y

# In[20]:


train_df = df.drop(df[df['일자'].isin(test_df['일자'].unique())].index)
train_y = train_df['수량']
train_x = train_df.drop(['수량', '일자'], axis=1)

test_df = df[df['일자'].isin(test_df['일자'].unique())]
test_x = test_df.drop(['수량', '일자'], axis=1)
test_x_iter = test_df.drop(['수량'], axis=1)


# #### 2) Train Model

# In[16]:


def train_and_predict(model, train_x, train_y, dev_x):
    model.fit(train_x, train_y)
    return model.predict(dev_x)

def iterative_train_and_predict(model, df, dev_x):
    predictions = []
    dev_dates = dev_x['일자'].unique()
    for i, dev_date in enumerate(dev_dates):        
        _train_df = df[df['일자'] < dev_date - 2]
        pred = train_and_predict(model, 
                                 _train_df.drop(['수량', '일자'], axis=1), 
                                 _train_df['수량'], 
                                 dev_x[dev_x['일자']==dev_date].drop(['일자'], axis=1))
        predictions.append(pred)
        
        if((i+1) % (int(len(dev_dates)/10)) == 0):
            print(">>", 10*int((i+1)/(int(len(dev_dates)/10))), "% >>", end="")
    print()
    return np.concatenate(predictions)


# In[18]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 100, random_state = 10, warm_start=False)


# In[24]:

print("Start Prediction !")
pred = iterative_train_and_predict(model, train_df, test_x_iter)


# #### 3) Prediction

# In[59]:


test_x['수량'] = pred


# In[74]:


result = pd.DataFrame()
result['일자'] = test_x['일자'].unique()
result['아침'] = pd.merge(result, test_x[test_x['식사명_아침']==1][['일자','수량']], how='outer', on=['일자'])['수량']
result['점심(일반)'] = pd.merge(result, test_x[test_x['식사명_점심(일반)']==1][['일자','수량']], how='outer', on=['일자'])['수량']
result['점심(양식)'] = pd.merge(result, test_x[test_x['식사명_점심(양식)']==1][['일자','수량']], how='outer', on=['일자'])['수량']
result['저녁'] = pd.merge(result, test_x[test_x['식사명_저녁']==1][['일자','수량']], how='outer', on=['일자'])['수량']
result.head()


# In[81]:


output = result[2::3]
output.head()


# In[82]:


output.isnull().any()


# In[88]:


output.to_csv("submission/submission_180801.csv", encoding='utf-8-sig', index=False)

