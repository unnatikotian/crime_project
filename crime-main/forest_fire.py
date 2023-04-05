#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings("ignore")

df = pd.read_csv("crime.csv")

# Testing out the time and date conversion for one entry

t = df['Date'][20]
print(t)
s1 = t[:11]
print(s1)
s2 = t[11:]
print(s2)

print(s2)
hr = s2[:2]
mins = s2[3:5]
sec = s2[6:8]
time_frame = s2[9:]
if (time_frame == 'PM'):
    if (int(hr) != 12):
        hr = str(int(hr) + 12)
else:
    if (int(hr) == 12):
        hr = '00'

print(hr, mins, sec)

# Testing out our code

month = s1[:2]
date = s1[3:5]
year = s1[6:10]

final_date = datetime(int(year), int(month), int(date), int(hr), int(mins), int(sec))
print(final_date)

# Testing out our code

month = s1[:2]
date = s1[3:5]
year = s1[6:10]

final_date = datetime(int(year), int(month), int(date), int(hr), int(mins), int(sec))
print(final_date)


# Time Conversion Function
def time_convert(date_time):
    s1 = date_time[:11]
    s2 = date_time[11:]

    month = s1[:2]
    date = s1[3:5]
    year = s1[6:10]

    hr = s2[:2]
    mins = s2[3:5]
    sec = s2[6:8]
    time_frame = s2[9:]
    if (time_frame == 'PM'):
        if (int(hr) != 12):
            hr = str(int(hr) + 12)
    else:
        if (int(hr) == 12):
            hr = '00'

    final_date = datetime(int(year), int(month), int(date), int(hr), int(mins), int(sec))
    return final_date


# Using apply() of pandas to apply time_convert on every row of the Date column
df['Date'] = df['Date'].apply(time_convert)


def month(x):
    return x.strftime("%B")


df['Month'] = df['Date'].apply(month)

# Make a new dataset for the predictions
cols = ['Date', 'Block', 'Location Description', 'Domestic', 'District', 'Month', 'Primary Type']
new_df = df[cols]


def new_hour(x):
    return int(x.strftime("%H"))


new_df['Hour'] = new_df['Date'].apply(new_hour)


def new_day(x):
    return int(x.strftime("%w"))


new_df['Day'] = new_df['Date'].apply(new_day)


def new_month(x):
    return int(x.strftime("%m"))


new_df['Month_num'] = new_df['Date'].apply(new_month)

new_df['Location Description'] = new_df['Location Description'].astype('category')
new_df['Domestic'] = new_df['Domestic'].astype('category')
# new_df['Primary Type'] = new_df['Primary Type'].astype('category')
# new_df.dtypes
new_df['Location_Cat'] = new_df['Location Description'].cat.codes
new_df['Domestic_Cat'] = new_df['Domestic'].cat.codes


def day_conv(x):
    return x.strftime("%a")


new_df['Day Name'] = new_df['Date'].apply(day_conv)

# Creating our explicit dataset
cri4 = new_df.groupby(['Month_num', 'Day', 'District', 'Hour'], as_index=False).agg({"Primary Type": "count"})
cri4 = cri4.sort_values(by=['District'], ascending=False)


# Feature Engineer and create a new feature
def crime_rate_assign(x):
    if (x <= 7):
        return 0
    else:
        return 1


cri4['Alarm'] = cri4['Primary Type'].apply(crime_rate_assign)
cri4 = cri4[['Month_num', 'Day', 'Hour', 'District', 'Primary Type', 'Alarm']]

cri4 = np.array(cri4)

X = cri4[1:, 0:4]
y = cri4[1:, 5]

y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

inputt = [int(x) for x in "2 0 22 31".split(' ')]
final = [np.array(inputt)]

b = classifier.predict_proba(final)

print(b)

pickle.dump(classifier, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
