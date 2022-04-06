import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(r"C:\Users\User\OneDrive - The American College of Greece\Documents\Εξόρυξη Δεδομένων\healthcare-dataset-stroke-data.csv")
replace_gender = {'gender':{'Male':0, "Female":1, 'Other':2}}
replace_ever_married = {'ever_married':{'Yes':0, 'No':1}}
replace_work_type = {'work_type':{'Private':0, 'Self-employed':1, 'children':2, 'Govt_job':3, 'Never_worked':4}}
replace_Residence_type = {'Residence_type':{'Urban':0, 'Rural':1}}
replace_smoking_status = {'smoking_status':{'formerly smoked':0, 'never smoked':1, 'smokes':2, 'Unknown':3}}
n=75
q=25

#data[data.isnull().any(axis=1)]cleanup null values

data = data.replace(replace_gender)
data = data.replace(replace_ever_married)
data = data.replace(replace_work_type)
data = data.replace(replace_Residence_type)
data = data.replace(replace_smoking_status)

data['bmi'].interpolate(method='linear', inplace=True)
# data = data.sample(frac=1).reset_index(drop=True)
np.savetxt('test.out', data, fmt="%10.5f")
# data.pop('id')
# data.pop('smoking_status')
# data.pop('bmi')
# data.pop('Residence_type')
# data.pop('work_type')
# data.pop('ever_married')
data_stroke = data.pop('stroke')


clf = RandomForestClassifier(random_state=True,n_estimators=100, class_weight='balanced')
clf.fit(data.head(int(len(data)*(n/100))), data_stroke.head(int(len(data)*(n/100))))
X = clf.predict(data.tail(int(len(data)*(q/100))))

A = f1_score(data_stroke.tail(int(len(data)*(q/100))), X, average='macro')
B = precision_score(data_stroke.tail(int(len(data)*(q/100))), X, average='macro')
C = recall_score(data_stroke.tail(int(len(data)*(q/100))), X, average='macro', zero_division=1)

errors = abs(X - data_stroke.tail(int(len(data)*(q/100))))
# print(round(np.mean(errors), 2))


print(A)
print(B)
print(C)