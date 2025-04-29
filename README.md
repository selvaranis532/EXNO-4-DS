# EXNO:4
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi (1).csv")
df
```
![image](https://github.com/user-attachments/assets/d79308b8-b175-42e5-87f1-ae875894aaab)
```
df.head()
```
![image](https://github.com/user-attachments/assets/b2e37cb5-e2f5-4375-b68c-e348ea266f72)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/1cb5a146-2961-4de4-b87c-0253bc93ddd6)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/4f8fadd6-fcd3-46fc-a7ec-aabba5c52223)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/7e8bc20e-3fa9-4f22-bca4-51d80cfaa99a)
```
df1=pd.read_csv("/content/bmi (1).csv")
df2=pd.read_csv("/content/bmi (1).csv")
df3=pd.read_csv("/content/bmi (1).csv")
df4=pd.read_csv("/content/bmi (1).csv")
df5=pd.read_csv("/content/bmi (1).csv")
df1
```
![image](https://github.com/user-attachments/assets/cd6729fa-2b1a-4d26-8ce8-7d45648c10b6)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/029a742d-a099-4197-a8ec-1b43e7d31959)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
![image](https://github.com/user-attachments/assets/2030ec42-4fd3-45ab-b93a-aa1441fd37f3)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/fa8c90f9-4227-4560-afc5-0ae0c65db3cf)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df4
```
![image](https://github.com/user-attachments/assets/46cbd2be-74a7-4b1e-84b1-2ef6997faa72)
```
import seaborn as sns
import pandas as pd

import numpy as np 
import seaborn as sns
```
```
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
```
```
data=pd.read_csv("/content/titanic_dataset (2).csv")
data
```
![image](https://github.com/user-attachments/assets/00c5b36c-1e73-4fd7-b397-18d5d3cd41da)
```
data=data.dropna()
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
```
```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")
```
```
data
```
![image](https://github.com/user-attachments/assets/d1a1b377-a66e-478b-82ce-090b603c67c0)

```
k=5 
selector=SelectKBest(score_func=chi2, k=k) 
x=pd.get_dummies(x) 
x_new=selector.fit_transform(x,y)
```
```
x_encoded =pd.get_dummies(x) 
selector=SelectKBest(score_func=chi2, k=5) 
x_new = selector.fit_transform(x_encoded,y)
```
```
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected_Feature:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/f3313d55-2cbe-461d-ab23-102fa99bb92f)
```
selector=SelectKBest(score_func=mutual_info_classif, k=5)
x_new = selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/25a41bfd-12b8-47bc-8ba3-681e94cbf828)
```
selector=SelectKBest(score_func=mutual_info_classif, k=5)
x_new = selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/7b1139b1-b8ca-430e-b1b6-6d0b05bfca1e)
```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
x=pd.get_dummies(x)
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/c71d5e14-1795-4d6d-ac40-9f54fba843bd)
```
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importances=model.feature_importances_
threshold=0.1
selected_features = x.columns[feature_importances>threshold]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/d29ab376-a268-4296-b055-fc3ce8c9edf7)
```
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importances=model.feature_importances_
threshold=0.15
selected_features = x.columns[feature_importances>threshold]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/ca1ac194-652b-4d94-8e12-db38fb8d1eb9)

# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset successfully.
