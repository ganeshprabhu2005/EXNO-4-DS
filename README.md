# EXNO:4-DS
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
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
       import pandas as pd
import numpy as np
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
<img width="550" height="220" alt="image" src="https://github.com/user-attachments/assets/65ae3e37-05de-4743-bb76-7f5f806500ee" />
data.isnull().sum()
<img width="289" height="637" alt="image" src="https://github.com/user-attachments/assets/5ab268ab-2052-4d7c-9ac8-722ad8dd851a" />
missing=data[data.isnull().any(axis=1)]
missing
<img width="564" height="234" alt="image" src="https://github.com/user-attachments/assets/48fdfc28-8a1b-481b-9765-5f1cec6bcd9c" />
data2=data.dropna(axis=0)
data2
<img width="536" height="220" alt="image" src="https://github.com/user-attachments/assets/4de817ec-bcad-49a5-89d1-6f3e09e8e178" />
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
<img width="430" height="257" alt="image" src="https://github.com/user-attachments/assets/0bdc8d05-da4e-4e4e-9907-4a442ef64cb0" />
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
<img width="455" height="585" alt="image" src="https://github.com/user-attachments/assets/2c1609f2-d11a-438e-bdca-af2956347be9" />
data2
<img width="535" height="201" alt="image" src="https://github.com/user-attachments/assets/e325b7b6-8c28-4cef-a7b6-1ade46364125" />
new_data=pd.get_dummies(data2, drop_first=True)
new_data
<img width="537" height="189" alt="image" src="https://github.com/user-attachments/assets/d361fea0-583b-4904-80a0-c66e0b12f34e" />
columns_list=list(new_data.columns)
print(columns_list)
<img width="540" height="20" alt="image" src="https://github.com/user-attachments/assets/8dac0a43-a2c8-47df-893f-9a24271750c1" />

features=list(set(columns_list)-set(['SalStat']))
print(features)
<img width="579" height="7" alt="image" src="https://github.com/user-attachments/assets/aab7bfa0-8082-4297-a2d6-483b3e34c63c" />
y=new_data['SalStat'].values
print(y)
<img width="274" height="29" alt="image" src="https://github.com/user-attachments/assets/03709560-7332-4af9-9a6d-acf61079fd52" />
x=new_data[features].values
print(x)
<img width="316" height="130" alt="image" src="https://github.com/user-attachments/assets/e5b94aea-963e-4183-ae76-2311250f5b0d" />
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
<img width="331" height="96" alt="image" src="https://github.com/user-attachments/assets/a1a05ffe-e08b-42f3-a5d3-74bf8e430dfd" />
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
<img width="312" height="11" alt="image" src="https://github.com/user-attachments/assets/d9aef582-a3dd-4910-8f88-324fac3126e7" />
print("Misclassified Samples : %d" % (test_y !=prediction).sum())

<img width="400" height="50" alt="image" src="https://github.com/user-attachments/assets/bb0bf271-1579-49f7-a515-6e78a2eed4dc" />
data.shape
<img width="233" height="34" alt="image" src="https://github.com/user-attachments/assets/11291867-3999-4fa2-a77b-f6b9d6c2a2da" />

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
<img width="395" height="249" alt="image" src="https://github.com/user-attachments/assets/f221c20e-71c3-4e33-a35f-cc62c6008186" />
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
<img width="544" height="254" alt="image" src="https://github.com/user-attachments/assets/b561abb1-c105-4abe-ae4b-17b4f6f7f80b" />
tips.time.unique()
<img width="513" height="95" alt="image" src="https://github.com/user-attachments/assets/df6ca9e2-93b6-487b-887b-e0a54ed90f7b" />
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

<img width="288" height="116" alt="image" src="https://github.com/user-attachments/assets/05533459-28d7-4735-a1a4-3a69ebac3bcb" />
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
<img width="420" height="64" alt="image" src="https://github.com/user-attachments/assets/a0cf7c8c-3405-477d-9932-d21c106d75af" />




































       
# RESULT:
       # INCLUDE YOUR RESULT HERE
        Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
        
