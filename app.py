import numpy as np
import pandas as pd


df = pd.read_csv('placement.csv')
# print(df)



# print(df.isnull().sum())



# from sklearn.impute import SimpleImputer
# si = SimpleImputer()
# df


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()


x= df.drop(columns=['placed'])
y= df['placed']
from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test = train_test_split(x,y , test_size = 0.2 , random_state= 42)


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()

rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)


from sklearn.metrics import accuracy_score
print("preformace:----",accuracy_score(y_test, y_pred))