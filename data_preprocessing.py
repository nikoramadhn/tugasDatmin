import numpy as np 
import matplotlib as plt 
import pandas as pd 

# dataset Sesuai 
dataset = pd.read_csv(r'F:\6\Data Mining 4607\Tugas\dataKasus.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X) 
print(y)

# //data set tampilan semua column
#data = pd.read_csv (r'F:\6\Data Mining 4607\Tugas\dataKasus.csv')   
#df = pd.DataFrame(data, columns= ['Country','Age','Salary','Purchased'])
#print (df)

# // Menghilangkan Missing Value (nan)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Encoding Data Kategori (atribut)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

#Encoding data kategori(class/label)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

#Membagi dataset ke dalam training set dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)
