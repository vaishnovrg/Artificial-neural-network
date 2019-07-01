import pandas as pd
import numpy as np

dataset=pd.read_excel('Data_Train.xlsx')

dataset=dataset.drop(columns=['Name','Location','New_Price'])

dataset.isnull().sum()
dataset.mean()
dataset['Seats'].fillna(5,inplace=True)
dataset['Mileage'].fillna('0 kmpl',inplace=True)
dataset['Engine'].fillna('0 CC',inplace=True)
dataset['Power'].fillna('0 bhp',inplace=True)

indexNames = dataset[ dataset['Fuel_Type'] == 'Electric' ].index
dataset.drop(indexNames,inplace=True)

dataset['Owner_Type'].replace('Fourth & Above','Fourth',inplace=True)

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,[-1]].values

#Z1=pd.DataFrame(X)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])
X[:,3]=le.fit_transform(X[:,3])
X[:,4]=le.fit_transform(X[:,4])
X[:,5]=le.fit_transform(X[:,5])
X[:,6]=le.fit_transform(X[:,6])
X[:,7]=le.fit_transform(X[:,7])
oe=OneHotEncoder(categorical_features=[2,4])
X=oe.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
Y=sc.fit_transform(Y)

import numpy.core.umath

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the second hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, Y, batch_size = 10, epochs = 100)

#----------------------------------------------------------------------------------------------------


dataset1=pd.read_excel('Data_Test.xlsx')

dataset1=dataset1.drop(columns=['Name','Location','New_Price'])

dataset1.isnull().sum()
dataset1.mean()
dataset1['Seats'].fillna(5,inplace=True)
dataset1['Mileage'].fillna('0 kmpl',inplace=True)
dataset1['Engine'].fillna('0 CC',inplace=True)
dataset1['Power'].fillna('0 bhp',inplace=True)

dataset1['Owner_Type'].replace('Fourth & Above','Fourth',inplace=True)

X_test=dataset1.iloc[:,:].values
#Z2=pd.DataFrame(X_test)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X_test[:,2]=le.fit_transform(X_test[:,2])
X_test[:,3]=le.fit_transform(X_test[:,3])
X_test[:,4]=le.fit_transform(X_test[:,4])
X_test[:,5]=le.fit_transform(X_test[:,5])
X_test[:,6]=le.fit_transform(X_test[:,6])
X_test[:,7]=le.fit_transform(X_test[:,7])
oe1=OneHotEncoder(categorical_features=[2,4])
X_test=oe1.fit_transform(X_test).toarray()


X_test=sc.fit_transform(X_test)

#y_pred=y_pred.reshape(1, -1)
#y_pred_new=sc.inverse_transform(y_pred)
y_pred = classifier.predict(X_test)

prediction = pd.DataFrame(y_pred, columns=['Price']).to_excel('prediction.xlsx')



