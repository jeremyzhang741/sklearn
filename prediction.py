import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

price_pre = pd.read_csv('train.csv')
k=10

corrmatrix = price_pre.corr()


cols = corrmatrix.nlargest(k,'SalePrice')['SalePrice'].index

cm1 = price_pre[cols].corr()

hm2 = sns.heatmap(cm1,square=True,annot=True,cmap='RdPu',fmt='.2f',annot_kws={'size':10})


cols1 = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

total = price_pre.isnull().sum().sort_values(ascending=False)
percent = (price_pre.isnull().sum()/price_pre.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys = ['Total','Percent'])

data1 = price_pre.drop(missing_data[missing_data['Total']>1].index,axis=1)

data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)

data2.isnull().sum().max()

feature_data = data2.drop(['SalePrice'],axis=1)
target_data = data2['SalePrice']


X_train,X_test,y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)

model.fit(X_train,y_train)
