import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#数据集有3类商品

sale_data = pd.read_csv('BlackFriday.csv')

set_list = list(set(sale_data['Product_ID']))

num = len(set_list)

sale_data.fillna(0,inplace=True)

le_U_ID = LabelEncoder()
sale_data['User_ID'] = le_U_ID.fit_transform(sale_data['User_ID'])
le_P_ID = LabelEncoder()
sale_data['Product_ID'] = le_P_ID.fit_transform(sale_data['Product_ID'])
sale_data['Gender'] = np.where(sale_data['Gender']=='M',1,0)
df_Age = pd.get_dummies(sale_data.Age)
df_CC = pd.get_dummies(sale_data.City_Category)
df_SIC = pd.get_dummies(sale_data.Stay_In_Current_City_Years)
df_ocup = pd.get_dummies(sale_data.Occupation)
df_encoded = pd.concat([sale_data,df_Age,df_CC,df_SIC],axis=1)
df_encoded = pd.concat([df_encoded,df_ocup],axis=1)
df_encoded.drop(['Occupation'],axis=1,inplace=True)
df_encoded.drop(['Age','City_Category','Stay_In_Current_City_Years'],axis=1,inplace=True)
df_frac = df_encoded.sample(frac=0.02,random_state=100)
df_frac = df_encoded.sample(frac=0.02,random_state=100)
X = df_frac.drop(['Purchase','User_ID','Product_ID','Product_Category_1','Product_Category_2','Product_Category_3'], axis=1)
y = df_frac['Purchase']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=100)
param_grid = {'n_estimators':[1,3,10,30,100,150,300],'max_depth':[1,3,5,7,9]}
grid_rf = GridSearchCV(RandomForestRegressor(),param_grid,cv=3,scoring='neg_mean_squared_error').fit(X_train,y_train)
print('Best parameter: {}'.format(grid_rf.best_params_))
print('Best score: {:.2f}'.format((-1*grid_rf.best_score_)**0.5))