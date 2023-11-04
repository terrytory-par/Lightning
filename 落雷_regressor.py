import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor



final= pd.read_csv('D:/mjbigdata/예측모델/낙뢰 예측/낙뢰data/낙뢰예측/5분단위/5MIN_JEONRA.CSV') # 파일 읽어오기 


final.rename(columns={"Unnamed: 0": "Date"}, inplace=True) # date column이 파일 불러올때 unnamed로 바뀜 

df = final



df.describe()



inpu = df.iloc[:,2:33] # input_variable

outpu = df[['count']] # target 값 

# window_size=2 로 설정 

future_prediction_hours = 2 
input1 = inpu.iloc[:-future_prediction_hours,:]

input2=  inpu.iloc[1:-future_prediction_hours+1,:]


for i in range(1,3):
    globals()['input{}'.format(i)].reset_index(inplace=True)
    del globals()['input{}'.format(i)]['index']

a = pd.DataFrame()
for i in range(1,3):
    a = pd.concat([a, globals()['input{}'.format(i)]],axis=1,ignore_index=False)

outpu = df[['count']]
output1 = outpu[2:]


output1.reset_index(inplace=True)
del output1['index']

total = pd.concat([a, output1], axis=1)

dataset= total


a = df.columns

a = a[2:]

list_1 = []
list_2 = []


for i in range(len(a)):
    b= a[i] + '_1'
    list_1.append(b)
    b= a[i] + '_2'
    list_2.append(b)


list_7 = ['count']

dataset.columns = list_1+list_2+list_7


dataset.describe()

dataset['count'].describe()

from sklearn.preprocessing import MinMaxScaler

def MinMaxScaling(df):    
    scaler = MinMaxScaler()

    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df.columns
    df_scaled.index = df.index

    return df_scaled

'''
from sklearn.preprocessing import RobustScaler
def RobusterScaling(df):    
    scaler = RobustScaler()

    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df.columns
    df_scaled.index = df.index

    return df_scaled


from sklearn.preprocessing import MaxAbsScaler
def MaxAbsScaling(df):    
    scaler = MaxAbsScaler()

    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df.columns
    df_scaled.index = df.index

    return df_scaled


from sklearn.preprocessing import Normalizer
def Normalizing(df):    
    scaler = Normalizer()

    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df.columns
    df_scaled.index = df.index

    return df_scaled
'''
dataset.columns


df2 = dataset.drop('count', axis=1)
df1 =  dataset[['count']]


data_scaled = MinMaxScaling(df2) # scaling
data_scaled.describe()


data_scaled['count'] = dataset['count'].values
data_scaled.describe()

# train/test split

df1= data_scaled[['count']]
df2= data_scaled.drop('count',axis=1)


split = int(len(data_scaled)*0.7)  # train test split 7:3

train = data_scaled[:split]
train.shape


X = df2.to_numpy()
y = df1.to_numpy()

df1 = df1.to_numpy()
df2 = df2.to_numpy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2, df1,test_size= 0.3,random_state=0)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


model=XGBRegressor(max_depth=10)

model.fit(X_train,y_train)

y_predict = model.predict(X_test)


print("training_model R2" , model.score(X_train,y_train)) # 모델의  R2
# print("test R2 : ", model.score(X_test,y_test))

print("preiction_value: " , int(y_predict[0])) # prediction 값 print


nana = pd.DataFrame(y_predict, columns=['y_predict'])
nana['test_label'] = y_test


import matplotlib.pyplot as plt
plt.plot(y_test, 'o',color='blue',alpha=.5, label='actual')
plt.plot(y_predict,'o',color='red' ,alpha=.5, label='prediction')
plt.xlabel('Times')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("y_test R2: " , r2_y_predict) # test와 predict 사이에 R2값

def adj_r2(r2, n, p):
    return 1 - (1-r2)*(n-1)/(n-p-1) # adjust R2

adj = adj_r2(r2_y_predict, X_test.shape[0], X_test.shape[1])
print("adj R2 : ",adj)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: " , RMSE(y_test,y_predict)) # rmse 출력 

import sklearn
from sklearn.metrics import mean_squared_error
mse = sklearn.metrics.mean_squared_error(y_test, y_predict)
print("MSE: ", mse) # mse 출력 

df2 = data_scaled
df2 = df2.drop('count',axis=1)

# 변수 중요도 출력 

def plot_feature_importances_cancer(model):
    plt.figure(figsize=(20,20))
    n_features = df2.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df2.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(model)
plt.show()
