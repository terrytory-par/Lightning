import pandas as pd
import numpy as np

final= pd.read_csv('D:/mjbigdata/예측모델/낙뢰 예측/낙뢰data/낙뢰예측/5분단위/5min_JEONRA.CSV') # 파일 경로 맞춰주기


final.rename(columns={"Unnamed: 0": "Date"}, inplace=True) # 불러들여오면 date column 명이 unnamed로 나옴 

df = final

df.columns

df.describe()

df['count'] = df['count'].apply(lambda x: 1 if x >0  else 0) # classification을 위해 낙뢰 횟수가 1 이상이면 1, 아니면 0으로 데이터 변환 


inpu = df.iloc[:,2:33]

outpu = df[['count']]


future_prediction_hours = 2  # window size 를 과거 2개의 row 를 가지고 예측
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


from sklearn.preprocessing import MinMaxScaler # 데이터의 범위가 다양하므로 0과 1사이로 scaling 

def MinMaxScaling(df):    
    scaler = MinMaxScaler()

    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df.columns
    df_scaled.index = df.index

    return df_scaled
'''
from sklearn.preprocessing import Normalizer
def Normalizing(df):    
    scaler = Normalizer()

    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df.columns
    df_scaled.index = df.index

    return df_scaled

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
'''

df2 = dataset.drop('count', axis=1)
df1 = dataset[['count']]

data_scaled = MinMaxScaling(df2)
data_scaled.describe()

data_scaled['count'] = dataset['count'].values
data_scaled.describe()

# train/test split

df1= data_scaled[['count']]
df2= data_scaled.drop('count',axis=1)


split = int(len(data_scaled)*0.7) # train test split 7:3

# k-fold - train 과 validation set를 나눌때 사용하는게 가장 효과가 큼

train = data_scaled[:split]
train.shape


df1= data_scaled[['count']]
df2= data_scaled.drop('count',axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2, df1,test_size= 0.3,random_state=0)

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# 혼동행렬, 정확도, 정밀도, 재현율, F1, AUC 불러오기
def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))

import matplotlib.pyplot as plt


# K-FOLD

cv = KFold(n_splits=10, random_state=0, shuffle=True)

train = data_scaled[:split]
train.shape

train.iloc[:,0:62]

model=XGBClassifier( max_depth=10, objective='multi:softmax', num_class=2 , use_label_encoder=False,eval_metric='mlogloss')

cnt_iter = 0
accuracy_list = []
precision_list = []
recall_list = []
F1_list = []
AUC_list = []

# k fold로 train, val n_split 만큼 돌리기 

for tidx, vidx  in cv.split(train):
    print('predict with KFold')
    train_cv = train.iloc[tidx] 
    val_cv = train.iloc[vidx]
    train_X = train_cv.iloc[:,0:62]
    train_Y = train_cv.iloc[:,-1:] 
    val_X = val_cv.iloc[:,0:62]
    val_Y = val_cv.iloc[:,-1:] 
    

# 각각의 데이터로 xgboost모델로 학습.
    model.fit(train_X, train_Y, eval_set=[(val_X,val_Y)]) # train data를 다시 train - valid data로 나눔 

# 예측
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    precision = precision_score(y_test, y_pred)
    precision_list.append(precision)
    recall = recall_score(y_test, y_pred)
    recall_list.append(recall)
    F1 = f1_score(y_test, y_pred)
    F1_list.append(F1)
    AUC = roc_auc_score(y_test, y_pred)
    AUC_list.append(AUC)
    
    get_clf_eval(y_test, y_pred)
    cnt_iter += 1
    print(cnt_iter,'회 완료')

print('\n## 교차 검증 별 정확도: ' ,np.round(accuracy_list,4))
print('## 평균 검증 정확도:{:.4f}'.format(np.mean(accuracy_list)))
print('## 평균 검증 정밀도: {:.4f}'.format(np.mean(precision_list)))
print('## 평균 검증 재현율: {:.4f}'.format(np.mean(recall_list)))
print('## 평균 검증 F1: {:.4f}'.format(np.mean(F1_list)))
print('## 평균 검증 AUC:{:.4f}'.format(np.mean(AUC_list)))

# 시간이 다소 걸림 


df2 = dataset.drop('count', axis=1)


# feature importance 출력 

def plot_feature_importances_cancer(model): 
    plt.figure(figsize=(100,100))
    n_features = df2.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df2.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(model)
plt.show()