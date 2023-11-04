
# -*- coding: utf-8 -*-


"""
기상 데이터의 기간은 2019년 1월 1일 ~ 2020년 12월 31일까지 입니다.
낙뢰데이터는 연도별 파일로 되어있으므로 2019년도, 2020년도 파일을 준비해주세요
"""



import pandas as pd
import numpy as np
import os


# 낙뢰데이터부터 전처리
# ---------------------------------------------------------------------

folders = os.listdir('d:/mjbigdata/예측모델/낙뢰 예측/낙뢰data/lightning data1') # 파일 경로 맞춰주세요
os.chdir("d:/mjbigdata/예측모델/낙뢰 예측/낙뢰data/lightning data1") # 맞춰주세요
df_all_years = pd.DataFrame()
for files in folders[:2]: 
    df= pd.read_csv(files, index_col=None, header=0)
    df_all_years = pd.concat([df_all_years, df]) # 낙뢰데이터 모든 기간 합쳐주세요 ex) 2019-2020

df_all = df_all_years.loc[:,['lgt_datetime','lgt_yy','lgt_mm','lgt_dd','lgt_hh', 'lgt_lat', 'lgt_lon','lgt_amp']] # 필요한 column 명만 추출


'''
서울 / 경기 

if_lat_range_yes =  (df_all['lgt_lat'] <= 37.91) & ( df_all['lgt_lat'] >= 37.25 )
if_lon_range_yes = (df_all['lgt_lon'] <= 127.07) & ( df_all['lgt_lon'] >= 126.44 )


충청

if_lat_range_yes =  (df_all['lgt_lat'] <= 36.64) & ( df_all['lgt_lat'] >= 36.0 )
if_lon_range_yes = (df_all['lgt_lon'] <= 128.33) & ( df_all['lgt_lon'] >= 127.37 )


경상
lat_min = min(36.6273,35.878,36.4351)
lat_max = max(36.6273,35.878,36.4351)

lon_min = min(128.1488,128.653,129.0401)
lon_max = max(128.1488,128.653,129.0401)


강원

if_lat_range_yes =  (df_all['lgt_lat'] <=38.2509) & ( df_all['lgt_lat'] >= 37.3375)
if_lon_range_yes = (df_all['lgt_lon'] <= 129.1243 ) & ( df_all['lgt_lon'] >= 127.7357)

전라

lat_min = min(36.0053,35.1729,35.3482,35.4213)
lat_max = max(36.0053,35.1729,35.3482,35.4213) 
lon_min = min(126.7614,126.8916,126.599,127.3965) 
lon_max = max(126.7614,126.8916,126.599,127.3965)

'''



# 현재 전라도 

lat_min = min(36.0053,35.1729,35.3482,35.4213)
lat_max = max(36.0053,35.1729,35.3482,35.4213) # 기상 관측 위경도 확인 후 그리드 생성 
lon_min = min(126.7614,126.8916,126.599,127.3965) # 위경도는 도별로 임의의 station을 선정하고 staion 간의 그리드를 생성
lon_max = max(126.7614,126.8916,126.599,127.3965)

if_lat_range_yes =  (df_all['lgt_lat'] <= lat_max) & ( df_all['lgt_lat'] >= lat_min )
if_lon_range_yes = (df_all['lgt_lon'] <= lon_max) & ( df_all['lgt_lon'] >= lon_min )

subset_df = df_all[if_lat_range_yes & if_lon_range_yes]

# 낙뢰데이터는 amp(세기)가 0 이어도 미세한 낙뢰라고 가정하여 존재하는 모든 데이터를 count =1 처리하였음

subset_df['count'] = subset_df['lgt_amp'].apply(lambda x: 1 if x >0  else 1) 

subset_df['lgt_datetime'] = pd.to_datetime(subset_df['lgt_datetime'])

subset_df = subset_df[['lgt_datetime', 'count']]

subset_df.set_index('lgt_datetime', inplace=True)

final= subset_df.resample('5T').sum() # 5분단위로 resample # 5분간 몇회나 치는지



# -----------------------------------------------------------------------------------------


# 기상청 데이터 전처리 시작
# 현재 전라도
station_list = [140,146,243,245,244,247,254,156,172,251]# 통합할 station_list 정해주기 / station은 기상청 asos 지점 번호/ 설명자료 확인 
station_list = sorted(station_list)
'''
Station_num 리스트 

서울/ 경기 : [99,98,108,116,112,119,201]

충청 : [131, 133, 135, 137, 226, 238, 273, 279]

경상 : [136, 137, 273, 276, 278, 279, 281]

전라 : [140,146,243,245,244,247,254,156,172,251]

강원 : [90,93,100,101,104,105,106,114,211,212,217]
'''


area_name = 'JEONRA' # 파일명과 연결 영어로
# area_name = 'SEOUL_GYEONGGI', 'CHUNGCHEONG','GYEONGSANG','JEONRA','GANGWON'




station_list = sorted(station_list)


# 지점별로 기간 통합하기 ; 2019-2020년 통합

for i in station_list:
    os.chdir(f"D:/mjbigdata/예측모델/낙뢰 예측/낙뢰data/asos 분단위/기상청 관측 데이터") # station 별로 폴더 생성한후 가져오기 
    globals()['station{}'.format(i)] = pd.read_csv(f'D:/mjbigdata/예측모델/낙뢰 예측/낙뢰data/asos 분단위/기상청 관측 데이터/meteo_observation_2019_2020_station{i}.csv')
    del globals()['station{}'.format(i)]['Unnamed: 0']

# 변수 복사

for i in station_list:
    globals()['s{}'.format(i)] = globals()['station{}'.format(i)] 


# 필요한 column들만 뽑아내서 새로운 이름 명명해주기
for i in station_list:
    globals()['s{}'.format(i)] = globals()['s{}'.format(i)][['DATE_TIME', 'TEMP', 'PRECIP_CUMUL', 'WIND_DIR', 'WIND_SPD','HUMID', 'HUMID']]
    globals()['s{}'.format(i)].columns = ['Date','기온','강수량','풍속','풍향','습도','기압']


# date 와 각 column 두개만 있는 dataframe 생성

def temperature_dataframe_processing(a):
    globals()['date{}'.format(a)] =globals()['s{}'.format(a)]['Date']
    globals()['station{}'.format(a)] = globals()['s{}'.format(a)].loc[:,['기온']] 
    globals()['station{}'.format(a)]= globals()['station{}'.format(a)].apply(pd.to_numeric, errors = 'coerce') 
    globals()['station{}'.format(a)].fillna(0, inplace=True)
    globals()['station{}'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['temp{}'.format(a)] = globals()['station{}'.format(a)][['Date','기온']]

def rn_dataframe_processing(a):
    globals()['date{}'.format(a)] =globals()['s{}'.format(a)]['Date']
    globals()['station{}'.format(a)] = globals()['s{}'.format(a)].loc[:,['강수량']]
    globals()['station{}'.format(a)]= globals()['station{}'.format(a)].apply(pd.to_numeric, errors = 'coerce') # 기록되지 않은 null값은 0으로 
    globals()['station{}'.format(a)].fillna(0, inplace=True)
    globals()['station{}'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['rn{}'.format(a)] = globals()['station{}'.format(a)][['Date','강수량']]
    
def ws_dataframe_processing(a):
    globals()['date{}'.format(a)] =globals()['s{}'.format(a)]['Date']
    globals()['station{}'.format(a)] = globals()['s{}'.format(a)].loc[:,['풍속']]
    globals()['station{}'.format(a)]= globals()['station{}'.format(a)].apply(pd.to_numeric, errors = 'coerce')
    globals()['station{}'.format(a)].fillna(0, inplace=True)
    globals()['station{}'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['ws{}'.format(a)] = globals()['station{}'.format(a)][['Date','풍속']]
    
def wd_dataframe_processing(a):
    globals()['date{}'.format(a)] =globals()['s{}'.format(a)]['Date']
    globals()['station{}'.format(a)] = globals()['s{}'.format(a)].loc[:,['풍향']]
    globals()['station{}'.format(a)]= globals()['station{}'.format(a)].apply(pd.to_numeric, errors = 'coerce')
    globals()['station{}'.format(a)].fillna(0, inplace=True)
    globals()['station{}'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['wd{}'.format(a)] = globals()['station{}'.format(a)][['Date','풍향']]
    
def hm_dataframe_processing(a):
    globals()['date{}'.format(a)] =globals()['s{}'.format(a)]['Date']
    globals()['station{}'.format(a)] = globals()['s{}'.format(a)].loc[:,['습도']]
    globals()['station{}'.format(a)]= globals()['station{}'.format(a)].apply(pd.to_numeric, errors = 'coerce')
    globals()['station{}'.format(a)].fillna(0, inplace=True)
    globals()['station{}'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['hm{}'.format(a)] = globals()['station{}'.format(a)][['Date','습도']]
    
def pa_dataframe_processing(a):
    globals()['date{}'.format(a)] =globals()['s{}'.format(a)]['Date']
    globals()['station{}'.format(a)] = globals()['s{}'.format(a)].loc[:,['기압']]
    globals()['station{}'.format(a)]= globals()['station{}'.format(a)].apply(pd.to_numeric, errors = 'coerce')
    globals()['station{}'.format(a)].fillna(0, inplace=True)
    globals()['station{}'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['pa{}'.format(a)] = globals()['station{}'.format(a)][['Date','기압']]
    
for a in station_list:
    temperature_dataframe_processing(a)
    
for a in station_list:
    rn_dataframe_processing(a)
    
for a in station_list:
    ws_dataframe_processing(a)
    
for a in station_list:
    wd_dataframe_processing(a)
    
for a in station_list:
    hm_dataframe_processing(a)
    
for a in station_list:
    pa_dataframe_processing(a)
    
for i in station_list:
    globals()['temp{}'.format(i)].reset_index(inplace=True) # 인덱스 sorting
    del globals()['temp{}'.format(i)]['index']
    
for i in station_list:
    globals()['rn{}'.format(i)].reset_index(inplace=True)
    del globals()['rn{}'.format(i)]['index']
    
for i in station_list:
    globals()['wd{}'.format(i)].reset_index(inplace=True)
    del globals()['wd{}'.format(i)]['index']
    
for i in station_list:
    globals()['ws{}'.format(i)].reset_index(inplace=True)
    del globals()['ws{}'.format(i)]['index']
    
for i in station_list:
    globals()['hm{}'.format(i)].reset_index(inplace=True)
    del globals()['hm{}'.format(i)]['index']
    

for i in station_list:
    globals()['pa{}'.format(i)].reset_index(inplace=True)
    del globals()['pa{}'.format(i)]['index']
    
# 중간중간 저장 필요

'''
for i in station_list:
    if not os.path.exists(f'temp{i}.csv'):
        globals()['temp{}'.format(i)].to_csv(f'temp{i}.csv', index=False, mode='w', encoding='utf-8-sig')
    else:  
        globals()['temp{}'.format(i)].to_csv(f'temp{i}.csv', mode ='a', header=False , encoding='utf-8-sig')
        
# 저장시 복사해서 변수명 바꾸기 
'''


# 5분단위 예측이므로 resample


for a in station_list:
    globals()['temp{}'.format(a)]['Date'] =  pd.to_datetime(globals()['temp{}'.format(a)]['Date'])
    globals()['temp{}'.format(a)] = globals()['temp{}'.format(a)].set_index('Date')
    globals()['temp{}'.format(a)] = globals()['temp{}'.format(a)].resample('5T').mean()
    globals()['temp{}'.format(a)].reset_index(inplace=True)
    
for a in station_list:
    globals()['rn{}'.format(a)]['Date'] =  pd.to_datetime(globals()['rn{}'.format(a)]['Date'])
    globals()['rn{}'.format(a)] = globals()['rn{}'.format(a)].set_index('Date')
    globals()['rn{}'.format(a)] = globals()['rn{}'.format(a)].resample('5T').mean()
    globals()['rn{}'.format(a)].reset_index(inplace=True)

for a in station_list:
    globals()['wd{}'.format(a)]['Date'] =  pd.to_datetime(globals()['wd{}'.format(a)]['Date'])
    globals()['wd{}'.format(a)] = globals()['wd{}'.format(a)].set_index('Date')
    globals()['wd{}'.format(a)] = globals()['wd{}'.format(a)].resample('5T').mean()
    globals()['wd{}'.format(a)].reset_index(inplace=True)

for a in station_list:
    globals()['ws{}'.format(a)]['Date'] =  pd.to_datetime(globals()['ws{}'.format(a)]['Date'])
    globals()['ws{}'.format(a)] = globals()['ws{}'.format(a)].set_index('Date')
    globals()['ws{}'.format(a)] = globals()['ws{}'.format(a)].resample('5T').mean()
    globals()['ws{}'.format(a)].reset_index(inplace=True)

for a in station_list:
    globals()['hm{}'.format(a)]['Date'] =  pd.to_datetime(globals()['hm{}'.format(a)]['Date'])
    globals()['hm{}'.format(a)] = globals()['hm{}'.format(a)].set_index('Date')
    globals()['hm{}'.format(a)] = globals()['hm{}'.format(a)].resample('5T').mean()
    globals()['hm{}'.format(a)].reset_index(inplace=True)

for a in station_list:
    globals()['pa{}'.format(a)]['Date'] =  pd.to_datetime(globals()['pa{}'.format(a)]['Date'])
    globals()['pa{}'.format(a)] = globals()['pa{}'.format(a)].set_index('Date')
    globals()['pa{}'.format(a)] = globals()['pa{}'.format(a)].resample('5T').mean()
    globals()['pa{}'.format(a)].reset_index(inplace=True)

for i in station_list:
    globals()['temp{}'.format(i)].reset_index(inplace=True)
    del globals()['temp{}'.format(i)]['index']

for i in station_list:
    globals()['rn{}'.format(i)].reset_index(inplace=True)
    del globals()['rn{}'.format(i)]['index']

for i in station_list:
    globals()['wd{}'.format(i)].reset_index(inplace=True)
    del globals()['wd{}'.format(i)]['index']

for i in station_list:
    globals()['ws{}'.format(i)].reset_index(inplace=True)
    del globals()['ws{}'.format(i)]['index']

for i in station_list:
    globals()['hm{}'.format(i)].reset_index(inplace=True)
    del globals()['hm{}'.format(i)]['index']

for i in station_list:
    globals()['pa{}'.format(i)].reset_index(inplace=True)
    del globals()['pa{}'.format(i)]['index']
    
# 변화율 추가

def change_temp(a):
    aa = globals()['temp{}'.format(a)][288:]
    aa.reset_index(inplace= True)
    del aa['index']
    
   
    fivemin_change = []
    for i in range(0,len(globals()['temp{}'.format(a)])-1):
        x= (globals()['temp{}'.format(a)]['기온'][i+1] - globals()['temp{}'.format(a)]['기온'][i])/ 5 # 5분 변화율 
        fivemin_change.append(x)
        
    
    
    onehour_change = []
    for i in range(0,len(globals()['temp{}'.format(a)])-12):
        x= (globals()['temp{}'.format(a)]['기온'][i+12] - globals()['temp{}'.format(a)]['기온'][i])/ 60 # 1시간 변화율 
        onehour_change.append(x)

        
    
    eighthour_change = []

    for i in range(0,len(globals()['temp{}'.format(a)])-96):
        x= (globals()['temp{}'.format(a)]['기온'][i+96] - globals()['temp{}'.format(a)]['기온'][i])/480 # 8시간 변화율 
        eighthour_change.append(x)

    
    twelvehour_change = []

    for i in range(0,len(globals()['temp{}'.format(a)])-144):
        x= (globals()['temp{}'.format(a)]['기온'][i+144] - globals()['temp{}'.format(a)]['기온'][i])/ 720 # 12시간 변화율
        twelvehour_change.append(x)
 
        
    oneday_change = [] 

    for i in range(0,len(globals()['temp{}'.format(a)])-288):
        x= (globals()['temp{}'.format(a)]['기온'][i+288] - globals()['temp{}'.format(a)]['기온'][i])/ 1440 # 하루기온 변화율 
        oneday_change.append(x)

        
    
    
    five = pd.DataFrame({'5분기온변화율': fivemin_change})
    hour1 = pd.DataFrame({'1시간기온변화율': onehour_change} )
    hour8 = pd.DataFrame({'8시간기온변화율': eighthour_change} )
    hour12= pd.DataFrame({'12시간기온변화율': twelvehour_change} )
    oneday = pd.DataFrame({'하루기온변화율': oneday_change})

    five = five[288-1:]
    five.reset_index(inplace=True)
    hour1 = hour1[288-12:]
    hour1.reset_index(inplace=True)
    hour8 = hour8[288-96:]
    hour8.reset_index(inplace=True)
    hour12 = hour12[288-144:]
    hour12.reset_index(inplace=True)
    oneday= oneday[288-288:]
    oneday.reset_index(inplace=True)
    
    del five['index']
    del hour1['index']
    del hour8['index']
    del hour12['index']
    del oneday['index']
    

    b = pd.merge(aa,five,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour1,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour8,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour12,how='outer', left_index=True, right_index=True)
    globals()['temp{}'.format(a)] = pd.merge(b, oneday, how= 'outer', left_index=True, right_index=True)

def change_ws(a):
    aa = globals()['ws{}'.format(a)][288:]
    aa.reset_index(inplace= True)
    del aa['index']
    
   
    fivemin_change = []
    for i in range(0,len(globals()['ws{}'.format(a)])-1):
        x= (globals()['ws{}'.format(a)]['풍속'][i+1] - globals()['ws{}'.format(a)]['풍속'][i])/ 5
        fivemin_change.append(x)
        
    
    
    onehour_change = []
    for i in range(0,len(globals()['ws{}'.format(a)])-12):
        x= (globals()['ws{}'.format(a)]['풍속'][i+12] - globals()['ws{}'.format(a)]['풍속'][i])/ 60
        onehour_change.append(x)

        
    
    eighthour_change = []

    for i in range(0,len(globals()['ws{}'.format(a)])-96):
        x= (globals()['ws{}'.format(a)]['풍속'][i+96] - globals()['ws{}'.format(a)]['풍속'][i])/480
        eighthour_change.append(x)

    
    twelvehour_change = []

    for i in range(0,len(globals()['ws{}'.format(a)])-144):
        x= (globals()['ws{}'.format(a)]['풍속'][i+144] - globals()['ws{}'.format(a)]['풍속'][i])/ 720
        twelvehour_change.append(x)
 
        
    oneday_change = [] 

    for i in range(0,len(globals()['ws{}'.format(a)])-288):
        x= (globals()['ws{}'.format(a)]['풍속'][i+288] - globals()['ws{}'.format(a)]['풍속'][i])/ 1440
        oneday_change.append(x)

        
    
    
    five = pd.DataFrame({'5분풍속변화율': fivemin_change})
    hour1 = pd.DataFrame({'1시간풍속변화율': onehour_change} )
    hour8 = pd.DataFrame({'8시간풍속변화율': eighthour_change} )
    hour12= pd.DataFrame({'12시간풍속변화율': twelvehour_change} )
    oneday = pd.DataFrame({'하루풍속변화율': oneday_change})

    five = five[288-1:]
    five.reset_index(inplace=True)
    hour1 = hour1[288-12:]
    hour1.reset_index(inplace=True)
    hour8 = hour8[288-96:]
    hour8.reset_index(inplace=True)
    hour12 = hour12[288-144:]
    hour12.reset_index(inplace=True)
    oneday= oneday[288-288:]
    oneday.reset_index(inplace=True)
    
    del five['index']
    del hour1['index']
    del hour8['index']
    del hour12['index']
    del oneday['index']
    

    b = pd.merge(aa,five,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour1,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour8,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour12,how='outer', left_index=True, right_index=True)
    globals()['ws{}'.format(a)] = pd.merge(b, oneday, how= 'outer', left_index=True, right_index=True)

def change_wd(a):
    aa = globals()['wd{}'.format(a)][288:]
    aa.reset_index(inplace= True)
    del aa['index']
    
   
    fivemin_change = []
    for i in range(0,len(globals()['wd{}'.format(a)])-1):
        x= (globals()['wd{}'.format(a)]['풍향'][i+1] - globals()['wd{}'.format(a)]['풍향'][i])/ 5
        fivemin_change.append(x)
        
    
    
    onehour_change = []
    for i in range(0,len(globals()['wd{}'.format(a)])-12):
        x= (globals()['wd{}'.format(a)]['풍향'][i+12] - globals()['wd{}'.format(a)]['풍향'][i])/ 60
        onehour_change.append(x)

        
    
    eighthour_change = []

    for i in range(0,len(globals()['wd{}'.format(a)])-96):
        x= (globals()['wd{}'.format(a)]['풍향'][i+96] - globals()['wd{}'.format(a)]['풍향'][i])/480
        eighthour_change.append(x)

    
    twelvehour_change = []

    for i in range(0,len(globals()['wd{}'.format(a)])-144):
        x= (globals()['wd{}'.format(a)]['풍향'][i+144] - globals()['wd{}'.format(a)]['풍향'][i])/ 720
        twelvehour_change.append(x)
 
        
    oneday_change = [] 

    for i in range(0,len(globals()['wd{}'.format(a)])-288):
        x= (globals()['wd{}'.format(a)]['풍향'][i+288] - globals()['wd{}'.format(a)]['풍향'][i])/ 1440
        oneday_change.append(x)

        
    
    
    five = pd.DataFrame({'5분풍향변화율': fivemin_change})
    hour1 = pd.DataFrame({'1시간풍향변화율': onehour_change} )
    hour8 = pd.DataFrame({'8시간풍향변화율': eighthour_change} )
    hour12= pd.DataFrame({'12시간풍향변화율': twelvehour_change} )
    oneday = pd.DataFrame({'하루풍향변화율': oneday_change})

    five = five[288-1:]
    five.reset_index(inplace=True)
    hour1 = hour1[288-12:]
    hour1.reset_index(inplace=True)
    hour8 = hour8[288-96:]
    hour8.reset_index(inplace=True)
    hour12 = hour12[288-144:]
    hour12.reset_index(inplace=True)
    oneday= oneday[288-288:]
    oneday.reset_index(inplace=True)
    
    del five['index']
    del hour1['index']
    del hour8['index']
    del hour12['index']
    del oneday['index']
    

    b = pd.merge(aa,five,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour1,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour8,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour12,how='outer', left_index=True, right_index=True)
    globals()['wd{}'.format(a)] = pd.merge(b, oneday, how= 'outer', left_index=True, right_index=True)

def change_hm(a):
    aa = globals()['hm{}'.format(a)][288:]
    aa.reset_index(inplace= True)
    del aa['index']
    
   
    fivemin_change = []
    for i in range(0,len(globals()['hm{}'.format(a)])-1):
        x= (globals()['hm{}'.format(a)]['습도'][i+1] - globals()['hm{}'.format(a)]['습도'][i])/ 5
        fivemin_change.append(x)
        
    
    
    onehour_change = []
    for i in range(0,len(globals()['hm{}'.format(a)])-12):
        x= (globals()['hm{}'.format(a)]['습도'][i+12] - globals()['hm{}'.format(a)]['습도'][i])/ 60
        onehour_change.append(x)

        
    
    eighthour_change = []

    for i in range(0,len(globals()['hm{}'.format(a)])-96):
        x= (globals()['hm{}'.format(a)]['습도'][i+96] - globals()['hm{}'.format(a)]['습도'][i])/480
        eighthour_change.append(x)

    
    twelvehour_change = []

    for i in range(0,len(globals()['hm{}'.format(a)])-144):
        x= (globals()['hm{}'.format(a)]['습도'][i+144] - globals()['hm{}'.format(a)]['습도'][i])/ 720
        twelvehour_change.append(x)
 
        
    oneday_change = [] 

    for i in range(0,len(globals()['hm{}'.format(a)])-288):
        x= (globals()['hm{}'.format(a)]['습도'][i+288] - globals()['hm{}'.format(a)]['습도'][i])/ 1440
        oneday_change.append(x)

        
    
    
    five = pd.DataFrame({'5분습도변화율': fivemin_change})
    hour1 = pd.DataFrame({'1시간습도변화율': onehour_change} )
    hour8 = pd.DataFrame({'8시간습도변화율': eighthour_change} )
    hour12= pd.DataFrame({'12시간습도변화율': twelvehour_change} )
    oneday = pd.DataFrame({'하루습도변화율': oneday_change})

    five = five[288-1:]
    five.reset_index(inplace=True)
    hour1 = hour1[288-12:]
    hour1.reset_index(inplace=True)
    hour8 = hour8[288-96:]
    hour8.reset_index(inplace=True)
    hour12 = hour12[288-144:]
    hour12.reset_index(inplace=True)
    oneday= oneday[288-288:]
    oneday.reset_index(inplace=True)
    
    del five['index']
    del hour1['index']
    del hour8['index']
    del hour12['index']
    del oneday['index']
    

    b = pd.merge(aa,five,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour1,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour8,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour12,how='outer', left_index=True, right_index=True)
    globals()['hm{}'.format(a)] = pd.merge(b, oneday, how= 'outer', left_index=True, right_index=True)

def change_pa(a):
    aa = globals()['pa{}'.format(a)][288:]
    aa.reset_index(inplace= True)
    del aa['index']
    
   
    fivemin_change = []
    for i in range(0,len(globals()['pa{}'.format(a)])-1):
        x= (globals()['pa{}'.format(a)]['기압'][i+1] - globals()['pa{}'.format(a)]['기압'][i])/ 5
        fivemin_change.append(x)
        
    
    
    onehour_change = []
    for i in range(0,len(globals()['pa{}'.format(a)])-12):
        x= (globals()['pa{}'.format(a)]['기압'][i+12] - globals()['pa{}'.format(a)]['기압'][i])/ 60
        onehour_change.append(x)

        
    
    eighthour_change = []

    for i in range(0,len(globals()['pa{}'.format(a)])-96):
        x= (globals()['pa{}'.format(a)]['기압'][i+96] - globals()['pa{}'.format(a)]['기압'][i])/480
        eighthour_change.append(x)

    
    twelvehour_change = []

    for i in range(0,len(globals()['pa{}'.format(a)])-144):
        x= (globals()['pa{}'.format(a)]['기압'][i+144] - globals()['pa{}'.format(a)]['기압'][i])/ 720
        twelvehour_change.append(x)
 
        
    oneday_change = [] 

    for i in range(0,len(globals()['pa{}'.format(a)])-288):
        x= (globals()['pa{}'.format(a)]['기압'][i+288] - globals()['pa{}'.format(a)]['기압'][i])/ 1440
        oneday_change.append(x)

        
    
    
    five = pd.DataFrame({'5분기압변화율': fivemin_change})
    hour1 = pd.DataFrame({'1시간기압변화율': onehour_change} )
    hour8 = pd.DataFrame({'8시간기압변화율': eighthour_change} )
    hour12= pd.DataFrame({'12시간기압변화율': twelvehour_change} )
    oneday = pd.DataFrame({'하루기압변화율': oneday_change})

    five = five[288-1:]
    five.reset_index(inplace=True)
    hour1 = hour1[288-12:]
    hour1.reset_index(inplace=True)
    hour8 = hour8[288-96:]
    hour8.reset_index(inplace=True)
    hour12 = hour12[288-144:]
    hour12.reset_index(inplace=True)
    oneday= oneday[288-288:]
    oneday.reset_index(inplace=True)
    
    del five['index']
    del hour1['index']
    del hour8['index']
    del hour12['index']
    del oneday['index']
    

    b = pd.merge(aa,five,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour1,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour8,how='outer', left_index=True, right_index=True)
    b = pd.merge(b, hour12,how='outer', left_index=True, right_index=True)
    globals()['pa{}'.format(a)] = pd.merge(b, oneday, how= 'outer', left_index=True, right_index=True)

for a in station_list:
    change_temp(a)

for a in station_list:
    change_wd(a)

for a in station_list:
    change_ws(a)

for a in station_list:
    change_pa(a)

for a in station_list:
    change_hm(a)

    
# 각자 변화율 추출 - 나중에 각각 merge해서 평균 해야하므로 따로 빼내기 


def temperature_dataframe_processing1(a):
    globals()['date{}'.format(a)] =globals()['temp{}'.format(a)]['Date']
    globals()['temp{}_1'.format(a)] = globals()['temp{}'.format(a)].loc[:,['기온']]
    globals()['temp{}_1'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['temp{}_1'.format(a)] = globals()['temp{}_1'.format(a)][['Date','기온']]

def temperature_dataframe_processing2(a):
    globals()['date{}'.format(a)] =globals()['temp{}'.format(a)]['Date']
    globals()['temp{}_2'.format(a)] = globals()['temp{}'.format(a)].loc[:,['5분기온변화율']]
    globals()['temp{}_2'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['temp{}_2'.format(a)] = globals()['temp{}_2'.format(a)][['Date','5분기온변화율']]

def temperature_dataframe_processing3(a):
    globals()['date{}'.format(a)] =globals()['temp{}'.format(a)]['Date']
    globals()['temp{}_3'.format(a)] = globals()['temp{}'.format(a)].loc[:,['1시간기온변화율']]
    globals()['temp{}_3'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['temp{}_3'.format(a)] = globals()['temp{}_3'.format(a)][['Date','1시간기온변화율']]

def temperature_dataframe_processing4(a):
    globals()['date{}'.format(a)] =globals()['temp{}'.format(a)]['Date']
    globals()['temp{}_4'.format(a)] = globals()['temp{}'.format(a)].loc[:,['8시간기온변화율']]
    globals()['temp{}_4'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['temp{}_4'.format(a)] = globals()['temp{}_4'.format(a)][['Date','8시간기온변화율']]

def temperature_dataframe_processing5(a):
    globals()['date{}'.format(a)] =globals()['temp{}'.format(a)]['Date']
    globals()['temp{}_5'.format(a)] = globals()['temp{}'.format(a)].loc[:,['12시간기온변화율']]
    globals()['temp{}_5'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['temp{}_5'.format(a)] = globals()['temp{}_5'.format(a)][['Date','12시간기온변화율']]

def temperature_dataframe_processing6(a):
    globals()['date{}'.format(a)] =globals()['temp{}'.format(a)]['Date']
    globals()['temp{}_6'.format(a)] = globals()['temp{}'.format(a)].loc[:,['하루기온변화율']]
    globals()['temp{}_6'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['temp{}_6'.format(a)] = globals()['temp{}_6'.format(a)][['Date','하루기온변화율']]

for a in station_list:
    temperature_dataframe_processing1(a)

for a in station_list:
    temperature_dataframe_processing2(a)

for a in station_list:
    temperature_dataframe_processing3(a)

for a in station_list:
    temperature_dataframe_processing4(a)

for a in station_list:
    temperature_dataframe_processing5(a)

for a in station_list:
    temperature_dataframe_processing6(a)
    
    
def ws_dataframe_processing1(a):
    globals()['date{}'.format(a)] =globals()['ws{}'.format(a)]['Date']
    globals()['ws{}_1'.format(a)] = globals()['ws{}'.format(a)].loc[:,['풍속']]
    globals()['ws{}_1'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['ws{}_1'.format(a)] = globals()['ws{}_1'.format(a)][['Date','풍속']]

def ws_dataframe_processing2(a):
    globals()['date{}'.format(a)] =globals()['ws{}'.format(a)]['Date']
    globals()['ws{}_2'.format(a)] = globals()['ws{}'.format(a)].loc[:,['5분풍속변화율']]
    globals()['ws{}_2'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['ws{}_2'.format(a)] = globals()['ws{}_2'.format(a)][['Date','5분풍속변화율']]

def ws_dataframe_processing3(a):
    globals()['date{}'.format(a)] =globals()['ws{}'.format(a)]['Date']
    globals()['ws{}_3'.format(a)] = globals()['ws{}'.format(a)].loc[:,['1시간풍속변화율']]
    globals()['ws{}_3'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['ws{}_3'.format(a)] = globals()['ws{}_3'.format(a)][['Date','1시간풍속변화율']]

def ws_dataframe_processing4(a):
    globals()['date{}'.format(a)] =globals()['ws{}'.format(a)]['Date']
    globals()['ws{}_4'.format(a)] = globals()['ws{}'.format(a)].loc[:,['8시간풍속변화율']]
    globals()['ws{}_4'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['ws{}_4'.format(a)] = globals()['ws{}_4'.format(a)][['Date','8시간풍속변화율']]

def ws_dataframe_processing5(a):
    globals()['date{}'.format(a)] =globals()['ws{}'.format(a)]['Date']
    globals()['ws{}_5'.format(a)] = globals()['ws{}'.format(a)].loc[:,['12시간풍속변화율']]
    globals()['ws{}_5'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['ws{}_5'.format(a)] = globals()['ws{}_5'.format(a)][['Date','12시간풍속변화율']]

def ws_dataframe_processing6(a):
    globals()['date{}'.format(a)] =globals()['ws{}'.format(a)]['Date']
    globals()['ws{}_6'.format(a)] = globals()['ws{}'.format(a)].loc[:,['하루풍속변화율']]
    globals()['ws{}_6'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['ws{}_6'.format(a)] = globals()['ws{}_6'.format(a)][['Date','하루풍속변화율']]

for a in station_list:
    ws_dataframe_processing1(a)


for a in station_list:
    ws_dataframe_processing2(a)

for a in station_list:
    ws_dataframe_processing3(a)

for a in station_list:
    ws_dataframe_processing4(a)

for a in station_list:
    ws_dataframe_processing5(a)

for a in station_list:
    ws_dataframe_processing6(a)   
    
    
    
# 풍향파트

def wd_dataframe_processing1(a):
    globals()['date{}'.format(a)] =globals()['wd{}'.format(a)]['Date']
    globals()['wd{}_1'.format(a)] = globals()['wd{}'.format(a)].loc[:,['풍향']]
    globals()['wd{}_1'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['wd{}_1'.format(a)] = globals()['wd{}_1'.format(a)][['Date','풍향']]

def wd_dataframe_processing2(a):
    globals()['date{}'.format(a)] =globals()['wd{}'.format(a)]['Date']
    globals()['wd{}_2'.format(a)] = globals()['wd{}'.format(a)].loc[:,['5분풍향변화율']]
    globals()['wd{}_2'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['wd{}_2'.format(a)] = globals()['wd{}_2'.format(a)][['Date','5분풍향변화율']]

def wd_dataframe_processing3(a):
    globals()['date{}'.format(a)] =globals()['wd{}'.format(a)]['Date']
    globals()['wd{}_3'.format(a)] = globals()['wd{}'.format(a)].loc[:,['1시간풍향변화율']]
    globals()['wd{}_3'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['wd{}_3'.format(a)] = globals()['wd{}_3'.format(a)][['Date','1시간풍향변화율']]

def wd_dataframe_processing4(a):
    globals()['date{}'.format(a)] =globals()['wd{}'.format(a)]['Date']
    globals()['wd{}_4'.format(a)] = globals()['wd{}'.format(a)].loc[:,['8시간풍향변화율']]
    globals()['wd{}_4'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['wd{}_4'.format(a)] = globals()['wd{}_4'.format(a)][['Date','8시간풍향변화율']]

def wd_dataframe_processing5(a):
    globals()['date{}'.format(a)] =globals()['wd{}'.format(a)]['Date']
    globals()['wd{}_5'.format(a)] = globals()['wd{}'.format(a)].loc[:,['12시간풍향변화율']]
    globals()['wd{}_5'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['wd{}_5'.format(a)] = globals()['wd{}_5'.format(a)][['Date','12시간풍향변화율']]

def wd_dataframe_processing6(a):
    globals()['date{}'.format(a)] =globals()['wd{}'.format(a)]['Date']
    globals()['wd{}_6'.format(a)] = globals()['wd{}'.format(a)].loc[:,['하루풍향변화율']]
    globals()['wd{}_6'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['wd{}_6'.format(a)] = globals()['wd{}_6'.format(a)][['Date','하루풍향변화율']]

for a in station_list:
    wd_dataframe_processing1(a)

for a in station_list:
    wd_dataframe_processing2(a)

for a in station_list:
    wd_dataframe_processing3(a)

for a in station_list:
    wd_dataframe_processing4(a)

for a in station_list:
    wd_dataframe_processing5(a)

for a in station_list:
    wd_dataframe_processing6(a)

# 습도 파트

def hm_dataframe_processing1(a):
    globals()['date{}'.format(a)] =globals()['hm{}'.format(a)]['Date']
    globals()['hm{}_1'.format(a)] = globals()['hm{}'.format(a)].loc[:,['습도']]
    globals()['hm{}_1'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['hm{}_1'.format(a)] = globals()['hm{}_1'.format(a)][['Date','습도']]

def hm_dataframe_processing2(a):
    globals()['date{}'.format(a)] =globals()['hm{}'.format(a)]['Date']
    globals()['hm{}_2'.format(a)] = globals()['hm{}'.format(a)].loc[:,['5분습도변화율']]
    globals()['hm{}_2'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['hm{}_2'.format(a)] = globals()['hm{}_2'.format(a)][['Date','5분습도변화율']]

def hm_dataframe_processing3(a):
    globals()['date{}'.format(a)] =globals()['hm{}'.format(a)]['Date']
    globals()['hm{}_3'.format(a)] = globals()['hm{}'.format(a)].loc[:,['1시간습도변화율']]
    globals()['hm{}_3'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['hm{}_3'.format(a)] = globals()['hm{}_3'.format(a)][['Date','1시간습도변화율']]

def hm_dataframe_processing4(a):
    globals()['date{}'.format(a)] =globals()['hm{}'.format(a)]['Date']
    globals()['hm{}_4'.format(a)] = globals()['hm{}'.format(a)].loc[:,['8시간습도변화율']]
    globals()['hm{}_4'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['hm{}_4'.format(a)] = globals()['hm{}_4'.format(a)][['Date','8시간습도변화율']]

def hm_dataframe_processing5(a):
    globals()['date{}'.format(a)] =globals()['hm{}'.format(a)]['Date']
    globals()['hm{}_5'.format(a)] = globals()['hm{}'.format(a)].loc[:,['12시간습도변화율']]
    globals()['hm{}_5'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['hm{}_5'.format(a)] = globals()['hm{}_5'.format(a)][['Date','12시간습도변화율']]

def hm_dataframe_processing6(a):
    globals()['date{}'.format(a)] =globals()['hm{}'.format(a)]['Date']
    globals()['hm{}_6'.format(a)] = globals()['hm{}'.format(a)].loc[:,['하루습도변화율']]
    globals()['hm{}_6'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['hm{}_6'.format(a)] = globals()['hm{}_6'.format(a)][['Date','하루습도변화율']]

for a in station_list:
    hm_dataframe_processing1(a)

for a in station_list:
    hm_dataframe_processing2(a)

for a in station_list:
    hm_dataframe_processing3(a)

for a in station_list:
    hm_dataframe_processing4(a)

for a in station_list:
    hm_dataframe_processing5(a)

for a in station_list:
    hm_dataframe_processing6(a)

# 기압파트

def pa_dataframe_processing1(a):
    globals()['date{}'.format(a)] =globals()['pa{}'.format(a)]['Date']
    globals()['pa{}_1'.format(a)] = globals()['pa{}'.format(a)].loc[:,['기압']]
    globals()['pa{}_1'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['pa{}_1'.format(a)] = globals()['pa{}_1'.format(a)][['Date','기압']]

def pa_dataframe_processing2(a):
    globals()['date{}'.format(a)] =globals()['pa{}'.format(a)]['Date']
    globals()['pa{}_2'.format(a)] = globals()['pa{}'.format(a)].loc[:,['5분기압변화율']]
    globals()['pa{}_2'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['pa{}_2'.format(a)] = globals()['pa{}_2'.format(a)][['Date','5분기압변화율']]

def pa_dataframe_processing3(a):
    globals()['date{}'.format(a)] =globals()['pa{}'.format(a)]['Date']
    globals()['pa{}_3'.format(a)] = globals()['pa{}'.format(a)].loc[:,['1시간기압변화율']]
    globals()['pa{}_3'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['pa{}_3'.format(a)] = globals()['pa{}_3'.format(a)][['Date','1시간기압변화율']]

def pa_dataframe_processing4(a):
    globals()['date{}'.format(a)] =globals()['pa{}'.format(a)]['Date']
    globals()['pa{}_4'.format(a)] = globals()['pa{}'.format(a)].loc[:,['8시간기압변화율']]
    globals()['pa{}_4'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['pa{}_4'.format(a)] = globals()['pa{}_4'.format(a)][['Date','8시간기압변화율']]

def pa_dataframe_processing5(a):
    globals()['date{}'.format(a)] =globals()['pa{}'.format(a)]['Date']
    globals()['pa{}_5'.format(a)] = globals()['pa{}'.format(a)].loc[:,['12시간기압변화율']]
    globals()['pa{}_5'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['pa{}_5'.format(a)] = globals()['pa{}_5'.format(a)][['Date','12시간기압변화율']]

def pa_dataframe_processing6(a):
    globals()['date{}'.format(a)] =globals()['pa{}'.format(a)]['Date']
    globals()['pa{}_6'.format(a)] = globals()['pa{}'.format(a)].loc[:,['하루기압변화율']]
    globals()['pa{}_6'.format(a)]['Date'] = globals()['date{}'.format(a)]
    globals()['pa{}_6'.format(a)] = globals()['pa{}_6'.format(a)][['Date','하루기압변화율']]

for a in station_list:
    pa_dataframe_processing1(a)

for a in station_list:
    pa_dataframe_processing2(a)

for a in station_list:
    pa_dataframe_processing3(a)

for a in station_list:
    pa_dataframe_processing4(a)

for a in station_list:
    pa_dataframe_processing5(a)

for a in station_list:
    pa_dataframe_processing6(a)
    
    
# 합치기
    

# 기온

temp_1 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]}) #  기간 변경시 맞춰주는것 필수
temp_2 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
temp_3 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
temp_4 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
temp_5 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
temp_6 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})

for a in station_list:
    temp_1 = pd.merge(temp_1,globals()['temp{}_1'.format(a)], how='outer', on ='Date')

for a in station_list:
    temp_2 = pd.merge(temp_2,globals()['temp{}_2'.format(a)], how='outer', on ='Date')

for a in station_list:
    temp_3 = pd.merge(temp_3,globals()['temp{}_3'.format(a)], how='outer', on ='Date')

for a in station_list:
    temp_4 = pd.merge(temp_4,globals()['temp{}_4'.format(a)], how='outer', on ='Date')

for a in station_list:
    temp_5 = pd.merge(temp_5,globals()['temp{}_5'.format(a)], how='outer', on ='Date')

for a in station_list:
    temp_6 = pd.merge(temp_6,globals()['temp{}_6'.format(a)], how='outer', on ='Date')

# 강수량 
rn = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})

for a in station_list:
    rn = pd.merge(rn,globals()['rn{}'.format(a)], how='outer', on ='Date')
rn

# 풍속

ws_1 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
ws_2 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
ws_3 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
ws_4 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
ws_5 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
ws_6 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})


for a in station_list:
    ws_1 = pd.merge(ws_1,globals()['ws{}_1'.format(a)], how='outer', on ='Date')

for a in station_list:
    ws_2 = pd.merge(ws_2,globals()['ws{}_2'.format(a)], how='outer', on ='Date')

for a in station_list:
    ws_3 = pd.merge(ws_3,globals()['ws{}_3'.format(a)], how='outer', on ='Date')

for a in station_list:
    ws_4 = pd.merge(ws_4,globals()['ws{}_4'.format(a)], how='outer', on ='Date')

for a in station_list:
    ws_5 = pd.merge(ws_5,globals()['ws{}_5'.format(a)], how='outer', on ='Date')

for a in station_list:
    ws_6 = pd.merge(ws_6,globals()['ws{}_6'.format(a)], how='outer', on ='Date')

# 풍향 
wd_1 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
wd_2 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
wd_3 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
wd_4 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
wd_5 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
wd_6 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})



for a in station_list:
    wd_1 = pd.merge(wd_1,globals()['wd{}_1'.format(a)], how='outer', on ='Date')


for a in station_list:
    wd_2 = pd.merge(wd_2,globals()['wd{}_2'.format(a)], how='outer', on ='Date')

for a in station_list:
    wd_3 = pd.merge(wd_3,globals()['wd{}_3'.format(a)], how='outer', on ='Date')

for a in station_list:
    wd_4= pd.merge(wd_4,globals()['wd{}_4'.format(a)], how='outer', on ='Date')

for a in station_list:
    wd_5 = pd.merge(wd_5,globals()['wd{}_5'.format(a)], how='outer', on ='Date')

for a in station_list:
    wd_6 = pd.merge(wd_6,globals()['wd{}_6'.format(a)], how='outer', on ='Date')

# 습도

hm_1 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
hm_2 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
hm_3 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
hm_4 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
hm_5 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
hm_6 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})



for a in station_list:
    hm_1 = pd.merge(hm_1,globals()['hm{}_1'.format(a)], how='outer', on ='Date')

for a in station_list:
    hm_2 = pd.merge(hm_2,globals()['hm{}_2'.format(a)], how='outer', on ='Date')

for a in station_list:
    hm_3 = pd.merge(hm_3,globals()['hm{}_3'.format(a)], how='outer', on ='Date')

for a in station_list:
    hm_4 = pd.merge(hm_4,globals()['hm{}_4'.format(a)], how='outer', on ='Date')

for a in station_list:
    hm_5 = pd.merge(hm_5,globals()['hm{}_5'.format(a)], how='outer', on ='Date')

for a in station_list:
    hm_6 = pd.merge(hm_6,globals()['hm{}_6'.format(a)], how='outer', on ='Date')

# 기압

pa_1 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
pa_2 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
pa_3 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
pa_4 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
pa_5 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})
pa_6 = pd.DataFrame({"Date": [pd.Timestamp('201901020000')]})


for a in station_list:
    pa_1 = pd.merge(pa_1,globals()['pa{}_1'.format(a)], how='outer', on ='Date')

for a in station_list:
    pa_2 = pd.merge(pa_2,globals()['pa{}_2'.format(a)], how='outer', on ='Date')

for a in station_list:
    pa_3 = pd.merge(pa_3,globals()['pa{}_3'.format(a)], how='outer', on ='Date')

for a in station_list:
    pa_4 = pd.merge(pa_4,globals()['pa{}_4'.format(a)], how='outer', on ='Date')

for a in station_list:
    pa_5 = pd.merge(pa_5,globals()['pa{}_5'.format(a)], how='outer', on ='Date')

for a in station_list:
    pa_6 = pd.merge(pa_6,globals()['pa{}_6'.format(a)], how='outer', on ='Date')

temp_1 = temp_1.sort_values(by='Date')
temp_2 = temp_2.sort_values(by='Date')
temp_3 = temp_3.sort_values(by='Date')
temp_4 = temp_4.sort_values(by='Date')
temp_5 = temp_5.sort_values(by='Date')
temp_6 = temp_6.sort_values(by='Date')





rn = rn.sort_values(by='Date')

ws_1 = ws_1.sort_values(by='Date')
ws_2 = ws_2.sort_values(by='Date')
ws_3 = ws_3.sort_values(by='Date')
ws_4 = ws_4.sort_values(by='Date')
ws_5 = ws_5.sort_values(by='Date')
ws_6 = ws_6.sort_values(by='Date')


wd_1 = wd_1.sort_values(by='Date')
wd_2 = wd_2.sort_values(by='Date')
wd_3 = wd_3.sort_values(by='Date')
wd_4 = wd_4.sort_values(by='Date')
wd_5 = wd_5.sort_values(by='Date')
wd_6 = wd_6.sort_values(by='Date')




hm_1 = hm_1.sort_values(by='Date')
hm_2 = hm_2.sort_values(by='Date')
hm_3 = hm_3.sort_values(by='Date')
hm_4 = hm_4.sort_values(by='Date')
hm_5 = hm_5.sort_values(by='Date')
hm_6 = hm_6.sort_values(by='Date')


pa_1 = pa_1.sort_values(by='Date')
pa_2 = pa_2.sort_values(by='Date')
pa_3 = pa_3.sort_values(by='Date')
pa_4 = pa_4.sort_values(by='Date')
pa_5 = pa_5.sort_values(by='Date')
pa_6 = pa_6.sort_values(by='Date')



########################################################

# 중복 데이터 혹시 있으면 제거 

temp_1.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
temp_2.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
temp_3.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
temp_4.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
temp_5.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
temp_6.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)



rn.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)

ws_1.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
ws_2.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
ws_3.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
ws_4.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
ws_5.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
ws_6.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)



wd_1.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
wd_2.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
wd_3.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
wd_4.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
wd_5.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
wd_6.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)




hm_1.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
hm_2.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
hm_3.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
hm_4.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
hm_5.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
hm_6.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)


pa_1.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
pa_2.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
pa_3.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
pa_4.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
pa_5.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)
pa_6.drop_duplicates(subset='Date', keep='first', inplace=True, ignore_index=False)



# mean , skipna=True

temp_1= temp_1.iloc[:,1:]
temp_mean1 = temp_1.mean(axis=1, skipna=True) # null값은 제외하고 평균 내기 
temp_mean1

temp_2= temp_2.iloc[:,1:]
temp_mean2 = temp_2.mean(axis=1, skipna=True)
temp_mean2

temp_3= temp_3.iloc[:,1:]
temp_mean3 = temp_3.mean(axis=1, skipna=True)
temp_mean3

temp_4= temp_4.iloc[:,1:]
temp_mean4 = temp_4.mean(axis=1, skipna=True)
temp_mean4

temp_5= temp_5.iloc[:,1:]
temp_mean5 = temp_5.mean(axis=1, skipna=True)
temp_mean5

temp_6= temp_6.iloc[:,1:]
temp_mean6 = temp_6.mean(axis=1, skipna=True)
temp_mean6

rn= rn.iloc[:,1:]
rn_mean = rn.mean(axis=1, skipna=True)

# diff () 사용 
# 누적강수량을 분단위 강수량으로
a= rn_mean.diff().fillna(rn_mean)

a

df1= pd.DataFrame({'rn':a})

a = df1['rn'] <0
a= df1[a]
a


df1[(df1['rn'] <0 )] = 0 # 날짜가 변하는 경계에 있을경우 마이너스가 나올수 있으므로 이 경우 0으로 치환 

df1

rn_mean= df1['rn'].values

rn_mean = rn_mean[288:]

rn_mean


ws_11= ws_1.iloc[:,1:]
ws_mean1 = ws_11.mean(axis=1, skipna=True)
ws_mean1

ws_22= ws_2.iloc[:,1:]
ws_mean2 = ws_22.mean(axis=1, skipna=True)
ws_mean2

ws_33= ws_3.iloc[:,1:]
ws_mean3 = ws_33.mean(axis=1, skipna=True)
ws_mean3

ws_44= ws_4.iloc[:,1:]
ws_mean4 = ws_44.mean(axis=1, skipna=True)
ws_mean4

ws_55= ws_5.iloc[:,1:]
ws_mean5 = ws_55.mean(axis=1, skipna=True)
ws_mean5

ws_66= ws_6.iloc[:,1:]
ws_mean6 = ws_66.mean(axis=1, skipna=True)
ws_mean6



wd_11= wd_1.iloc[:,1:]
wd_mean1 = wd_11.mean(axis=1, skipna=True)
wd_mean1

wd_22= wd_2.iloc[:,1:]
wd_mean2 = wd_22.mean(axis=1, skipna=True)
wd_mean2

wd_33= wd_3.iloc[:,1:]
wd_mean3 = wd_33.mean(axis=1, skipna=True)
wd_mean3

wd_44= wd_4.iloc[:,1:]
wd_mean4 = wd_44.mean(axis=1, skipna=True)
wd_mean4

wd_55= wd_5.iloc[:,1:]
wd_mean5 = wd_55.mean(axis=1, skipna=True)
wd_mean5

wd_66= wd_6.iloc[:,1:]
wd_mean6 = wd_66.mean(axis=1, skipna=True)
wd_mean6



hm_11= hm_1.iloc[:,1:]
hm_mean1 = hm_11.mean(axis=1, skipna=True)
hm_mean1

hm_22= hm_2.iloc[:,1:]
hm_mean2 = hm_22.mean(axis=1, skipna=True)
hm_mean2

hm_33= hm_3.iloc[:,1:]
hm_mean3 = hm_33.mean(axis=1, skipna=True)
hm_mean3

hm_44= hm_4.iloc[:,1:]
hm_mean4 = hm_44.mean(axis=1, skipna=True)
hm_mean4

hm_55= hm_5.iloc[:,1:]
hm_mean5 = hm_55.mean(axis=1, skipna=True)
hm_mean5

hm_66= hm_6.iloc[:,1:]
hm_mean6 = hm_66.mean(axis=1, skipna=True)
hm_mean6



pa_11= pa_1.iloc[:,1:]
pa_mean1 = pa_11.mean(axis=1, skipna=True)
pa_mean1

pa_22= pa_2.iloc[:,1:]
pa_mean2 = pa_22.mean(axis=1, skipna=True)
pa_mean2

pa_33= pa_3.iloc[:,1:]
pa_mean3 = pa_33.mean(axis=1, skipna=True)
pa_mean3

pa_44= pa_4.iloc[:,1:]
pa_mean4 = pa_44.mean(axis=1, skipna=True)
pa_mean4

pa_55= pa_5.iloc[:,1:]
pa_mean5 = pa_55.mean(axis=1, skipna=True)
pa_mean5

pa_66= pa_6.iloc[:,1:]
pa_mean6 = pa_66.mean(axis=1, skipna=True)
pa_mean6

date = pa_5['Date']

date.value_counts()

asos_total= pd.DataFrame({ 'Date': date, 
                          'temp_mean': temp_mean1,
                          'temp_change_5min':temp_mean2,
                          'temp_change_1h':temp_mean3,
                          'temp_change_8h':temp_mean4 ,
                          'temp_change_12h':temp_mean5,
                          'temp_change_oneday':temp_mean6,
    
                          
                          'rn_mean':rn_mean,
                          
                          'ws_mean': ws_mean1,
                          'ws_change_5min':ws_mean2,
                          'ws_change_1h': ws_mean3, 
                          'ws_change_8h':ws_mean4, 
                          'ws_change_12h': ws_mean5, 
                          'ws_change_oneday':ws_mean6,
                          
                          
                          'wd_mean':wd_mean1, 
                          'wd_change_5min':wd_mean2, 
                          'wd_change_1h': wd_mean3,
                          'wd_change_8h':wd_mean4, 
                          'wd_change_12h': wd_mean5, 
                          'wd_change_oneday':wd_mean6,

                          
                          'humid_mean':hm_mean1, 
                          'humid_change_5min':hm_mean2,
                          'humid_change_1h': hm_mean3, 
                          'humid_change_8h':hm_mean4, 
                          'humid_change_12h': hm_mean5,
                          'humid_change_oneday':hm_mean6,

                          
                          'pa_mean':pa_mean1, 
                          'pa_change_5min':pa_mean2,
                          'pa_change_1h': pa_mean3,
                          'pa_change_8h':pa_mean4, 
                          'pa_change_12h': pa_mean5,
                          'pa_change_oneday': pa_mean6

                         
                         
                         })

asos_total

# 이제 asos 랑 낙뢰데이터 merge 전 dataframe 전처리

total = asos_total

total['Date'] = pd.to_datetime(total['Date'])

total.set_index('Date',inplace=True)



minutes = pd.merge(left= final, right = total, how='outer', left_index=True, right_index=True)


minutes.fillna(0, inplace=True) #count 값 null은 0 처리 


minutes.to_csv(f'D:/5MIN_{area_name}.csv') # 파일경로, 파일명  바꿔주기 

minutes



