#!/usr/bin/env python
# coding: utf-8

# # 1.문제정의(목표설정)
# - 머신러닝 전체 과정 진행 후 , Kaggle에 주어진 과정 해결하자
# - 배송이 되어진 ID값을 도출해라?전자 상거래 물품 배송 ? 성공했는지 실패했는지?
# ### 결론: 물건이 제 시간에 배송이 되었는지 예측해보자!

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # 2.데이터 수집(데이터 로드)
# - kaggle 사이트에서 대회용 데이터를 로드

# In[43]:


train=pd.read_csv("./data/Doei/train.csv",index_col='ID')
test=pd.read_csv("./data/Doei/test.csv",index_col='ID')
train.isnull().sum()


# In[44]:


# 데이터 프레임 전체 행 보기
pd.set_option('display.max_rows',None)


# In[45]:


#학습데이터
train


# In[46]:


#테스트데이터
test


# In[47]:


print(train.shape)
print(test.shape)


# In[48]:


train.describe()
test.describe()


# In[49]:


train.info()
print('-'*50)
test.info()


# 이상치 먼저 우선순위로 처리
# ### 이상치가 있는 데이터[컬럼]
# - train: Mode_of_Shipment ,Product_importance
# - test : Mode_of_Shipment, Product_importance
# ### 결측치가 있는 데이터[컬럼]
# - train:Customer_care_calls  ,Prior_purchases , Discount_offered 
# - test:Customer_care_calls ,Prior_purchases  ,Discount_offered
# 

# In[50]:


#이상치 데이터 확인 - ?
train['Mode_of_Shipment'].unique()
train['Product_importance'].unique()
test['Mode_of_Shipment'].unique()
test['Product_importance'].unique()


# - ID: ID 고객 번호입니다.
# - 창고 블록: 회사에는 A,B,C,D,E와 같은 블록으로 나누어진 큰 창고가 있습니다.
# - 배송 모드:회사는 제품을 선박, 비행 및 도로와 같은 다양한 방법으로 배송합니다.
# - 고객 관리 전화: 발송물 조회를 위한 문의로 걸려온 전화 수.
# - 고객 등급: 그 회사는 모든 고객들로부터 등급을 매겼다. 1이 가장 낮음(최악), 5가 가장 높음(최고)입니다.
# - 제품 비용: 제품 비용(미국 달러)
# - 이전 구매: 이전 구입 횟수입니다.
# - 제품 중요도: 회사는 제품을 저, 중, 고 등 다양한 파라미터로 분류했습니다.
# - 성별: 남성과 여성.
# - 할인 혜택: 그 특정 제품에 대한 할인이 제공됩니다.
# - 가중치: 그것은 그램 단위의 무게이다.
# - 정시에 도달함: 이 값은 목표 변수입니다. 여기서 1 제품이 제시간에 도달하지 못했음을 나타내고 0은 제시간에 도달했음을 나타냅니다.

# - ID: ID Number of Customers.<br>
# - Warehouse block: The Company have big Warehouse which is divided in to block such as A,B,C,D,E.<br>
# - Mode of shipment:The Company Ships the products in multiple way such as Ship, Flight and Road.<br>
# - Customer care calls: The number of calls made from enquiry for enquiry of the shipment.<br>
# - Customer rating: The company has rated from every customer. 1 is the lowest (Worst), 5 is the highest (Best).<br>
# - Cost of the product: Cost of the Product in US Dollars.<br>
# - Prior purchases: The Number of Prior Purchase.<br>
# - Product importance: The company has categorized the product in the various parameter such as low, medium, high.<br>
# - Gender: Male and Female.<br>
# - Discount offered: Discount offered on that specific product.<br>
# - Weight in gms: It is the weight in grams.<br>
# - Reached on time: It is the target variable, where 1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time.<br>

# In[51]:


train.describe()


# # 3.데이터 전처리(이상치 , 결측치 처리하기)
# 이상치 먼저 우선순위로 처리
# ### 이상치가 있는 데이터[컬럼]
# - train: Cost_of_the_Product,Discount_offered,Customer_rating,Prior_purchases
# - test : 
# ### 결측치가 있는 데이터[컬럼]
# - train:Customer_care_calls  ,Prior_purchases , Discount_offered 
# - test:Customer_care_calls ,Prior_purchases  ,Discount_offered

# In[52]:


#결측치가 너무 많은 데이터는 우선 날림
#train.drop('Discount_offered', axis=1, inplace=True)
#test.drop('Discount_offered', axis=1, inplace=True)


# In[53]:


#상관관계를 확인해보자
#수치형 데이터만 서로의 관계도를 알수 있음.
#수치형 데이터중엔 물건이 제시간에 도착한 데이터 상관관계가 가장 높은 건
#Discount_offered 데이터 -  할인 혜택과 물건이 제시간에 도착한 것이 연관이 있다!?
train.corr()
#train['Discount_offered'].describe()
do_midian=train['Discount_offered'].median()
print(do_midian)
train['Discount_offered']=train['Discount_offered'].fillna(do_midian)


# In[54]:


#Mode_of_Shipment 결측치 채우기
train['Product_importance'].unique()
train['Product_importance'].value_counts()

train['Product_importance']=train['Product_importance'].replace('?','low')
test['Product_importance']=test['Product_importance'].replace('?','low')

train['Product_importance']=train['Product_importance'].replace('highh','high')
test['Product_importance']=test['Product_importance'].replace('highh','high')
train['Product_importance']=train['Product_importance'].replace('loww','low')
test['Product_importance']=test['Product_importance'].replace('loww','low')
train['Product_importance']=train['Product_importance'].replace('mediumm','medium')
test['Product_importance']=test['Product_importance'].replace('mediumm','medium')

train['Product_importance'].value_counts()
#train['Product_importance']


# In[55]:


#Mode_of_Shipment 결측치 채우기
train['Mode_of_Shipment'].unique()
train['Mode_of_Shipment'].value_counts()
train['Mode_of_Shipment']=train['Mode_of_Shipment'].str.strip()
test['Mode_of_Shipment']=test['Mode_of_Shipment'].str.strip()
train['Mode_of_Shipment']=train['Mode_of_Shipment'].replace('Shipzk','Ship')
test['Mode_of_Shipment']=test['Mode_of_Shipment'].replace('Shipzk','Ship')
train['Mode_of_Shipment']=train['Mode_of_Shipment'].replace('Shipzk','Ship')
test['Mode_of_Shipment']=test['Mode_of_Shipment'].replace('Shipzk','Ship')
train['Mode_of_Shipment']=train['Mode_of_Shipment'].replace('Flightzk','Flight')
test['Mode_of_Shipment']=test['Mode_of_Shipment'].replace('Flightzk','Flight')
train['Mode_of_Shipment']=train['Mode_of_Shipment'].replace('Roadzk','Road')
test['Mode_of_Shipment']=test['Mode_of_Shipment'].replace('Roadzk','Road')
train['Mode_of_Shipment']=train['Mode_of_Shipment'].replace('?','Ship')
test['Mode_of_Shipment']=test['Mode_of_Shipment'].replace('?','Ship')

train['Mode_of_Shipment'].value_counts()


# In[56]:


#Weight_in_gms 이상치 처리,결측치 처리
#Weight_in_gms 유니크의 평균값
#평균값 구하기 위해, 열을 문자열 데이터 -> 숫자형으로 변경해주기위해 따로 변수선언
train_weight_convert=train['Weight_in_gms'].replace('?','0');
test_weight_convert=test['Weight_in_gms'].replace('?','0');
#그 변수선언을 숫자형으로 변경
#train_weight_convert 
try_to_number=pd.to_numeric(train_weight_convert)
try_to_number2=pd.to_numeric(test_weight_convert)

#유니크값 기준 평균값을 구한 후 주석처리
#try_to_number.unique().mean() 3631
#print(str(int(try_to_number.unique().mean())))
#print(str(int(try_to_number2.unique().mean())))
#type(str(int(try_to_number.unique().mean())))
train['Weight_in_gms']=train['Weight_in_gms'].replace('?',str(int(try_to_number.unique().mean())));
test['Weight_in_gms']=test['Weight_in_gms'].replace('?',str(int(try_to_number.unique().mean())));


# In[57]:


#Cost_of_the_Product 이상치 처리해야할 컬럼 3개 
#평균값으로 넣는것으로 예외처리.
train[train['Cost_of_the_Product']==train['Cost_of_the_Product'].max()]
train['Cost_of_the_Product'].unique()[[train['Cost_of_the_Product'].unique() != 9999]].mean()


# In[58]:


#pd.set_option('display.max_rows',None)
#Discount_offered 이상치 처리해야할 컬럼 11개 
#train[train['Discount_offered']==train['Discount_offered'].max()]
#train['Discount_offered'].value_counts()
#train['Discount_offered'].unique()


# In[59]:


#Customer_rating 이상치 처리해야할 컬럼 2개 
#고객 등급은 최대 5 
#이상치 처리하는 방법 - 이상치많지않아 빈도수 가장 많은 3번으로 바꾸기
train[train['Customer_rating']==train['Customer_rating'].max()]
train['Customer_rating'].value_counts()


# In[60]:


#Prior_purchases 이상치 101개
train[train['Prior_purchases']==train['Prior_purchases'].max()]


# In[61]:


import seaborn as sb

plt.figure(figsize=(20,20))
sb.heatmap(train.corr(),annot=True)


# In[62]:


def fill_Customer_care_calls(data):
    if pd.isna(data['Customer_care_calls']):
        print(f"진행중{data['Cost_of_the_Product']} 의 Prior_purchases값{data['Prior_purchases']}")
        return pt1.loc[data['Prior_purchases'], data['Mode_of_Shipment']]
    else:
        return data['Customer_care_calls']


# In[63]:


pt1 = train.pivot_table(values='Customer_care_calls',
                       index=['Prior_purchases', 'Mode_of_Shipment'],
                       aggfunc='mean'
                       )
pt1


# In[64]:


#train['Customer_care_calls'] = train.apply(fill_Customer_care_calls, axis=1).astype('int64')


# # 데이터 전처리 ( 이상치 , 결측치 ) 처리
# 

# In[65]:


#train['Cost_of_the_Product'].mean()
#rain['Cost_of_the_Product']=train['Cost_of_the_Product'].fillna()
#제품 비용 이상치 처리 완료!
train['Cost_of_the_Product']=train['Cost_of_the_Product'].replace(9999,214)
test['Cost_of_the_Product']=test['Cost_of_the_Product'].replace(9999,214)

train['Cost_of_the_Product'].unique()
test['Cost_of_the_Product'].unique()


# In[66]:


#고객등급 이상치 처리 완료
train['Customer_rating']=train['Customer_rating'].replace(99,3)
test['Customer_rating']=test['Customer_rating'].replace(99,3)

train['Customer_rating'].unique()
test['Customer_rating'].unique()


# In[67]:


#train['Prior_purchases'] 처리하기위해선 상관관계가 가장높은 고객전화수를 결측치 채워야한다.
#그룹별로 보기위해서!

pd.reset_option('display')
#train['Prior_purchases'].value_counts()
#빈도가 가장 많은게 3.0
#train['Prior_purchases'].unique()
#950개 Nan 값
#train['Prior_purchases'][train['Prior_purchases'].isnull()]


# In[68]:


#train['Prior_purchases'] 처리하기위해선 상관관계가 가장높은 고객전화수를 결측치 채워야한다.
#일단 평균값인 4로 비어있는값을 채워넣음
train['Customer_care_calls']=train['Customer_care_calls'].fillna(4)
test['Customer_care_calls']=test['Customer_care_calls'].fillna(4)


# In[69]:


#상관관계를 통해 Prior_purchases 채워넣기 위해

train['Prior_purchases']=train['Prior_purchases'].fillna(3)
test['Prior_purchases']=test['Prior_purchases'].fillna(3)

train_deck = train[['Prior_purchases','Customer_care_calls']].groupby(['Prior_purchases']).count()
train_deck


# In[70]:


#분류에 따른 수치값 평균 함수 적용 시 가라데이터보다 6%나 감소시킴.
#train['Customer_care_calls'] = train.apply(fill_Customer_care_calls,axis=1).astype('int64')
#test['Customer_care_calls'] = test.apply(fill_Customer_care_calls,axis=1).astype('int64')


# In[71]:


train.info()
test.info()


# In[72]:


#plt.figure(figsize=(20,20))
#sb.heatmap(train.corr(),annot=True)


# In[73]:


#전처리 끝 원핫 인코딩 진행 하기.
train.drop("Gender",axis=1,inplace=True)
test.drop("Gender",axis=1,inplace=True)
train = pd.get_dummies(train, columns=['Warehouse_block ', 'Product_importance','Mode_of_Shipment','Customer_rating','Prior_purchases','Discount_offered'], drop_first=True)
test = pd.get_dummies(test, columns=['Warehouse_block ','Product_importance', 'Mode_of_Shipment','Customer_rating','Prior_purchases','Discount_offered'], drop_first=True)


# In[74]:


X=train.drop(['Reached.on.Time_Y.N'], axis=1)
y=train['Reached.on.Time_Y.N']


# In[75]:


from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score , plot_roc_curve, accuracy_score
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[76]:


#정확도에 따른 모델 선택하는 함수
Choice_Model_score = []
Choice_Model_string = []


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
param = {
    "C":[0.001,0.01,0.1,1,10,100,1000]
             }
rf= RandomForestClassifier()
ad = AdaBoostClassifier(base_estimator =rf)
dt = DecisionTreeClassifier()
kn = KNeighborsClassifier(n_neighbors=32)
lr = LogisticRegression(random_state= 43, solver='lbfgs', max_iter=1000)
gp = GaussianProcessClassifier(1.0 * RBF(1.0))
mlp = MLPClassifier(alpha=1, max_iter=1000)
gnb = GaussianNB()
svc = SVC(random_state = 43, C = 10, gamma = 0.1, kernel ='rbf')
gs = GridSearchCV(LogisticRegression(max_iter=15000,random_state=777,C=0.001), 
              param,cv=10)

Choice_Model_string.clear()
Choice_Model_score.clear()

models = [rf,ad, dt, kn, svc, mlp, lr, gnb,gs]
models_name=["랜덤포레스트","에이다부스트","분류트리",
             "최근접","비선형분류모델(svc)","다층 퍼셉트론(신경망)","비선형 이진 분류 모델",
             "나이브모델",'그리드써치']
cnt = 0;
#모델별 정확도?
for model in models:
    model.fit(X_train, y_train)
    pre = model.predict(X_test)
    scores = cross_val_score(model, X_test, y_test, cv=5).mean().round(3)
    #f1score = metrics.f1_score(y_test, y_pred).round(3)
    print(model, '\n', 'Accuracy:', scores, '\n')
    Choice_Model_string.append([models[cnt]])
    Choice_Model_score.append(scores)
    cnt += 1


# In[78]:


def show_ranking():    
    dic = {'model':models_name,'score':Choice_Model_score}
    model_result_pd=pd.DataFrame(dic)
    df_s =model_result_pd.sort_values(by=['score'],ascending=False)
    return df_s


# In[92]:


show_ranking()


# In[95]:


for model in models:
    pre = model.predict(test)
    #result = pd.read_csv("./data/Doei/sampleSubmission.csv")
    result['Reached.on.Time_Y.N'] = pre
    #result.to_csv('kn2021-12-27_2.csv', index =False)
    #final_result = pd.read_csv("./kn2021-12-27_2.csv")
    #final_result
    #최종결과값에 따른 1,0 갯수
    print(f"{final_result['Reached.on.Time_Y.N'].value_counts()}")


# In[88]:


#고객문의가 3일때와 , 각 고객


# In[ ]:




