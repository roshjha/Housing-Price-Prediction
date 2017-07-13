

# @author: roshan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

train=pd.read_csv('HPtrain.csv')
test=pd.read_csv('HPtest.csv')

print "Training data size" , train.shape
print "Testing data size" , test.shape

#print train.SalePrice.describe()

target=np.log(train.SalePrice)
'''print "Skew is" , target.skew()
plot.hist(target, color='blue')
plot.show()'''

numeric_features=train.select_dtypes(include=[np.number])
#print numeric_features.dtypes

correlation=numeric_features.corr()

print correlation['SalePrice'].sort_values(ascending=False)[:5], "\n"
print correlation['SalePrice'].sort_values(ascending=False)[-5:], "\n"
'''
quality_pivot=train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

quality_pivot.plot(kind='bar', color='blue')
plot.xlabel('Overall Quality')
plot.ylabel('Median Sale Price')
plot.xticks(rotation=0)
plot.show()

plot.scatter(x=train['GrLivArea'], y=target)
plot.ylabel('Sale Price')
plot.xlabel('Ground Living Area')
plot.show()'''

train=train[train['GarageArea']<1200] #done to remove outliers for this.
'''plot.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plot.xlim(-200, 1600)
plot.ylabel('Sale Price')
plot.xlabel('Garage Area')
plot.show()

'''
non_numeric_features=train.select_dtypes(exclude=[np.number])
print non_numeric_features.dtypes
#print non_numeric_features.describe()

''' Transform Categorical attirbutes to numeric using One hot encoding'''


train['enc_street']=pd.get_dummies(train.Street, drop_first=True)
test['enc_street']=pd.get_dummies(test.Street, drop_first=True)

train['enc_Alley']=pd.get_dummies(train.Alley, drop_first=True)
test['enc_Alley']=pd.get_dummies(test.Alley, drop_first=True)

train['enc_Alley']=pd.get_dummies(train.Alley, drop_first=True)
test['enc_Alley']=pd.get_dummies(test.Alley, drop_first=True)

def encodeKitchenQual(x):
	if x=='Ex':
		return 5
	elif x=='Gd':
		return 4
	else:
		return 1

train['KitchenQual_enc']=train.KitchenQual.apply(encodeKitchenQual)
test['KitchenQual_enc']=test.KitchenQual.apply(encodeKitchenQual)

def encodeSaleCond(x): 
	if x=='Partial':
          return 3
        else:
          return 0

train['SaleCondition_enc']=train.SaleCondition.apply(encodeSaleCond)
test['SaleCondition_enc']=test.SaleCondition.apply(encodeSaleCond)


def encodeFoundation(x): 
	if x=='PConc':
          return 2
        elif x=='Wood':
          return 1
        else:
          return 0

train['Foundation_enc']=train.Foundation.apply(encodeFoundation)
test['Foundation_enc']=test.Foundation.apply(encodeFoundation)

def encodeCentralAir(x):
	if(x=='Y'):
		return 3
	else:
		return 1

train['CentralAir_enc']=train.CentralAir.apply(encodeCentralAir)
test['CentralAir_enc']=test.CentralAir.apply(encodeCentralAir)


def encodeHS(x):
	if(x=='2Story' or x=='2.5Fin'):
		return 3
	else:
		return 1

train['HS_enc']=train.HouseStyle.apply(encodeHS)
test['HS_enc']=test.HouseStyle.apply(encodeHS)

def encodeExC(x):
	if(x=='Ex'):
		return 5
	else:
		return 1

train['ExC_enc']=train.ExterCond.apply(encodeExC)
test['ExC_enc']=test.ExterCond.apply(encodeExC)

'''
Sale_pivot=train.pivot_table(index='SaleCondition_enc', values='SalePrice', aggfunc=np.median)
Sale_pivot.plot(kind='bar', color='blue')
plot.xlabel('Sale Condition Encoded')
plot.ylabel('Median Sale Price')
plot.xticks(rotation=0)
plot.show()'''

data=train.select_dtypes(include=[np.number]).interpolate().dropna()
#check if all data points are not null
print sum(data.isnull().sum()!=0)

'''Building Model'''

y=np.log(train.SalePrice)
X=data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42, test_size=.33)


from sklearn import linear_model


print "Using Linear Regression"
lR=linear_model.LinearRegression()
model=lR.fit(X_train, y_train)
print "Accuracy is ", model.score(X_test, y_test)


print "Using Ridge"
rR=linear_model.Ridge(alpha=10**2)
model=rR.fit(X_train, y_train)
print "Accuracy is ", model.score(X_test, y_test)


condition_pivot = train.pivot_table(index='ExterQual',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plot.xlabel('HouseStyle')
plot.ylabel('Median Sale Price')
plot.xticks(rotation=0)
plot.show()