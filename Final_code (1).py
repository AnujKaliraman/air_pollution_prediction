import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
import numpy as np
#from sklearn import svm
from sklearn.linear_model import LinearRegression

data_file = "C:/Users/anuj/Desktop/project EBI/us_pollution_Texas.xlsx"

data=pd.read_excel(data_file)

data[data["State"] == "Houston"]

data=data.drop(["S.No", "State Code", "State", "County", "City", "Date Local"], axis=1)
data_scatterplt= data.drop(["County Code", "Site Num", "NO2 1st Max Hour", "NO2 1st Max Hour", "SO2 1st Max Hour", "CO 1st Max Hour", "O3 1st Max Hour"], axis=1)
cols = data_scatterplt.columns

print(data)
print(data_scatterplt)

attributes = list(data.columns)
attributes_scat = list(data_scatterplt.columns)

print(attributes)
print(attributes_scat)


#correlation matrix
data_no2= data_scatterplt.drop(["O3 Mean", "O3 1st Max Value", "O3 AQI", "SO2 Mean", "SO2 1st Max Value", "SO2 AQI", "CO Mean", "CO 1st Max Value", "CO AQI"], axis=1)
data_o3 = data_scatterplt.drop(["NO2 Mean", "NO2 1st Max Value", "NO2 AQI", "SO2 Mean", "SO2 1st Max Value", "SO2 AQI", "CO Mean", "CO 1st Max Value", "CO AQI"], axis=1)
data_so2 = data_scatterplt.drop(["O3 Mean", "O3 1st Max Value", "O3 AQI", "NO2 Mean", "NO2 1st Max Value", "NO2 AQI", "CO Mean", "CO 1st Max Value", "CO AQI"], axis=1)
data_co = data_scatterplt.drop(["O3 Mean", "O3 1st Max Value", "O3 AQI", "NO2 Mean", "NO2 1st Max Value", "NO2 AQI", "SO2 Mean", "SO2 1st Max Value", "SO2 AQI"], axis=1)

cols_no2 = data_no2.columns
cols_o3 = data_o3.columns
cols_so2 = data_so2.columns
cols_co = data_co.columns

#show scatter plot correlation matrix
#NO2
sns.set(style='whitegrid', context='notebook')
sns.pairplot(data_no2, height=2.5)
plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI/figures/scatter_no2.png', dpi=300)
plt.show()

cm = np.corrcoef(data_no2.values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, 
            cbar=True,
            annot=True, 
            square=True,
            fmt='.2f',
            annot_kws={'size': 15},
            yticklabels=cols_no2.values,
            xticklabels=cols_no2.values)

plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI//figures/corr_mat_no2.png', dpi=300)
plt.show()

#O3
sns.set(style='whitegrid', context='notebook')
sns.pairplot(data_o3, height=2.5)
plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI/figures/scatter_o3.png', dpi=300)
plt.show()

cm = np.corrcoef(data_o3.values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, 
            cbar=True,
            annot=True, 
            square=True,
            fmt='.2f',
            annot_kws={'size': 15},
            yticklabels=cols_o3.values,
            xticklabels=cols_o3.values)

plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI//figures/corr_mat_o3.png', dpi=300)
plt.show()

#SO2

sns.set(style='whitegrid', context='notebook')
sns.pairplot(data_so2, height=2.5)
plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI/figures/scatter_so2.png', dpi=300)
plt.show()

cm = np.corrcoef(data_so2.values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, 
            cbar=True,
            annot=True, 
            square=True,
            fmt='.2f',
            annot_kws={'size': 15},
            yticklabels=cols_so2.values,
            xticklabels=cols_so2.values)

plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI//figures/corr_mat_so2.png', dpi=300)
plt.show()

#CO

sns.set(style='whitegrid', context='notebook')
sns.pairplot(data_co, height=2.5)
plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI/figures/scatter_co.png', dpi=300)
plt.show()

cm = np.corrcoef(data_co.values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, 
            cbar=True,
            annot=True, 
            square=True,
            fmt='.2f',
            annot_kws={'size': 15},
            yticklabels=cols_co.values,
            xticklabels=cols_co.values)

plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI//figures/corr_mat_co.png', dpi=300)
plt.show()



#no2
#Decision Tree Regression 
attributes_no2 = list(data_no2.columns)

x = data_no2[attributes_no2].values

y = data_no2['NO2 AQI'].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5, random_state=0)

print(x_train.shape, y_train.shape)

reg_tree = DecisionTreeRegressor(max_depth=3)

reg_tree.fit(x_train,y_train)

y_train_pred = reg_tree.predict(x_train)
y_test_pred = reg_tree.predict(x_test)

print('MSE train_set: %.6f, test_set: %.6f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train_set: %.6f, test_set: %.6f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

#Plot decision tree regressor



plt.scatter(y_train_pred,  
            y_train_pred - y_train,
            c='blue',
            marker='o',
            s=35,
            alpha=0.5,
            label = "Training Data")
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='Red', 
            marker='s', 
            s=35,
            alpha=0.7,
            label = "Test Data")

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='lower left')
plt.hlines(y=0, xmin=0, xmax=75, lw=2, color='red')
plt.xlim([0, 75])
plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI/figures/decisiontree_no2.png', dpi=300)
plt.show()

#o3
#Decision Tree Regression 
attributes_o3 = list(data_o3.columns)

x = data_o3[attributes_o3].values

y = data_o3['O3 AQI'].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5, random_state=0)

print(x_train.shape, y_train.shape)

reg_tree = DecisionTreeRegressor(max_depth=3)

reg_tree.fit(x_train,y_train)

y_train_pred = reg_tree.predict(x_train)
y_test_pred = reg_tree.predict(x_test)

print('MSE train_set: %.6f, test_set: %.6f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train_set: %.6f, test_set: %.6f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

#Plot decision tree regressor

plt.scatter(y_train_pred,  
            y_train_pred - y_train,
            c='blue',
            marker='o',
            s=35,
            alpha=0.5,
            label = "Training Data")
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='Red', 
            marker='s', 
            s=35,
            alpha=0.7,
            label = "Test Data")

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='lower left')
plt.hlines(y=0, xmin=0, xmax=75, lw=2, color='red')
plt.xlim([0, 75])
plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI/figures/decisiontree_o3.png', dpi=300)
plt.show()



#so2
#Decision Tree Regression 
attributes_so2 = list(data_so2.columns)

x = data_so2[attributes_so2].values

y = data_so2['SO2 AQI'].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5, random_state=0)

print(x_train.shape, y_train.shape)

reg_tree = DecisionTreeRegressor(max_depth=3)

reg_tree.fit(x_train,y_train)

y_train_pred = reg_tree.predict(x_train)
y_test_pred = reg_tree.predict(x_test)

print('MSE train_set: %.6f, test_set: %.6f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train_set: %.6f, test_set: %.6f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

#Plot decision tree regressor

plt.scatter(y_train_pred,  
            y_train_pred - y_train,
            c='blue',
            marker='o',
            s=35,
            alpha=0.5,
            label = "Training Data")
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='Red', 
            marker='s', 
            s=35,
            alpha=0.7,
            label = "Test Data")

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='lower left')
plt.hlines(y=0, xmin=0, xmax=75, lw=2, color='red')
plt.xlim([0, 75])
plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI/figures/decisiontree_so2.png', dpi=300)
plt.show()


#co
#Decision Tree Regression 
attributes_co = list(data_co.columns)

x = data_co[attributes_co].values

y = data_co['CO AQI'].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5, random_state=0)

print(x_train.shape, y_train.shape)

reg_tree = DecisionTreeRegressor(max_depth=3)

reg_tree.fit(x_train,y_train)

y_train_pred = reg_tree.predict(x_train)
y_test_pred = reg_tree.predict(x_test)

print('MSE train_set: %.6f, test_set: %.6f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train_set: %.6f, test_set: %.6f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

#Plot decision tree regressor

plt.scatter(y_train_pred,  
            y_train_pred - y_train,
            c='blue',
            marker='o',
            s=35,
            alpha=0.5,
            label = "Training Data")
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='Red', 
            marker='s', 
            s=35,
            alpha=0.7,
            label = "Test Data")

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='lower left')
plt.hlines(y=0, xmin=0, xmax=75, lw=2, color='red')
plt.xlim([0, 75])
plt.tight_layout()
plt.savefig('C:/Users/anuj/Desktop/project EBI/figures/decisiontree_co.png', dpi=300)
plt.show()

