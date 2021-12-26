import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

wine = pd.read_csv(r'wine.data',header=None)
wine.columns =  ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
# wine = wine.values
# x = wine[:,0]
# y = wine[:,1:]
x,y = wine.iloc[:,1:].values,wine.iloc[:,0].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
feat_labels = wine.columns[1:]
# n_estimators：森林中树的数量
# n_jobs  整数 可选（默认=1） 适合和预测并行运行的作业数，如果为-1，则将作业数设置为核心数
forest = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
testforest = forest.score(x_test,y_test)
print("******************************************")
print('ramdomforest_score=',testforest)
# if y_test == y_testforest:
    
 
# 下面对训练好的随机森林，完成重要性评估
# feature_importances_  可以调取关于特征重要程度
importances = forest.feature_importances_
print("importances:",importances)
x_columns = wine.columns[1:]
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
# 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
# 到根，根部重要程度高于叶子。
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
 

# 筛选变量（选择重要性比较高的变量）
threshold = 0.15
x_selected = x_train[:,importances > threshold]
 
# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.title("feature importance",fontsize = 18)
plt.ylabel("import level",fontsize = 15,rotation=90)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
for i in range(x_columns.shape[0]):
    plt.bar(i,importances[indices[i]],color='orange',align='center')
    #plt.xticks(np.arange(x_columns.shape[0]),x_columns,rotation=90,fontsize=15)
    plt.xticks(np.arange(indices.shape[0]),feat_labels[indices],rotation=90,fontsize=15)
plt.show()

