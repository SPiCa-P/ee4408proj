# -*-coding = utf-8 -*-
# @Time : 2021/12/21 14:42
# @Author : Liu ChaoWei U2101053
# @FILE : Random Forest for iris.py
# @Software: PyCharm

import time
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# plot for classification by using two features input
def plot(x, y, test_data, test_label, model):
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()

    cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
    cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    plt.xlabel('petal length in cm', fontsize=10)
    plt.ylabel('petal width in cm', fontsize=10)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('classification for iris using random forest with two features')
    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=cm_dark)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label, s=30, edgecolors='k', zorder=2,
                cmap=cm_dark)
    # 画分类界面
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = model.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.show()


# define main function for different number of decision trees in the random forest
def main_for_number_of_decision_tress():
    iris = datasets.load_iris()
    print(iris)
    x = iris["data"]
    y = iris["target"]
    # divide the data and labels into train and test
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6,
                                                                      test_size=0.4)
    print("\ntrain size：", train_data.shape)
    print("train lable：", train_label.shape)
    print("test size：", test_data.shape)
    print("test label：", test_label.shape)
    scores_fortreenumber = []
    Time = []
    for j in range(1, 30):
        scores = 0
        t1 = time.time()
        for i in range(500):
            # 实例化随机森林分类器
            clf = RandomForestClassifier(j, criterion="entropy")
            # 训练模型
            clf.fit(train_data, train_label)
            # 评价模型
            score = clf.score(test_data, test_label)
            scores = score + scores
        scores_fortreenumber.append(scores / 500)
        print(str(j) + " decision trees" + "      accuracy：", scores / 500)
        t2 = time.time()
        Time.append(t2 - t1)

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    x = list(range(1, 30))
    plt.figure(figsize=[12, 4.8])
    plt.subplot(121)
    plt.plot(x, scores_fortreenumber, color="red")
    plt.legend(["accuracy"])
    plt.xlabel('the number of decision trees')
    plt.ylabel('accuracy')
    plt.subplot(122)
    plt.plot(x, Time, color="blue")
    plt.xlabel('the number of decision trees')
    plt.ylabel('Time(s)')
    plt.legend(["Time"])
    plt.suptitle('The accuracies and expense of random forest for different number of trees')
    plt.show()
    # plot(x,y,test_data,test_label,clf)
    #
    #
    #     print("\n特征重要程度为：")
    #     info = [*zip(feature_names, clf.feature_importances_)]
    #     for cell in info:
    #         print(cell)


# define main function for normal random foreast by using two input features and four input features
def main_for_noraml_random_foreast():
    # 利用自定义函数导入Iris数据集
    iris = datasets.load_iris()
    x = iris.data[:, (1, 3)]
    y = iris["target"]
    # divide the data and labels into train and test
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6,
                                                                      test_size=0.4)    
    scores_fortreenumber = []
    scores = 0
    for i in range(1000):
        # 实例化随机森林分类器
        clf = RandomForestClassifier(10, criterion="gini")
        # 训练模型
        clf.fit(train_data, train_label)
        # 评价模型
        score = clf.score(test_data, test_label)
        scores = score + scores
    scores_fortreenumber.append(scores / 1000)
    print("      accuracy：", scores / 1000)
    # plot(x,y,test_data,test_label,clf)
    #
    #
    #     print("\n特征重要程度为：")
    #     info = [*zip(feature_names, clf.feature_importances_)]
    #     for cell in info:
    #         print(cell)


if __name__ == "__main__":
    main_for_noraml_random_foreast()
    main_for_number_of_decision_tress()
