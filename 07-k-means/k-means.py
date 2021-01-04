import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D


def k_means_demo_01():
    # 随机生成数据集
    data = []
    for i in range(10_0000):
        line = []
        line.append(random.random() * random.randint(100, 100_0000))
        line.append(random.random() * random.randint(100, 100_0000))
        data.append(line)
    data = pd.DataFrame(data, columns=['x', 'y'])

    print(data)
    print('--' * 50)

    # k-means聚类
    km = KMeans(n_clusters=5)
    km.fit(data)
    predicted_lables = km.predict(data)

    print(predicted_lables)
    print('==' * 50)

    # 画图展示结果
    plt.figure(figsize=(16 * 1.5, 9 * 1.5), dpi=300)
    colors = ['red', 'blue', 'yellow', 'purple', '#54FF9F', '#EE2C2C']
    plt.scatter(data['x'], data['y'], color=[colors[i] for i in predicted_lables])
    # plt.show()
    plt.savefig("./k-means-xy.svg")

    # 轮廓系数
    print(silhouette_score(data, predicted_lables))


def k_means_demo_02():
    # 随机生成数据集
    data = []
    for i in range(100_0000):
        line = []
        line.append(random.random() * random.randint(100, 100_0000))
        line.append(random.random() * random.randint(100, 100_0000))
        line.append(random.random() * random.randint(100, 100_0000))
        data.append(line)
    data = pd.DataFrame(data, columns=['x', 'y', 'z'])

    print(data)
    print('--' * 50)

    # k-means聚类
    km = KMeans(n_clusters=5)
    km.fit(data)
    predicted_lables = km.predict(data)

    print(predicted_lables)
    print('==' * 50)

    colors = ['red', 'blue', 'yellow', 'purple', '#54FF9F', '#EE2C2C']

    # 画图展示结果
    fig = plt.figure(figsize=(16 * 1.5, 9 * 1.5), dpi=150)
    ax = fig.gca(projection='3d')
    ax.scatter(data['x'], data['y'], data['z'], color=[colors[i] for i in predicted_lables])

    plt.show()
    plt.savefig("./k-means-xy.svg")

    # 轮廓系数
    print(silhouette_score(data, predicted_lables))


def k_means_demo_03():
    data = circle()

    # plt.figure(figsize=(16, 9), dpi=300)
    # plt.scatter(data['x'], data['y'])
    # plt.show()

    print(data)
    print('--' * 50)

    # k-means聚类
    km = KMeans(n_clusters=6)
    km.fit(data)
    predicted_lables = km.predict(data)

    print(predicted_lables)
    print('==' * 50)

    colors = ['red', 'blue', 'yellow', 'purple', '#54FF9F', '#EE2C2C']

    # 画图展示结果
    plt.figure(figsize=(16 * 1.5, 9 * 1.5), dpi=300)
    colors = ['red', 'blue', 'yellow', 'purple', '#54FF9F', '#EE2C2C']
    plt.scatter(data['x'], data['y'], color=[colors[i] for i in predicted_lables])
    # plt.show()
    plt.savefig("./k-means-circle-xy.svg")

    # 轮廓系数
    print(silhouette_score(data, predicted_lables))


def circle(cx=10, cy=10, r=8, w=2):
    ring = []
    while True:
        point = []
        x = random.random() * 100
        y = random.random() * 100
        if r ** 2 <= (x - cx) ** 2 + (y - cy) ** 2 <= (r + w) ** 2:
            point.append(x)
            point.append(y)
            ring.append(point)

            if len(ring) >= 100_0000:
                break

    ring = pd.DataFrame(ring, columns=['x', 'y'])
    return ring


if __name__ == '__main__':
    # k_means_demo01()
    # k_means_demo_02()
    k_means_demo_03()
