from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


def logistics_demo():
    """
    逻辑回归处理肿瘤数据
    :return:
    """

    # 构造列标签名字
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
              'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    # 读取数据
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column)

    # In [6]: data
    # Out[6]:
    #      Sample code number  Clump Thickness  Uniformity of Cell Size  Uniformity of Cell Shape  ...  Bland Chromatin  Normal Nucleoli Mitoses  Class
    # 0               1000025                5                        1                         1  ...                3                1       1      2
    # 1               1002945                5                        4                         4  ...                3                2       1      2
    # 2               1015425                3                        1                         1  ...                3                1       1      2
    # 3               1016277                6                        8                         8  ...                3                7       1      2
    # 4               1017023                4                        1                         1  ...                3                1       1      2
    # ..                  ...              ...                      ...                       ...  ...              ...              ...     ...    ...
    # 694              776715                3                        1                         1  ...                1                1       1      2
    # 695              841769                2                        1                         1  ...                1                1       1      2
    # 696              888820                5                       10                        10  ...                8               10       2      4
    # 697              897471                4                        8                         6  ...               10                6       1      4
    # 698              897471                4                        8                         8  ...               10                4       1      4
    #
    # [699 rows x 11 columns]

    # 处理缺失值
    data.replace(to_replace='?', value=np.nan, inplace=True)
    data.dropna(inplace=True)

    # In [14]: data
    # Out[14]:
    #      Sample code number  Clump Thickness  Uniformity of Cell Size  Uniformity of Cell Shape  ...  Bland Chromatin  Normal Nucleoli Mitoses  Class
    # 0               1000025                5                        1                         1  ...                3                1       1      2
    # 1               1002945                5                        4                         4  ...                3                2       1      2
    # 2               1015425                3                        1                         1  ...                3                1       1      2
    # 3               1016277                6                        8                         8  ...                3                7       1      2
    # 4               1017023                4                        1                         1  ...                3                1       1      2
    # ..                  ...              ...                      ...                       ...  ...              ...              ...     ...    ...
    # 694              776715                3                        1                         1  ...                1                1       1      2
    # 695              841769                2                        1                         1  ...                1                1       1      2
    # 696              888820                5                       10                        10  ...                8               10       2      4
    # 697              897471                4                        8                         6  ...               10                6       1      4
    # 698              897471                4                        8                         8  ...               10                4       1      4
    #
    # [683 rows x 11 columns]

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 标准化
    stder = StandardScaler()
    x_train = stder.fit_transform(x_train)
    x_test = stder.transform(x_test)

    log = LogisticRegression()
    log.fit(x_train, y_train)
    print("回归系数", log.coef_)
    print("准确率：", log.score(x_test, y_test))


if __name__ == '__main__':
    logistics_demo()
