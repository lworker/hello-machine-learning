import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def rf_demo01():
    """
    随机森林预测泰坦尼克号幸存者
    :return:
    """

    data = pd.read_csv("./titanic.csv")
    data.drop(["row_id", "PassengerId"], axis=1, inplace=True)

    y = data['Survived']
    x = data.drop(['Survived'], axis=1)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    rf = RandomForestClassifier()

    para = {"n_estimators": list(range(30, 120, 10)),
            "max_depth": list(range(5, 30, 5)),
            "criterion": ["gini", "entropy"],
            "min_samples_split": list(range(20, 50, 5)),
            "min_samples_leaf": list(range(5, 30, 5))}

    # 网格搜索最优参数，再加十次十折交叉验证
    gc = GridSearchCV(rf, param_grid=para, cv=10)

    gc.fit(x_train, y_train)

    print("准确率", gc.score(x_test, y_test))
    print("best_params_", gc.best_params_)


def rf_demo02():
    """
    随机森林 facebook入住问题
    :return:
    """

    # 读取文件
    data = pd.read_csv("./small.csv")
    # data = pd.read_csv("./train.csv")

    # 删除记录编号
    data.drop(["row_id"], axis=1, inplace=True)

    timestamp = pd.to_datetime(data["time"], unit='s')
    time_index = pd.DatetimeIndex(timestamp)

    # 构造新特征
    data["weekday"] = time_index.weekday
    data["hour"] = time_index.hour

    data.drop(["time"], axis=1, inplace=True)

    # 删除入住次数过低的地点
    counted_by_palce = data.groupby("place_id").count()
    selected_place = counted_by_palce[counted_by_palce.x > 10].reset_index()
    data = data[data["place_id"].isin(selected_place.place_id)]

    # 特征、标签
    y = data["place_id"]
    x = data.drop(["place_id"], axis=1)

    # 划分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 随机森林
    rf = RandomForestClassifier(max_depth=25, min_samples_split=80, min_samples_leaf=30, )
    rf.fit(x_train, y_train)
    print(rf.score(x_test, y_test))


if __name__ == '__main__':
    rf_demo01()

    print("=" * 100)

    rf_demo02()
