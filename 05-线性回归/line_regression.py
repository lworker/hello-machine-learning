from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sgd_regression_demo01():
    """
    梯度下降法线性回归
    :return:
    """
    
    data = load_boston()

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

    xstd = StandardScaler()
    x_train = xstd.fit_transform(x_train)
    x_test = xstd.transform(x_test)

    ystd = StandardScaler()
    y_train = ystd.fit_transform(y_train.reshape(-1, 1))
    y_test = ystd.transform(y_test.reshape(-1, 1))

    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)

    # 输出回归系数
    print("回归系数", sgd.coef_)

    # 预测值
    y_predicted = ystd.inverse_transform(sgd.predict(x_test))
    print("预测值", y_predicted)

    print("均方误差", mean_squared_error(ystd.inverse_transform(y_test), y_predicted))


if __name__ == '__main__':
    sgd_regression_demo01()
