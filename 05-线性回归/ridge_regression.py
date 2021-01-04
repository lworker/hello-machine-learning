import sklearn.datasets
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ridge_regression_deom01():
    data = load_boston()

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

    xstd = StandardScaler()
    x_train = xstd.fit_transform(x_train)
    x_test = xstd.transform(x_test)

    ystd = StandardScaler()
    y_train = ystd.fit_transform(y_train.reshape(-1, 1))

    ridge = Ridge()
    ridge.fit(x_train, y_train)
    predicted_y = ystd.inverse_transform(ridge.predict(x_test))

    print("预测值", predicted_y)

    print("均方误差", mean_squared_error(y_test, predicted_y))


if __name__ == '__main__':
    ridge_regression_deom01()
