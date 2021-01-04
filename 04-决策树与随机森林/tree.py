import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus


def decision_tree_demo01():
    """
    决策树预测泰坦尼克号幸存者
    :return:
    """
    
    data = pd.read_csv("./titanic.csv")
    data.drop(["row_id", "PassengerId"], axis=1, inplace=True)
    
    y = data['Survived']
    x = data.drop(['Survived'], axis=1)
    
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
    decision_tree = DecisionTreeClassifier(max_depth=9, min_samples_split=18, min_samples_leaf=8)
    decision_tree.fit(x_train, y_train)
    print(decision_tree.score(x_test, y_test))
    
    # 决策树可视化
    dot_data = tree.export_graphviz(decision_tree)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_jpg("tree.jpg")


if __name__ == '__main__':
    decision_tree_demo01()
