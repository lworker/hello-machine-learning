import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import jieba


def demo_01():
    """
    字典特征抽取
    :return:
    """

    dict_vectorizer = DictVectorizer(sparse=False)
    source_data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    data = dict_vectorizer.fit_transform(source_data)

    # one-hot编码
    print(dict_vectorizer.get_feature_names())
    print(data)


def demo_02():
    """
    文本特征抽取
    """

    cv = CountVectorizer()
    data = cv.fit_transform(["life is short,i like python", "life is too long,i dislike python"])

    print(cv.get_feature_names())
    # ['dislike', 'is', 'life', 'like', 'long', 'python', 'short', 'too']

    print(data.toarray())
    # [[0 1 1 1 0 1 1 0]
    #  [1 1 1 0 1 1 0 1]]

    print(data)
    #   (0, 2)	1
    #   (0, 1)	1
    #   (0, 6)	1
    #   (0, 3)	1
    #   (0, 5)	1
    #   (1, 2)	1
    #   (1, 1)	1
    #   (1, 5)	1
    #   (1, 7)	1
    #   (1, 4)	1
    #   (1, 0)	1


def demo_03():
    """
    中文特征抽取
    :return:
    """
    txts = [
        "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
        "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
        "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    for i in range(len(txts)):
        txts[i] = " ".join(list(jieba.cut(txts[i])))

    cv = CountVectorizer()
    data = cv.fit_transform(txts)

    print(cv.get_feature_names())
    print(data.toarray())
    print(data)


def demo_04():
    """
    tf-idf
    :return:
    """
    txts = [
        "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
        "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
        "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    for i in range(len(txts)):
        txts[i] = " ".join(list(jieba.cut(txts[i])))

    tv = TfidfVectorizer()
    data = tv.fit_transform(txts)

    print(tv.get_feature_names())
    print(data.toarray())
    print(data)


def demo_05():
    """
    归一化
    :return:
    """

    mm = MinMaxScaler()
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)


def demo_06():
    """
    标准化
    :return:
    """
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)


def demo_07():
    """
    特征选择：删除低方差列
    :return:
    """
    var = VarianceThreshold(threshold=0.1)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    print(data)


def demo_08():
    """
    PCA主成分分析降维
    :return:
    """
    pca = PCA(n_components=0.95)
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    print(data)


if __name__ == '__main__':
    demo_01()

    demo_02()

    demo_03()

    demo_04()

    demo_05()

    demo_06()

    demo_07()

    demo_08()
