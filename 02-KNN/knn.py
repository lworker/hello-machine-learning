import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def knn_facebook_check_ins():
    """
    使用knn算法解决Facebook_V_Predicting_Check_Ins
    
    数据集下载地址：
    https://storage.googleapis.com/kagglesdsdata/competitions/5186/37497/train.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1609689215&Signature
    =BaK2JC9SLXWG67TZgjV7agcgOnBV8eWCEoovU2HWvb0uuOhmU9pAcslNAK%2FB1LmQASIUPOvenNofiI9%2FNg8RwiOk4dCN9pZY4GmkupTugnGCx96xfdir4DvDLhhLke6VYU5CxcgzZdj4JbsR0bcLNW1nejPncJ7dxHxXO1iB70Jv%2BNzx2F
    %2B2bKC1f9NBiIrhXtnyQEZZ0ajDk6s9pCaCAy2f8l9yAZeD7M0OXmG1r7E%2BYZhrqX6XUiJQ5%2F0oJF5l4gk0LufDr1BqaDeDWOPVJBS9B29SSZ%2FcNoD5UwwnbYwicODnlBbKVco9dAqcTPB49YmjY91cEXJkFwXiG%2F6gWw%3D%3D&response-content-disposition
    =attachment%3B+filename%3Dtrain.csv.zip
    
    :return:
    """
    
    # 读取文件
    data = pd.read_csv("./small.csv")
    # data = pd.read_csv("./train.csv")
    
    # In [3]: data
    # Out[3]:
    #         row_id       x       y  accuracy    time    place_id
    # 0            0  0.7941  9.0809        54  470702  8523065625
    # 1            1  5.9567  4.7968        13  186555  1757726713
    # 2            2  8.3078  7.0407        74  322648  1137537235
    # 3            3  7.3665  2.5165        65  704587  6567393236
    # 4            4  4.0961  1.1307        31  472130  7440663949
    # ...        ...     ...     ...       ...     ...         ...
    # 499994  499994  3.6970  3.8030         5   11820  8369273861
    # 499995  499995  7.5907  0.3640       518  713434  7691257599
    # 499996  499996  5.8723  9.1669        48  486531  4994928133
    # 499997  499997  2.4630  2.7223        65  310182  1434731239
    # 499998  499998  1.2904  6.5082       361  396761  2342888843
    #
    # [499999 rows x 6 columns]
    #
    # In [4]: data.info()
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 499999 entries, 0 to 499998
    # Data columns (total 6 columns):
    #  #   Column    Non-Null Count   Dtype
    # ---  ------    --------------   -----
    #  0   row_id    499999 non-null  int64
    #  1   x         499999 non-null  float64
    #  2   y         499999 non-null  float64
    #  3   accuracy  499999 non-null  int64
    #  4   time      499999 non-null  int64
    #  5   place_id  499999 non-null  int64
    # dtypes: float64(2), int64(4)
    # memory usage: 22.9 MB
    
    # 删除记录编号
    data.drop(["row_id"], axis=1, inplace=True)
    
    # In [8]: data
    # Out[8]:
    #              x       y  accuracy    time    place_id
    # 0       0.7941  9.0809        54  470702  8523065625
    # 1       5.9567  4.7968        13  186555  1757726713
    # 2       8.3078  7.0407        74  322648  1137537235
    # 3       7.3665  2.5165        65  704587  6567393236
    # 4       4.0961  1.1307        31  472130  7440663949
    # ...        ...     ...       ...     ...         ...
    # 499994  3.6970  3.8030         5   11820  8369273861
    # 499995  7.5907  0.3640       518  713434  7691257599
    # 499996  5.8723  9.1669        48  486531  4994928133
    # 499997  2.4630  2.7223        65  310182  1434731239
    # 499998  1.2904  6.5082       361  396761  2342888843
    #
    # [499999 rows x 5 columns]
    
    # Convert argument to datetime.
    timestamp = pd.to_datetime(data["time"], unit='s')
    
    # In [10]: timestamp
    # Out[10]:
    # 0        1970-01-06 10:45:02
    # 1        1970-01-03 03:49:15
    # 2        1970-01-04 17:37:28
    # 3        1970-01-09 03:43:07
    # 4        1970-01-06 11:08:50
    #                  ...
    # 499994   1970-01-01 03:17:00
    # 499995   1970-01-09 06:10:34
    # 499996   1970-01-06 15:08:51
    # 499997   1970-01-04 14:09:42
    # 499998   1970-01-05 14:12:41
    # Name: time, Length: 499999, dtype: datetime64[ns]
    
    time_index = pd.DatetimeIndex(timestamp)
    
    # In [12]: time_index
    # Out[12]:
    # DatetimeIndex(['1970-01-06 10:45:02', '1970-01-03 03:49:15',
    #                '1970-01-04 17:37:28', '1970-01-09 03:43:07',
    #                '1970-01-06 11:08:50', '1970-01-03 01:27:45',
    #                '1970-01-08 17:13:49', '1970-01-05 06:30:02',
    #                '1970-01-02 22:13:04', '1970-01-05 15:07:40',
    #                ...
    #                '1970-01-01 21:05:09', '1970-01-09 11:34:36',
    #                '1970-01-08 21:59:59', '1970-01-09 01:06:04',
    #                '1970-01-09 13:23:04', '1970-01-01 03:17:00',
    #                '1970-01-09 06:10:34', '1970-01-06 15:08:51',
    #                '1970-01-04 14:09:42', '1970-01-05 14:12:41'],
    #               dtype='datetime64[ns]', name='time', length=499999, freq=None)
    
    # 构造新特征
    data["weekday"] = time_index.weekday
    data["hour"] = time_index.hour
    
    # In [15]: data
    # Out[15]:
    #              x       y  accuracy    time    place_id  weekday  hour
    # 0       0.7941  9.0809        54  470702  8523065625        1    10
    # 1       5.9567  4.7968        13  186555  1757726713        5     3
    # 2       8.3078  7.0407        74  322648  1137537235        6    17
    # 3       7.3665  2.5165        65  704587  6567393236        4     3
    # 4       4.0961  1.1307        31  472130  7440663949        1    11
    # ...        ...     ...       ...     ...         ...      ...   ...
    # 499994  3.6970  3.8030         5   11820  8369273861        3     3
    # 499995  7.5907  0.3640       518  713434  7691257599        4     6
    # 499996  5.8723  9.1669        48  486531  4994928133        1    15
    # 499997  2.4630  2.7223        65  310182  1434731239        6    14
    # 499998  1.2904  6.5082       361  396761  2342888843        0    14
    #
    # [499999 rows x 7 columns]
    
    data.drop(["time"], axis=1, inplace=True)
    
    # In [17]: data
    # Out[17]:
    #              x       y  accuracy    place_id  weekday  hour
    # 0       0.7941  9.0809        54  8523065625        1    10
    # 1       5.9567  4.7968        13  1757726713        5     3
    # 2       8.3078  7.0407        74  1137537235        6    17
    # 3       7.3665  2.5165        65  6567393236        4     3
    # 4       4.0961  1.1307        31  7440663949        1    11
    # ...        ...     ...       ...         ...      ...   ...
    # 499994  3.6970  3.8030         5  8369273861        3     3
    # 499995  7.5907  0.3640       518  7691257599        4     6
    # 499996  5.8723  9.1669        48  4994928133        1    15
    # 499997  2.4630  2.7223        65  1434731239        6    14
    # 499998  1.2904  6.5082       361  2342888843        0    14
    #
    # [499999 rows x 6 columns]
    
    # 删除入住次数过低的地点
    counted_by_palce = data.groupby("place_id").count()
    
    # In [19]: counted_by_palce
    # Out[19]:
    #              x   y  accuracy  weekday  hour
    # place_id
    # 1000015801   3   3         3        3     3
    # 1000017288   2   2         2        2     2
    # 1000025138  11  11        11       11    11
    # 1000052096   9   9         9        9     9
    # 1000213704   5   5         5        5     5
    # ...         ..  ..       ...      ...   ...
    # 9999755282   1   1         1        1     1
    # 9999855083   1   1         1        1     1
    # 9999862567   1   1         1        1     1
    # 9999916757  11  11        11       11    11
    # 9999932225   4   4         4        4     4
    #
    # [95323 rows x 5 columns]
    
    selected_place = counted_by_palce[counted_by_palce.x > 10].reset_index()
    
    # In [26]: selected_place = counted_by_palce[counted_by_palce.x > 15].reset_index()
    #
    # In [27]: selected_place
    # Out[27]:
    #         place_id   x   y  accuracy  weekday  hour
    # 0     1000705331  19  19        19       19    19
    # 1     1001113605  16  16        16       16    16
    # 2     1001749677  17  17        17       17    17
    # 3     1002803051  20  20        20       20    20
    # 4     1004864193  21  21        21       21    21
    # ...          ...  ..  ..       ...      ...   ...
    # 5611  9986262808  19  19        19       19    19
    # 5612  9987944667  19  19        19       19    19
    # 5613  9987944703  26  26        26       26    26
    # 5614  9994432014  18  18        18       18    18
    # 5615  9998323806  25  25        25       25    25
    #
    # [5616 rows x 6 columns]
    
    data = data[data["place_id"].isin(selected_place.place_id)]
    
    # In [29]: data
    # Out[29]:
    #              x       y  accuracy    place_id  weekday  hour
    # 17      0.7084  8.9051        69  8016758016        6    17
    # 21      4.2830  3.1855        62  3938338894        1    19
    # 24      1.2837  7.5588         4  9885174082        5    20
    # 28      7.2382  4.3998        75  9713229580        3     3
    # 29      4.5083  1.8794         2  6163271747        6    14
    # ...        ...     ...       ...         ...      ...   ...
    # 499986  4.8401  3.3060        72  8216523459        2     7
    # 499990  3.0056  6.8140        58  2720206174        4    11
    # 499991  4.6500  6.0751        92  3813043135        3    21
    # 499996  5.8723  9.1669        48  4994928133        1    15
    # 499998  1.2904  6.5082       361  2342888843        0    14
    #
    # [113869 rows x 6 columns]
    
    # 特征、标签
    y = data["place_id"]
    x = data.drop(["place_id"], axis=1)
    
    # 划分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    
    for k in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=k)
        
        knn.fit(x_train, y_train)
        print("k=", k, knn.score(x_test, y_test))


if __name__ == '__main__':
    knn_facebook_check_ins()
