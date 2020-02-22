# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py   
 
@Contact :   luoguangwen@163.com


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/22 14:16    Kevin        1.0          None
"""
from keras.datasets import imdb
from keras.datasets import mnist
from keras.datasets import reuters

def get_imdb_data():
    """
    返回imdb数据集。 此处使用keras内置imdb模块进行加载。首次调用时会从网络下载。
    如果是缺省安装数据存储在'C:/Users/[用户名]/.keras/datasets/（windows系统）。
    imdb数据集：它包含来自互联网电影数据库（IMDB）的 50 000 条严重两极分
    化的评论。数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，
    训练集和测试集都包含 50% 的正面评论和 50% 的负面评论。
    :return:
    """
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print('-' * 40)
    print('IMDB数据集原始数据信息(keras):')
    print('--> imdb dataset  train data\'s shape:', train_data.shape, '\tndim:', train_data.ndim)
    print('--> imdb dataset  train label\'s shape:', train_labels.shape, '\tndim:', train_labels.ndim)
    print('--> imdb dataset  test data\'s shape:', test_data.shape, '\tndim:', test_data.ndim)
    print('--> imdb dataset  test label\'s shape:', test_labels.shape, '\tndim:', test_labels.ndim)

    return train_data,train_labels,test_data,test_labels

def get_mnist_data():
    """
    返回mnist数据集，此处使用keras内置mnist模块进行加载。首次调用时会从网络下载。
    如果是缺省安装数据存储在'C:/Users/[用户名]/.keras/datasets/（windows系统）。
    MNIST数据集：是机器学习领域的一个经典数据集，其历史几乎和这个领域一样长，而且
    已被人们深入研究。这个数据集包含 60 000 张训练图像和 10 000 张测试图像，由美
    国国家标准与技术研究院（National Institute of Standards and Technology，
    即 MNIST 中 的 NIST）在 20 世纪 80 年代收集得到。你可以将“解决”MNIST 问题看
    作深度学习的“HelloWorld”，正是用它来验证你的算法是否按预期运行。
    :return:
    """
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print('-' * 40)
    print('MNIST数据集原始数据信息(keras):')
    # (60000,28,28) 60000个像素为28*28的图片
    print('--> MNIST dataset train image\'s shape:', train_images.shape, '\tndim:', train_images.ndim)
    # （60000，）
    print('--> MNIST dataset train label\'s shape:', train_labels.shape, '\tndim:', train_labels.ndim)

    # (10000,28,28) 10000个像素为28*28的图片
    print('--> MNIST dataset test image\'s shape:', test_images.shape, '\tndim:', test_images.ndim)
    print('--> MNIST dataset test label\'s shape:', test_labels.shape, '\tndim:', test_labels.ndim)

    return train_images, train_labels, test_images, test_labels

    # 下面供数据向化量处理参考
    print('-' * 40)
    print('数据向量化处理，目标：形状为（60000，28*28），取值0~1 float32类型数组')
    train_images = train_images.reshape((60000, 28 * 28))  # (60000,28,28)  --> (60000,28*28)
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))  # (10000,28,28)  --> (10000,28*28)
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)  #
    test_labels = to_categorical(test_labels)

    print('-->train image\'s shape', train_images.shape, '\tndim:', train_images.ndim)
    print('-->train label\'s shape', train_labels.shape, '\tndim:', train_labels.ndim)
    print('-->test image\'s shape', test_images.shape, '\tndim:', test_images.ndim)
    print('-->test label\'s shape', test_labels.shape, '\tndim:', test_labels.ndim)

def get_reuters_data():
    """
    返回reuters数据集，此处使用keras内置reuters模块进行加载。首次调用时会从网络下载。
    如果是缺省安装数据存储在'C:/Users/[用户名]/.keras/datasets/（windows系统）。
    用路透社数据集，它包含许多短新闻及其对应的主题，由路透社在 1986 年发布。
    它是一个简单的、广泛使用的文本分类数据集。它包括 46 个不同的主题：某些主题的样本更多，但训练集中每个主题都有至少 10 个样本。
    :return:
    """
    (train_images, train_labels), (test_images, test_labels) = reuters.load_data(num_words=10000)
    print('-' * 40)
    print('REUTERS数据集原始数据信息(keras):')
    print('--> REUTERS dataset train data\'s shape:', train_images.shape, '\tndim:', train_images.ndim)
    print('--> REUTERS dataset train label\'s shape:', train_labels.shape, '\tndim:', train_labels.ndim)
    print('--> REUTERS dataset test data\'s shape:', test_images.shape, '\tndim:', test_images.ndim)
    print('--> REUTERS dataset test label\'s shape:', test_labels.shape, '\tndim:', test_labels.ndim)

    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    get_imdb_data()
    get_mnist_data()
    get_reuters_data()
    """
    执行结果如下所示：
----------------------------------------
IMDB数据集原始数据信息(keras):
--> imdb dataset  train data's shape: (25000,) 	ndim: 1
--> imdb dataset  train label's shape: (25000,) 	ndim: 1
--> imdb dataset  test data's shape: (25000,) 	ndim: 1
--> imdb dataset  test label's shape: (25000,) 	ndim: 1
----------------------------------------
MNIST数据集原始数据信息(keras):
--> MNIST dataset train image's shape: (60000, 28, 28) 	ndim: 3
--> MNIST dataset train label's shape: (60000,) 	ndim: 1
--> MNIST dataset test image's shape: (10000, 28, 28) 	ndim: 3
--> MNIST dataset test label's shape: (10000,) 	ndim: 1
----------------------------------------
REUTERS数据集原始数据信息(keras):
--> REUTERS dataset train data's shape: (8982,) 	ndim: 1
--> REUTERS dataset train label's shape: (8982,) 	ndim: 1
--> REUTERS dataset test data's shape: (2246,) 	ndim: 1
--> REUTERS dataset test label's shape: (2246,) 	ndim: 1
    """
    pass