# -*- encoding: utf-8 -*-
"""
@File    :   imdb_conv_model.py   
@Contact :   luoguangwen@163.com
@Create Time: 2020/2/28 22:30      
@Author  : Kevin.Luo  
@Version : 1.0 
@Desciption :
"""

import sys

from keras.datasets import mnist
from keras.datasets import imdb
from keras.utils import to_categorical

from keras import models
from keras import layers

import matplotlib.pyplot as plt


def plt_config():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 设置支持显示 负号

    plt.rcParams['lines.linewidth'] = 5  # 设置线宽
    plt.rcParams['lines.color'] = 'red'  # 设置线颜色
    plt.rcParams['lines.linestyle'] = '-'  # 设置线的类型

    plt.title('中文标题')


def plt_test():
    height = [155, 179, 158, 179, 182, 160, 190, 172, 170, 169, 158, 185, 165, 184, 178, 190, 159, 181, 191, 176]
    bins = range(150, 191, 5)
    print(height)
    print(bins)
    plt.hist(height, bins=bins)  # 直方图
    plt.show()

def mnist_conv_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model



def prepare_imdb_data():
    (train_images, train_labels), (test_images, test_labels) = imdb.load_data(num_words=10000)
    print('-' * 40)
    print('原始数据信息:')
    print('-->train image\'s shape:', train_images.shape, '\tndim:',
          train_images.ndim)  # (60000,28,28) 60000个像素为28*28的图片
    print('-->train label\'s shape:', train_labels.shape, '\tndim:', train_labels.ndim)
    print('-->test image\'s shape:', test_images.shape, '\tndim:', test_images.ndim)
    print('-->test label\'s shape:', test_labels.shape, '\tndim:', test_labels.ndim)


def prepare_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print('-' * 40)
    print('原始数据信息:')
    print('-->train image\'s shape:', train_images.shape, '\tndim:',
          train_images.ndim)  # (60000,28,28) 60000个像素为28*28的图片
    print('-->train label\'s shape:', train_labels.shape, '\tndim:', train_labels.ndim)
    print('-->test image\'s shape:', test_images.shape, '\tndim:', test_images.ndim)
    print('-->test label\'s shape:', test_labels.shape, '\tndim:', test_labels.ndim)
    # plt.imshow(train_images[20],cmap=plt.cm.binary)
    # plt.show()

    print('-' * 40)
    print('数据处理，目标：形状为（60000，28,28,1），取值0~1 float32类型数组')
    train_images = train_images.reshape((60000, 28 ,28,1))  # (60000,28,28)  --> (60000,28*28)
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 ,28,1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)  #
    test_labels = to_categorical(test_labels)

    print('-->train image\'s shape', train_images.shape, '\tndim:',
          train_images.ndim)  # (60000,28,28) 60000个像素为28*28的图片
    print('-->train label\'s shape', train_labels.shape, '\tndim:', train_labels.ndim)
    print('-->test image\'s shape', test_images.shape, '\tndim:', test_images.ndim)
    print('-->test label\'s shape', test_labels.shape, '\tndim:', test_labels.ndim)

    return train_images, train_labels, test_images, test_labels


def main():
    prepare_imdb_data()
    return


    # 准备数据
    train_images, train_labels, test_images, test_labels = prepare_mnist_data()

    print('*'*60)
    print('Enter \'T\' to train model\nEnter \'L\' to load model\nEnter \'C\' to exit\n')
    print('*' * 60)
    one_char = sys.stdin.read(1)
    while 1 :
        print(one_char)
        if one_char == 'L' or one_char == 'l' :
            # 测试加载保存到本地文件中的模型
            conv_model = models.load_model(r'imdb_conv_model.h5')
            conv_model.summary()
            test_loss, test_acc = conv_model.evaluate(test_images, test_labels)
            print('test_loss:', test_loss, '\ntest_acc:', test_acc)
            return
        elif one_char == 't' or one_char == 'T' :
            # 构建模型
            conv_model = mnist_conv_model()
            conv_model.summary()

            # 使用训练数据训练模型
            conv_model.fit(train_images, train_labels, epochs=5, batch_size=128)

            # 使用测试数据评估模型
            test_loss, test_acc = conv_model.evaluate(test_images, test_labels)
            print('test_loss:', test_loss, '\ntest_acc:', test_acc)

            conv_model.save(r'./imdb_conv_model.h5')
            return
        elif one_char == 'c' or one_char == 'C':
            return
        else:
            one_char = sys.stdin.read(1)





if __name__ == '__main__':
    main()
    pass