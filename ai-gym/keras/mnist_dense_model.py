# -*- encoding: utf-8 -*-
"""
@File    :   mnist_dense_model.py   
 
@Contact :   luoguangwen@163.com


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/22 18:46    Kevin        1.0          None
"""
import sys

from keras.datasets import mnist
from keras.utils import to_categorical

from keras import models
from keras import layers

import matplotlib.pyplot as plt



def mnist_dense_model():
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return network



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
    print('数据处理，目标：形状为（60000，28*28），取值0~1 float32类型数组')
    train_images = train_images.reshape((60000, 28 * 28))  # (60000,28,28)  --> (60000,28*28)
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
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


    # 准备数据
    train_images, train_labels, test_images, test_labels = prepare_mnist_data()

    print('*' * 60)
    print('Enter \'T\' to train model\nEnter \'L\' to load model\nEnter \'C\' to exit\n')
    print('*' * 60)
    one_char = sys.stdin.read(1)

    while 1 :
        print(one_char)
        if one_char == 'L' or one_char == 'l' :
            # 测试加载保存到本地文件中的模型
            network = models.load_model(r'mnist_dense_model.h5')
            test_loss,test_acc = network.evaluate(test_images, test_labels)

            print('test_loss:', test_loss, '\ntest_acc:', test_acc)
            return
        elif one_char == 't' or one_char == 'T' :
            # 构建模型
            network = mnist_dense_model()

            # 使用训练数据训练模型
            history = network.fit(train_images, train_labels, epochs=5, batch_size=128)

            #保存模型到文件
            network.save(r'./mnist_dense_model.h5')

            #图形显示训练精度
            print(history.history)
            acc = history.history['accuracy']
            loss = history.history['loss']

            epochs = range(1, len(acc) + 1)
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, loss, 'b', label='Training loss')
            plt.title('Training acc and loss')
            plt.show()

            # 使用测试数据评估模型
            test_loss,test_acc =  network.evaluate(test_images, test_labels)
           # print('test_loss:', test_loss, '\ntest_acc:', test_acc)


            return
        elif one_char == 'c' or one_char == 'C':
            return
        else:
            one_char = sys.stdin.read(1)



if __name__ == '__main__':
    main()
    pass
