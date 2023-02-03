import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import adam_v2


#载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
#(60000,28,28)->(60000,28,28,1) 1为图片的深度，黑白为1，彩色为3
x_train = x_train.reshape(x_train.shape[0], 28,28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28,28,1).astype('float32')
#数据归一化处理
x_train/=255.0
x_test/=255.0

#标签转one-hot操作
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)


#定义顺序模型
model = Sequential()
#第一个卷积层
model.add(Convolution2D(
    input_shape = (28,28,1),
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation = 'relu'
))

#第一个池化层
model.add(MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same',
))
#第二个卷积层
model.add(Convolution2D(64,5,strides=1,padding='same',activation = 'relu'))
#第二个池化层
model.add(MaxPooling2D(2,2,'same'))
#把第二个池化层的输出扁平化为1维
model.add(Flatten())
#全连接层
model.add(Dense(1024,activation = 'relu'))
#输出层
model.add(Dense(10,activation='softmax'))

#模型
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#训练模型
model.fit(x_train,y_train,batch_size=64,epochs=10)

#评估模型
loss,accuracy = model.evaluate(x_test,y_test)

print('test loss',loss)
print('test accuracy',accuracy)
