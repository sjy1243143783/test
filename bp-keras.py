import tensorflow as tf
import keras              
from keras.datasets import mnist  #数据集
from keras.layers import Dense,Dropout  #神经网络层导入
from keras.models import Sequential #模型类型：Sequential序列模型和Model函数模型
from keras.utils import np_utils


#加载mnist数据集
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#mnist图像数据是28*28矩阵，全连接层是一排一排的数据形状，需要reshape为784(28*28),
x_train=x_train.reshape(60000,784).astype('float32')  #训练数据集60000个
x_test=x_test.reshape(10000,784).astype('float32')   #测试数据集10000个
#数据归一化处理
x_train/=255.0
x_test/=255.0

#标签转one-hot操作
y_train=np_utils.to_categorical(y_train,num_classes=10)#10个类别
y_test =np_utils.to_categorical(y_test,num_classes=10)


#搭建网络模型
#定义顺序模型
model=Sequential()  

#第一隐含层，input_shape为输入层输入结构，activation激活函数
model.add(Dense(units=128,input_shape=(784,),activation='relu'))
#第二层
model.add(Dense(units=128,activation='relu'))
#防止过拟合
model.add(Dropout(0.5))
#输出层,手写数字共0~9十个类别
model.add(Dense(units=10,activation='softmax'))

#模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#训练
model.fit(x=x_train,y=y_train,batch_size=64,epochs=10)
#模型测试
loss,acc=model.evaluate(x_test,y_test)
print('test loss',loss)
print('test accuracy',acc)

