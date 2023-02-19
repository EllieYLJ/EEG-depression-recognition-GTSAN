from keras.models import Model
from keras.layers import add, Layer,Input, Conv1D, Activation, Flatten, Dense,GRU,Lambda,Dropout,Concatenate,SpatialDropout1D
from keras.utils import to_categorical
import scipy.io 
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split 
from keras.layers import SeparableConv1D,MaxPooling1D,AveragePooling1D,GlobalMaxPooling1D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.optimizers import SGD, Nadam, Adam, RMSprop
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



    
data = scipy.io.loadmat("E:\data\psd.mat")
data = data['features']
label= scipy.io.loadmat("E:\data\psd.mat")
label = label['label']


mean = data.mean(axis=0)
data -= mean 
std = data.std(axis=0) 
data /= std
x_train, x_test, y_train, y_test = train_test_split(data,label, test_size=0.2, random_state=42)



num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_train=x_train.reshape(2400,63,4)
x_test = x_test.astype('float32')
x_test=x_test.reshape(600,63,4)



def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations

class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
 
 
# Residual block 残差块
def ResBlock(x, filters, kernel_size,dilation_rate,F2):
    r = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_uniform',dilation_rate=dilation_rate, activation='relu')(
        x)
    r       = BatchNormalization()(r)
    r       = Activation('relu')(r)
    r       = SpatialDropout1D(0.2)(r)
    r       = SeparableConv1D(F2, kernel_size, strides=1, use_bias=False,padding='same',dilation_rate=1, depth_multiplier=1)(r)
    r       = BatchNormalization()(r)
    r       = Activation('relu')(r)
    r       = SpatialDropout1D(0.2)(r)
    # r       = MaxPooling1D()(r)





    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  	# shortcut (shortcut)
    o = add([r, shortcut])
    # Activation function
    o = Activation('relu')(o)  
    return o
def slice(x,index):
    return x[:,:,:,index]

 
# Sequence Model 时序模型
def TCN(x_train,y_train,x_test,y_test,return_sequences=False):
    inputs = Input(shape=(63,4))
    x = ResBlock(inputs, filters=64, kernel_size=5,dilation_rate=1,F2=64)
    # x1 = Dropout(0.2)(x1)
    x= ResBlock(x, filters=16, kernel_size=3, dilation_rate=2,F2=16)
    # x1 = Dropout(0.2)(x1)
    x = ResBlock(x, filters=32, kernel_size=2, dilation_rate=4,F2=32)
    # x1 = Dropout(0.2)(x1)
    # batch_size = x1[0]
    # batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
    # nb_filters = x1[-1]
    # x= [batch_size, nb_filters]
    

    x2 = GRU(32,return_sequences=True)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    x = Concatenate(axis=-1)([x, x2])
    x = AttentionLayer(32)(x)
    x = Dense(2, activation='softmax')(x)
 
    # x = Flatten()(x)



    return Model(input=inputs, output=x)

    
model =  TCN(x_train,y_train,x_test,y_test)
optim = RMSprop(lr = 0.02)
    # View network structure 查看网络结构
model.summary()
    # Compile model 编译模型
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # Training model 训练模型
model.fit(x_train,y_train, batch_size=128, nb_epoch=150, verbose=1, validation_data=(x_test, y_test))
    # Assessment model 评估模型
pre = model.evaluate(x_test,y_test, batch_size=64, verbose=2)
print('test_loss:', pre[0], '- test_acc:', pre[1])
 



y_test1 = [np.argmax(item) for item in y_test]#将onehot编码转成一般编码  
# for item in y_test:
y_pred = model.predict(x_test)      
aa = [np.argmax(item) for item in y_test]#将onehot编码转成一般编码 
# # TCN(x_train,y_train,x_test,y_test)
y_pred1= [np.argmax(item) for item in y_pred]#将onehot编码转成一般编码  



def spe(Y_test,Y_pred,n):
    
    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    
    return spe


print("specificity_score:",spe(y_test1, y_pred1,2))
print("accuracy_score:", accuracy_score(y_test1, y_pred1))
print("precision_score:", metrics.precision_score(y_test1, y_pred1))
print("recall_score:", metrics.recall_score(y_test1, y_pred1))


