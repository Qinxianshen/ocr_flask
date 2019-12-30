#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model
# import keras.backend as K


#设置keras tensorflow gpu利用率 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

from . import keys
from . import densenet


#借入flask框架后出现问题：ValueError: Tensor Tensor("out_2/div:0", shape=(?, ?, 5990), dtype=float32) is not an element of this graph.
#解决方案：https://blog.csdn.net/qq_31112205/article/details/102700427
#在tf.get_default_graph()模式下运行keras.models.predict（我采用的方法）


graph = tf.get_default_graph()    # 声明get_default_graph()，注意需要在模型加载前，最好是在引库的后面


reload(densenet)

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

input = Input(shape=(32, None, 1), name='the_input')
y_pred= densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

modelPath = os.path.join(os.getcwd(), 'densenet/models/weights_densenet.h5')
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    
    img = img.resize([width, 32], Image.ANTIALIAS)
   
    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    
    X = img.reshape([1, 32, width, 1])

    global graph                      # 新添加的代码。。。。。。
    with graph.as_default():          # 新添加的代码。。。。。。
        y_pred = basemodel.predict(X) # 修改的代码。。。。。。。
    
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)

    return out
