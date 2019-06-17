from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D


def fire_module(x,num_filter_squeeze,num_filter_expand,block_num):
    fire2_squeeze = Conv2D(num_filter_squeeze, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire%i_squeeze'%(block_num))(x)
    fire2_expand1 = Conv2D(num_filter_expand, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire%i_expand1'%(block_num))(fire2_squeeze)
    fire2_expand2 = Conv2D(num_filter_expand, (3, 3), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire%i_expand2'%(block_num))(fire2_squeeze)
    merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])
    return merge2
 
# Define the Model 
nb_classes=1000 ## number of classes
inputs=(224, 224,3) ## size of the input

input_img = Input(shape=inputs)

conv1 = Conv2D(96,(7, 7), activation='relu', kernel_initializer='glorot_uniform',strides=(2, 2), padding='same', name='conv1')(input_img)

maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

merge2 = fire_module(maxpool1,16,64,2)

merge3 = fire_module(merge2,16,64,3)

merge4=fire_module(merge3,32,128,4)

maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

merge5=fire_module(maxpool4,32,128,5)

merge6=fire_module(merge5,48,192,6)

merge7=fire_module(merge6,48,192,7)

merge8=fire_module(merge7,64,256,8)

maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)

merge9=fire_module(maxpool8,64,256,9)

fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
conv10 = Conv2D(nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='valid', name='conv10')(fire9_dropout)

global_avgpool10 = GlobalAveragePooling2D()(conv10)
softmax = Activation("softmax", name='softmax')(global_avgpool10)

### build the model

model=Model(inputs=input_img, outputs=softmax)
