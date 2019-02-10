
# coding: utf-8

# In[1]:


## https://github.com/titu1994/DenseNet/blob/master/densenet.py

import numpy as np
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model, convert_dense_weights_data_format
from keras.utils.data_utils import get_file
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

def makeOneHot(y):
     u=np.unique(y)
     x=np.zeros((len(u),len(y)))
     for i in range(len(y)):
         x[y[i],i]=1
     return x
 
######################################### Loading and preprocessing data #######################################
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train=X_train/255
X_test=X_test/255
Y_train=makeOneHot(Y_train).T
Y_test=makeOneHot(Y_test).T
X_train, X_Dev, Y_train, Y_Dev = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)


# In[2]:


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4,conv_num='NA',dense_num="NA"):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    x = BatchNormalization(axis=-1, epsilon=1.1e-5,name="Btch1Convblock"+conv_num+"DensBlock"+dense_num)(ip)
    x = Activation('relu',name="Relu1Convblock"+conv_num+"DensBlock"+dense_num)(x)

    if bottleneck:
        inter_channel = nb_filter * 4  # According to the paper, each 1*1 Conv produces 4*k feature-maps

        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay),name="bottlenkConvblock"+conv_num+"DensBlock"+dense_num)(x)
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False,
              name="conv33Convblock"+conv_num+"DensBlock"+dense_num)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


# In[3]:


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    nb_filter=int(nb_filter * compression)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x) # Suspected if there should be Relu here !!!!!!!
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


# In[4]:


def __dense_block(x, nb_layers,growth_rate=12, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  return_concat_list=False,num="NA"):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay,conv_num=str(i+1),dense_num=num)
        x_list.append(cb)
        x = concatenate([x, cb], axis=concat_axis)

    nb_filter=0
    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


# In[5]:


def __create_dense_net(nb_classes, img_input, include_top, growth_rate=12,nb_layers_per_block=[4,6,8], bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        growth_rate: number of filters to add per dense block
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
        subsample_initial:
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # layers in each dense block
    nb_layers = list(nb_layers_per_block)  # Convert tuple to list

    final_nb_layer = nb_layers[-1]
    nb_layers = nb_layers[:-1]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    nb_filter = 2 * growth_rate
    
    # compute compression factor
    compression = 1.0 - reduction
    #print(nb_filter)
    # Initial convolution
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)

    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Add dense blocks
    for block_idx in range(len(nb_layers_per_block) - 1):
        x, _ = __dense_block(x, nb_layers[block_idx], growth_rate=growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay,num=str(block_idx+1))
        # add transition_block
        x = __transition_block(x, 2*growth_rate, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, _ = __dense_block(x, final_nb_layer, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay,num=str(len(nb_layers_per_block)))

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(nb_classes, activation=activation)(x)

    return x


# In[6]:


def DenseNet(input_shape,growth_rate=12,nb_layers_per_block=[4,4,4],
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
             include_top=True, weights=None,
             classes=10, activation='softmax'):
    '''Instantiate the DenseNet architecture,
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            bottleneck: flag to add bottleneck blocks in between dense blocks
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
            include_top: whether to include the fully-connected
                layer at the top of the network.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
        # Returns
            A Keras model instance.
        '''

    # Determine proper input shape
    
    img_input = Input(shape=input_shape)
    x = __create_dense_net(classes, img_input, include_top,
                           growth_rate, nb_layers_per_block, bottleneck, reduction,
                           dropout_rate, weight_decay, subsample_initial_block, activation)

    model = Model(img_input, x, name='densenet')

    
    return model


# In[7]:


from keras.optimizers import Adam
model = DenseNet((32, 32, 3), nb_layers_per_block=[6,12,24,16],
                 growth_rate=12, bottleneck=True, weights=None,dropout_rate=0.2)

optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
print("Model Ready.")
#model.load_weights("Dense6_12_24_16_k12.h5")


batchsize=16
generator = ImageDataGenerator(rotation_range=15,width_shift_range=5./32,height_shift_range=5./32,horizontal_flip=True)
generator.fit(X_train, seed=0)
train_generator=generator.flow(X_train,Y_train, batch_size=batchsize)


# In[9]:


history=model.fit_generator(train_generator, steps_per_epoch=len(X_train) // batchsize, epochs=10, validation_data=(X_Dev, Y_Dev))
model.save("Dense6_12_24_16_k12.h5")

