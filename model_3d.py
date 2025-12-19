import tensorflow as tf
import param_3d
import numpy as np

import tensorflow as tf

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda

from tensorflow.keras import layers, models, optimizers


def get_pig_model_n(n,k1,k2):
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_"+str(n)+"_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    if unique:
        latest_weight = max(glob.glob(os.path.join("models_gmm_"+str(n)+"_unique_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    if t1:
        latest_weight = max(glob.glob(os.path.join("models_gmm_"+str(n)+"_t1_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model
    
def get_pig_model(k1,k2):
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    if t1:
        latest_weight = max(glob.glob(os.path.join("models_gmm_t1_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    elif t2:
        latest_weight = max(glob.glob(os.path.join("models_gmm_t2_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model


def unet_model(input_shape):
    inputs = tf.keras.Input(input_shape)
    x = inputs
    
    # Encoder
    for filters in [16, 32, 64]:
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Bottleneck
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # Add batch normalization
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # Add batch normalization
    
    # Decoder
    for filters in [64, 32, 16]:
        x = layers.Conv3DTranspose(filters, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
    
    outputs = layers.Conv3D(2, (1, 1, 1), activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model



class detect_12Net(tf.keras.Model):
    def __init__(self):
        super(detect_12Net, self).__init__()
        self.conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2, padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(16, activation='relu')
        self.fc2 = Dense(2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# class detect_12Net(tf.keras.Model):
#     def __init__(self):
#         super(detect_12Net, self).__init__()
#         self.conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')
#         self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
#         self.conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
#         self.conv3 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
#         self.flatten = tf.keras.layers.Flatten()

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#         return x

# class detect_24Net(tf.keras.Model):
#     def __init__(self):
#         super(detect_24Net, self).__init__()
#         self.conv1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')
#         self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=3, padding='same')
#         self.conv2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='valid')
#         self.conv3 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
#         self.flatten = tf.keras.layers.Flatten()

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#         return x

class detect_24Net(tf.keras.Model):
    def __init__(self):
        super(detect_24Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
        self.conv3 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='valid')
        self.conv4 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='valid')
        self.conv5 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='valid')
        self.conv6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='valid')
        self.conv7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
        self.conv8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='valid')
        self.conv9 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='valid')
        self.conv10 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.flatten(x)
        return x


# class detect_48Net(tf.keras.Model):
#     def __init__(self):
#         super(detect_48Net, self).__init__()
#         self.conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')
#         self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=3, padding='same')
#         self.conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
#         self.conv3 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
#         self.flatten = tf.keras.layers.Flatten()

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#         return x


class detect_48Net(tf.keras.Model):
    def __init__(self):
        super(detect_48Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
        self.conv3 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='valid')
        self.conv4 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='valid')
        self.conv5 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='valid')
        self.conv6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='valid')
        self.conv7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
        self.conv8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='valid')
        self.conv9 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='valid')
        self.conv10 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.flatten(x)
        return x


# Define the model
class calib_12Net(tf.keras.Model):
    def __init__(self, num_classes=135):
        super(calib_12Net, self).__init__()
        self.conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')
        self.pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class calib_24Net(tf.keras.Model):
    def __init__(self, num_classes=135):
        super(calib_24Net, self).__init__()
        self.conv1 = Conv3D(32, (5, 5, 5), activation='relu', padding='same')
        self.pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2, padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(64, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class calib_48Net(tf.keras.Model):
    def __init__(self, num_classes=135):
        super(calib_48Net, self).__init__()
        self.conv1 = Conv3D(64, (5, 5, 5), activation='relu', padding='same')
        self.pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2, padding='same')
        self.conv2 = Conv3D(64, (5, 5, 5), activation='relu', padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Define weight and bias initialization functions
def weight_variable(shape, name):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv3d(x, W, stride, pad="SAME"):
    return tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding=pad)

def max_pool3d(x, ksize, stride):
    return tf.nn.max_pool3d(x, ksize=[1, ksize, ksize, ksize, 1], strides=[1, stride, stride, stride, 1], padding="SAME")





import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import orthogonal

from tensorflow.keras import layers, models, Input
from tensorflow.keras.initializers import orthogonal


# # Define the 3D Convolution Layer
# def ConvolutionLayer3D(x, filters, kernel, strides, padding, block_id, kernel_init=orthogonal()):
#     prefix = f'block_{block_id}_'
#     x = layers.Conv3D(filters, kernel_size=kernel, strides=strides, padding=padding,
#                       kernel_initializer=kernel_init, name=prefix+'conv')(x)
    
#     x = layers.LeakyReLU(name=prefix+'lrelu')(x)
#     x = layers.Dropout(0.2, name=prefix+'drop')(x)
#     # x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
#     return x

# # Define the 3D Deconvolution Layer
# def DeconvolutionLayer3D(x, filters, kernel, strides, padding, block_id, kernel_init=orthogonal()):
#     prefix = f'block_{block_id}_'
#     x = layers.Conv3DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
#                                kernel_initializer=kernel_init, name=prefix+'de-conv')(x)
#     x = layers.Conv3DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
#                                kernel_initializer=kernel_init, name=prefix+'de-conv')(x)
#     x = layers.LeakyReLU(name=prefix+'lrelu')(x)
#     x = layers.Dropout(0.2, name=prefix+'drop')(x)
#     # x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
#     return x

# # Define the Noise Model
# def noiseModel(input_shape):
#     inputs = layers.Input(shape=input_shape)
#     # Encoder
#     conv1 = ConvolutionLayer3D(inputs, 64, (3, 3, 3), strides=(2, 2, 2), padding='same', block_id=1)
#     conv2 = ConvolutionLayer3D(conv1, 128, (3, 3, 3), strides=(2, 2, 2), padding='same', block_id=2)
#     conv3 = ConvolutionLayer3D(conv2, 192, (3, 3, 3), strides=(2, 2, 2), padding='same', block_id=3)
#     # Decoder
#     deconv1 = DeconvolutionLayer3D(conv3, 128, (3, 3, 3), strides=(2, 2, 2), padding='same', block_id=4)
#     deconv2 = DeconvolutionLayer3D(deconv1, 64, (3, 3, 3), strides=(2, 2, 2), padding='same', block_id=5)
#     deconv3 = DeconvolutionLayer3D(deconv2, 1, (3, 3, 3), strides=(2, 2, 2), padding='same', block_id=6)
#     noise_model = models.Model(inputs=inputs, outputs=deconv3)
#     return noise_model

# # Define the Noise Model
# def noiseModel(input_shape):
#     inputs = layers.Input(shape=input_shape)
    
#     en = [16, 16, 32, 32, 64, 64, 128,128, 192]
#     de = [192, 128, 128, 64, 64, 32, 32, 16, 16, 1]
    
#     x = inputs
#     # Encoder
#     for i, filters in enumerate(en):
#         x = ConvolutionLayer3D(x, filters, (3, 3, 3), strides=(2, 2, 2) if i < len(en)-1 else (1, 1, 1), padding='same', block_id=i+1)
    
#     # Decoder
#     for i, filters in enumerate(de):
#         x = DeconvolutionLayer3D(x, filters, (3, 3, 3), strides=(2, 2, 2) if i < len(de)-1 else (1, 1, 1), padding='same', block_id=len(en) + i + 1)
    
#     noise_model = models.Model(inputs=inputs, outputs=x)
#     return noise_model

def ConvolutionLayer3D(x, filters, kernel, strides, padding, block_id, layer_id, kernel_init=orthogonal()):
    prefix = f'block_{block_id}_layer_{layer_id}_'
    x = layers.Conv3D(filters, kernel_size=kernel, strides=strides, padding=padding,
                      kernel_initializer=kernel_init, name=prefix+'conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(0.2, name=prefix+'drop')(x)
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    return x

def DeconvolutionLayer3D(x, filters, kernel, strides, padding, block_id, layer_id, kernel_init=orthogonal()):
    prefix = f'block_{block_id}_layer_{layer_id}_'
    x = layers.Conv3DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
                               kernel_initializer=kernel_init, name=prefix+'de-conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(0.2, name=prefix+'drop')(x)
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    return x
    
import voxelmorph as vxm
def OneShotModel(input_shape, final_activation='softmax'):
    inputs = layers.Input(shape=input_shape)
    
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    

    unet_model = vxm.networks.Unet(inshape=input_shape, nb_features=(en, de), batch_norm=False,
               nb_conv_per_level=2,
               final_activation_function=final_activation)

    return unet_model
    
def noiseModel(input_shape, final_activation=None):
    inputs = layers.Input(shape=input_shape)
    
    en = [16, 16, 32, 32, 64, 64, 128]
    de = [128, 64, 64, 32, 32, 16, 16, 1]
    
    x = inputs

    unet_model = vxm.networks.Unet(inshape=input_shape, nb_features=(en, de), batch_norm=False,
               nb_conv_per_level=2,
               final_activation_function=final_activation)

    # # Encoder
    # for i, filters in enumerate(en):
    #     x = ConvolutionLayer3D(x, filters, (3, 3, 3), strides=(1, 1, 1), padding='same', block_id=i+1, layer_id=1)
    #     x = ConvolutionLayer3D(x, filters, (3, 3, 3), strides=(1, 1, 1), padding='same', block_id=i+1, layer_id=2)
    #     if i < len(en) - 1:
    #         x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same', name=f'block_{i+1}_pool')(x)
    
    # # Decoder
    # for i, filters in enumerate(de):
    #     if i < len(de) - 1:
    #         x = layers.UpSampling3D(size=(2, 2, 2), name=f'block_{len(en)+i+1}_upsample')(x)
    #     x = DeconvolutionLayer3D(x, filters, (3, 3, 3), strides=(1, 1, 1), padding='same', block_id=len(en) + i + 1, layer_id=1)
    #     x = DeconvolutionLayer3D(x, filters, (3, 3, 3), strides=(1, 1, 1), padding='same', block_id=len(en) + i + 1, layer_id=2)
    
    # # Final output layer with optional activation
    # x = layers.Conv3D(1, (1, 1, 1), padding='same', activation=final_activation, name='final_conv')(x)
    # input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))

    # generated_img_norm = min_max_norm(generated_img)
            
    # segmentation = unet_model(generated_img_norm)

    # noise_model = models.Model(inputs=inputs, outputs=x)
    return unet_model