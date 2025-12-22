# tf, keras imports
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras

import neurite_sandbox as nes
import neurite as ne

try:
    from tqdm.keras import TqdmCallback
except ImportError as e:
    print(e)

tf.keras.backend.clear_session()  # clear keras session

try:
    from tensorflow.keras.backend import set_session
except ImportError as e:
    print(e)

try:
    import tensorflow_addons as tfa
except ImportError as e:
    print(e)

try:
    print("tf device:", tf.test.gpu_device_name())
except ImportError as e:
    print(e)
