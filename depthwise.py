import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D

# model
inputs = Input(shape=(4, 4, 3))
x = DepthwiseConv2D((3, 3), strides=(
    1, 1), depth_multiplier=1, padding='same')(inputs)
model = Model(inputs, x)
model.load_weights('model.h5')
print(model.summary())

# data
input_x = np.load('input_x.npy')
output_x = np.load('output_x.npy')
o = model.predict(input_x)

print(np.allclose(output_x, o))
