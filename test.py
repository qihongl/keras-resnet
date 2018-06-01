import pytest
from keras import backend as K
from keras.preprocessing import image
from resnet import ResnetBuilder
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = ResnetBuilder.build_resnet_18((3, 224, 224), 100)
K.set_image_dim_ordering('tf')
model.compile(loss="categorical_crossentropy", optimizer="sgd")
# _test_model_compile(model)


# load an image
img_dir = 'images'
img_name = 'stanford.jpg'
img_path = os.path.join(img_dir, img_name)
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

np.shape(x)

y_hat = model.predict(x)
np.shape(y_hat)
# model.summary()


import pandas as pd

pd.read_csv('resnet18_cifar10.csv')
