from keras.layers import merge, Lambda, Convolution2D, Deconvolution2D, MaxPooling2D, Input, Reshape, Permute, ZeroPadding2D, UpSampling2D, Cropping2D
from keras.layers.core import Activation, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16


class FullyConvolutionalNetwork():
    def __init__(self, batchsize=1, img_height=224, img_width=224, FCN_CLASSES=21):
        self.batchsize = batchsize
        self.img_height = img_height
        self.img_width = img_width
        self.FCN_CLASSES = FCN_CLASSES
        self.vgg16 = VGG16(include_top=False,
						   weights='imagenet',
  						   input_tensor=None,
						   input_shape=(self.img_height, self.img_width, 3))

	def create_model(self, train_flag=True):
		input_image = Input(shape=(self.img_height, self.img_width, 3))
		conv1_1 = self.vgg16.layers[1](input_image)
		conv1_2 = self.vgg16.layers[2](conv1_1)
		pool1 = self.vgg16.layers[3](conv1_2)
		conv2_1 = self.vgg16.layers[4](pool1) 
		conv2_2 = self.vgg16.layers[5](conv2_1)
		pool2 = self.vgg16.layers[6](conv2_2) 
		conv3_1 = self.vgg16.layers[7](pool2) 
		conv3_2 = self.vgg16.layers[8](conv3_1)
		conv3_3 = self.vgg16.layers[9](conv3_2)
		pool3 = self.vgg16.layers[10](conv3_3) #(None, 28, 28, 256)
		conv4_1 = self.vgg16.layers[11](pool3)
		conv4_2 = self.vgg16.layers[12](conv4_1)
		conv4_3 = self.vgg16.layers[13](conv4_2)
		pool4 = self.vgg16.layers[14](conv4_3) #(None, 14, 14, 512)
		conv5_1 = self.vgg16.layers[15](pool4)
		conv5_2 = self.vgg16.layers[16](conv5_1)
		conv5_3 = self.vgg16.layers[17](conv5_2)
		pool5 = self.vgg16.layers[18](conv5_3) #(None, 7, 7, 512)

i		score_pool3 = Convolution2D(self.FCN_CLASSES, 1, 1, activation='relu')(pool3) #(None, 28, 28, FCN_CLASSES)

		upsample_pool4 = UpSampling2D((2,2))(pool4) #(None, 28, 28, 512)
		score_pool4 = Convolution2D(self.FCN_CLASSES, 4, 4, activation='relu', border_mode='same')(upsample_pool4) #(None, 28, 28, FCN_CLASSES)

		upsample_pool5 = UpSampling2D((4,4))(pool5) #(None, 28, 28, 512)
		score_pool5 = Convolution2D(self.FCN_CLASSES, 8, 8, activation='relu', border_mode='same')(upsample_pool5)
		merged = merge([score_pool3, score_pool4, score_pool5])
		upsample_final = UpSampling2D((8,8))(merged) #(None, 224, 224, FCN_CLASSES)
		score_final = Convolution2D(self.FCN_CLASSES, 16, 16, activation='relu', border_mode='same')(upsample_final) #(None, 224, 224, FCN_CLASSES)

		pred = score_final(
'''
		conv6 = Convolution2D(pool5, 4096, 7, 7)
		conv6 = Activation('relu')(conv6)
		drop6 = Dropout(0.5)(conv6)
		conv7 = Convolution2D(drop6, 4096, 1, 1)
		conv7 = Activation('relu')(conv7)
		drop7 = Dropout(0.5)(conv7) #(None, 
		upsample_final = UpSampling2D((4,4))(drop7) 
		score_final =  
'''
		
