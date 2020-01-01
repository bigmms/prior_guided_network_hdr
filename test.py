from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from keras.models import Model
from data_loader import DataLoader
import numpy as np
from my_unet import ctom_unet
from skimage.color import ycbcr2rgb
from skimage.io import imsave
import argparse


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1024
        self.img_cols = 1024
        self.channels = 3*2

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.data_loader = DataLoader(dataset_name='HDR', expect_length=self.img_rows)

        self.generator = self.build_generator()

    def build_generator(self):
        my_model = ctom_unet(self.img_shape,
                               num_classes=3,
                               use_batch_norm=True,
                               upsample_mode='deconv',  # 'deconv' or 'simple'
                               use_dropout_on_upsampling=False,
                               dropout=0.3,
                               dropout_change_per_layer=0.0,
                               filters=16,
                               num_layers=6,
                               output_activation='tanh')
        return Model(my_model.input, my_model.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./saved_model/generator_0058.h5')
    parser.add_argument('--testing_dir', type=str, default='./HDR/')
    args = parser.parse_args()

    model_weight = args.model
    testing_dir = args.testing_dir

    dcgan = DCGAN()
    dcgan.generator.load_weights(model_weight)

    for ii, (new_size, raw_fused_cbcr, raw_imgs_luma_1024, raw_imgs_luma_512, raw_imgs_luma_256, raw_imgs_luma_128, raw_imgs_luma_64,raw_imgs_luma_32,raw_imgs_detail_1024, raw_imgs_detail_512, raw_imgs_detail_256, raw_imgs_detail_128, raw_imgs_detail_64,\
                  raw_imgs_detail_32) in enumerate(dcgan.data_loader.load_testing_data(testing_dir)):
        fake_imgs = dcgan.generator.predict(
            [raw_imgs_luma_1024,
             raw_imgs_detail_32,
             raw_imgs_detail_64,
             raw_imgs_detail_128,
             raw_imgs_detail_256,
             raw_imgs_detail_512,
             raw_imgs_detail_1024,
             raw_imgs_luma_32,
             raw_imgs_luma_64,
             raw_imgs_luma_128,
             raw_imgs_luma_256,
             raw_imgs_luma_512])

        img0 = fake_imgs[0][0, :new_size[1], :new_size[0], :]
        img_cbcr = raw_fused_cbcr[:new_size[1], :new_size[0], :]
        img0 = (0.5 * img0 + 0.5) * 255
        res0 = np.concatenate([img0, img_cbcr], axis=2)
        res_rgb0 = ycbcr2rgb(res0)
        output_RGB = np.clip(res_rgb0, 0, 1)

        imsave("./Results/%06d_fused.png" % (ii), output_RGB)



