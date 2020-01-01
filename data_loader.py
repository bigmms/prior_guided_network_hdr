import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
from glob import glob
from skimage.color import rgb2ycbcr
from fusion import fusion,cfusion


class DataLoader():
    def __init__(self, dataset_name='HDR', expect_length=1024):
        self.dataset_name = dataset_name
        self.expect_length = expect_length

    def get_new_img_size(self, img_shape, aspect_length):
        img_width = img_shape[1]
        img_height = img_shape[0]
        if img_width > img_height:
            img_height = int(aspect_length * img_height / img_width)
            img_width = self.expect_length
        else:
            img_width = int(aspect_length * img_width / img_height)
            img_height = self.expect_length
        return img_width, img_height

    def load_batch(self, is_testing=False, batch_size=1):
        data_type = "train" if not is_testing else "test"
        raw_folders = glob(
            'I:/Papers/HDR/HDR_pyramid/%s/%s/raw/*' % (self.dataset_name, data_type))
        target_folders = glob(
            'I:/Papers/HDR/HDR_pyramid/%s/%s/target/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(raw_folders) / batch_size)

        for i in range(self.n_batches - 1):
            raw_batch = raw_folders[i * batch_size:(i + 1) * batch_size]
            target_batch = target_folders[i * batch_size:(i + 1) * batch_size]
            raw_imgs = []
            target_imgs = []

            for ii in range(batch_size):
                raw = imread(raw_batch[ii])
                raw0 = raw[0:round(raw.shape[0] / 2), :, :]
                new_size = self.get_new_img_size(raw0.shape, self.expect_length)
                raw0_resize = resize(raw0, (new_size[1], new_size[0]), preserve_range=True, anti_aliasing=False)
                black = np.zeros((self.expect_length, self.expect_length, 3))
                black[:new_size[1], :new_size[0], :] = raw0_resize
                raw0_new = black


                raw1 = raw[round(raw.shape[0] / 2):, :, :]
                raw1_resize = resize(raw1, (new_size[1], new_size[0]), preserve_range=True, anti_aliasing=False)
                black = np.zeros((self.expect_length, self.expect_length, 3))
                black[:new_size[1], :new_size[0], :] = raw1_resize
                raw1_new = black


                raw_fused = np.uint8(np.clip(cfusion(raw0_new, raw1_new) * 255, 0, 255))
                detail, luma = fusion(raw_fused,multi= True)

                raw_img = np.concatenate([raw0_new, raw1_new],axis=2)
                raw_img = raw_img[:, :, (0, 3, 1, 4, 2, 5)]
                raw_imgs.append(raw_img)

                target_img = imread(target_batch[ii])
                target_img_resize = resize(target_img, (new_size[1], new_size[0]), preserve_range=True, anti_aliasing=False)
                black = np.zeros((self.expect_length, self.expect_length, 3))
                black[:new_size[1], :new_size[0], :] = target_img_resize
                target_img_new = black
                target_img_y = rgb2ycbcr(np.uint8(target_img_new))[:, :, 0:1]
                target_imgs.append(target_img_y)

            raw_imgs_luma_1024 = np.expand_dims(np.array(luma[0])*2 - 1, axis=0)
            raw_imgs_luma_512 = np.expand_dims(np.array(luma[1])*2 - 1, axis=0)
            raw_imgs_luma_256 = np.expand_dims(np.array(luma[2])*2 - 1, axis=0)
            raw_imgs_luma_128 = np.expand_dims(np.array(luma[3])*2 - 1, axis=0)
            raw_imgs_luma_64 = np.expand_dims(np.array(luma[4])*2 - 1, axis=0)
            raw_imgs_luma_32 = np.expand_dims(np.array(luma[5])*2 - 1, axis=0)

            raw_imgs_detail_1024 = np.expand_dims(np.array(detail[0]), axis=0)
            raw_imgs_detail_512 = np.expand_dims(np.array(detail[1]), axis=0)
            raw_imgs_detail_256 = np.expand_dims(np.array(detail[2]), axis=0)
            raw_imgs_detail_128 = np.expand_dims(np.array(detail[3]), axis=0)
            raw_imgs_detail_64 = np.expand_dims(np.array(detail[4]), axis=0)
            raw_imgs_detail_32 = np.expand_dims(np.array(detail[5]), axis=0)


            yield np.array(target_imgs) / 127.5 - 1., np.array(raw_imgs) / 127.5 - 1., raw_imgs_luma_512, raw_imgs_luma_256, raw_imgs_luma_128, raw_imgs_luma_64, raw_imgs_luma_32, raw_imgs_detail_1024, raw_imgs_detail_512, raw_imgs_detail_256, raw_imgs_detail_128, raw_imgs_detail_64, raw_imgs_detail_32

    def load_data(self, batch_size=1, is_testing=True):
        data_type = "train" if not is_testing else "test_sum"

        raw_folders = glob(
            'I:/Papers/HDR/HDR_pyramid/%s/%s/raw/*' % (self.dataset_name, data_type))

        idx = (len(raw_folders) * np.random.rand(batch_size)).astype(int)[0] - 1

        raw = imread(raw_folders[idx])

        raw_imgs = []

        raw0 = raw[0:round(raw.shape[0] / 2), :, :]
        new_size = self.get_new_img_size(raw0.shape, self.expect_length)
        raw0_resize = resize(raw0, (new_size[1], new_size[0]), preserve_range=True, anti_aliasing=False)
        black = np.zeros((self.expect_length, self.expect_length, 3))
        black[:new_size[1], :new_size[0], :] = raw0_resize
        raw0_new = black



        raw1 = raw[round(raw.shape[0] / 2):, :, :]
        raw1_resize = resize(raw1, (new_size[1], new_size[0]), preserve_range=True, anti_aliasing=False)
        black = np.zeros((self.expect_length, self.expect_length, 3))
        black[:new_size[1], :new_size[0], :] = raw1_resize
        raw1_new = black


        raw_fused = np.uint8(np.clip(cfusion(raw0_new,raw1_new)*255,0,255))
        raw_fused_cbcr = rgb2ycbcr(raw_fused)[:, :, 1:3]
        detail, luma = fusion(raw_fused, multi=True)

        raw_imgs_luma_1024 = np.expand_dims(np.array(luma[0])*2 - 1, axis=0)
        raw_imgs_luma_512 = np.expand_dims(np.array(luma[1])*2 - 1, axis=0)
        raw_imgs_luma_256 = np.expand_dims(np.array(luma[2])*2 - 1, axis=0)
        raw_imgs_luma_128 = np.expand_dims(np.array(luma[3])*2 - 1, axis=0)
        raw_imgs_luma_64 = np.expand_dims(np.array(luma[4])*2 - 1, axis=0)
        raw_imgs_luma_32 = np.expand_dims(np.array(luma[5])*2 - 1, axis=0)

        raw_imgs_detail_1024 = np.expand_dims(np.array(detail[0]), axis=0)
        raw_imgs_detail_512 = np.expand_dims(np.array(detail[1]), axis=0)
        raw_imgs_detail_256 = np.expand_dims(np.array(detail[2]), axis=0)
        raw_imgs_detail_128 = np.expand_dims(np.array(detail[3]), axis=0)
        raw_imgs_detail_64 = np.expand_dims(np.array(detail[4]), axis=0)
        raw_imgs_detail_32 = np.expand_dims(np.array(detail[5]), axis=0)

        raw_img = np.concatenate([raw0_new, raw1_new],axis=2)
        raw_imgs.append(raw_img)
        # inputs = np.expand_dims(np.array(raw_imgs) / 127.5 - 1., axis=-1)

        return new_size, raw_fused_cbcr, np.array(raw_imgs) / 127.5 - 1., raw_imgs_luma_512, raw_imgs_luma_256, raw_imgs_luma_128, raw_imgs_luma_64, raw_imgs_luma_32, raw_imgs_detail_1024, raw_imgs_detail_512, raw_imgs_detail_256, raw_imgs_detail_128, raw_imgs_detail_64, raw_imgs_detail_32

    def load_testing_data(self, raw_dir):

        raw_folders = glob(raw_dir + '*')

        for idx in range(len(raw_folders)):
            raw_imgs = []
            raw = imread(raw_folders[idx])

            raw0 = raw[0:round(raw.shape[0] / 2), :, :]
            new_size = self.get_new_img_size(raw0.shape, self.expect_length)
            raw0_resize = resize(raw0, (new_size[1], new_size[0]), preserve_range=True, anti_aliasing=False)
            black = np.zeros((self.expect_length, self.expect_length, 3))
            black[:new_size[1], :new_size[0], :] = raw0_resize
            raw0_new = black
            # raw0_y = rgb2ycbcr(np.uint8(raw0_new))[:, :, 0:1]

            raw1 = raw[round(raw.shape[0] / 2):, :, :]
            raw1_resize = resize(raw1, (new_size[1], new_size[0]), preserve_range=True, anti_aliasing=False)
            black = np.zeros((self.expect_length, self.expect_length, 3))
            black[:new_size[1], :new_size[0], :] = raw1_resize
            raw1_new = black
            # raw1_y = rgb2ycbcr(np.uint8(raw1_new))[:, :, 0:1]

            raw_fused = np.uint8(np.clip(cfusion(raw0_new, raw1_new) * 255, 0, 255))
            raw_fused_cbcr = rgb2ycbcr(raw_fused)[:, :, 1:3]
            # raw_fused_y = rgb2ycbcr(raw_fused)[:, :, 0]
            detail, luma = fusion(raw_fused, multi=True)

            raw_imgs_luma_1024 = np.expand_dims(np.array(luma[0])*2 - 1, axis=0)
            raw_imgs_luma_512 = np.expand_dims(np.array(luma[1])*2 - 1, axis=0)
            raw_imgs_luma_256 = np.expand_dims(np.array(luma[2])*2 - 1, axis=0)
            raw_imgs_luma_128 = np.expand_dims(np.array(luma[3])*2 - 1, axis=0)
            raw_imgs_luma_64 = np.expand_dims(np.array(luma[4])*2 - 1, axis=0)
            raw_imgs_luma_32 = np.expand_dims(np.array(luma[5])*2 - 1, axis=0)

            raw_imgs_detail_1024 = np.expand_dims(np.array(detail[0]), axis=0)
            raw_imgs_detail_512 = np.expand_dims(np.array(detail[1]), axis=0)
            raw_imgs_detail_256 = np.expand_dims(np.array(detail[2]), axis=0)
            raw_imgs_detail_128 = np.expand_dims(np.array(detail[3]), axis=0)
            raw_imgs_detail_64 = np.expand_dims(np.array(detail[4]), axis=0)
            raw_imgs_detail_32 = np.expand_dims(np.array(detail[5]), axis=0)


            raw_img = np.concatenate([raw0_new, raw1_new], axis=2)
            raw_imgs.append(raw_img)

            yield new_size, raw_fused_cbcr, np.array(
                raw_imgs) / 127.5 - 1., raw_imgs_luma_512, raw_imgs_luma_256, raw_imgs_luma_128, raw_imgs_luma_64, raw_imgs_luma_32, raw_imgs_detail_1024, raw_imgs_detail_512, raw_imgs_detail_256, raw_imgs_detail_128, raw_imgs_detail_64, raw_imgs_detail_32,
