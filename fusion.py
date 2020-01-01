# import utils_fusion
import numpy as np
from skimage.transform import pyramid_gaussian, resize
import cv2
from scipy import signal
from cv2.ximgproc import guidedFilter
import matplotlib.pyplot as plt
from skimage.color import rgb2ycbcr, rgb2yuv



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def la_filter(mono):
    img_shape = mono.shape
    C = np.zeros(img_shape)
    t1 = list([[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]])
    # for i in range(0, img_shape[0]):
    #     for j in range(0, img_shape[1]):
    #         C[i, j] = abs(np.sum(mono[i:i + 3, j:j + 3] * t1))
    myj = signal.convolve2d(mono, t1, mode="same")
    return myj


def contrast(I,exposure_num,img_rows,img_cols):
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        mono = rgb2gray(I[i])
        C[:, :, i] = la_filter(mono)

    return C


def saturation(I,exposure_num,img_rows,img_cols):
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        R = I[i][:, :, 0]
        G = I[i][:, :, 1]
        B = I[i][:, :, 2]
        mu = (R + G + B) / 3
        C[:, :, i] = np.sqrt(((R - mu) ** 2 + (G - mu) ** 2 + (B - mu) ** 2) / 3)
    return C


def well_exposedness(I,exposure_num,img_rows,img_cols):
    sig = 0.2
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        R = np.exp(-.4 * (I[i][:, :, 0] - 0.5) ** 2 / sig ** 2)
        G = np.exp(-.4 * (I[i][:, :, 1] - 0.5) ** 2 / sig ** 2)
        B = np.exp(-.4 * (I[i][:, :, 2] - 0.5) ** 2 / sig ** 2)
        C[:, :, i] = R * G * B
    return C


def gaussian_pyramid(I,nlev,multi):
    pyr = []

    # for ii in range(0,nlev):
    #     temp = pyramid_gaussian(I, downscale=2)
    #     pyr.append(temp)
    for (i, resized) in enumerate(pyramid_gaussian(I, downscale=2, multichannel=multi)):
        if i == nlev:
            break
        pyr.append(resized)
    return pyr


def laplacian_pyramid(I,nlev, mult= True):
    pyr = []
    expand = []
    pyrg = gaussian_pyramid(I,nlev,multi= mult)
    for i in range(0, nlev-1):

        # expand_temp = cv2.resize(pyrg[i + 1], (pyrg[i].shape[1], pyrg[i].shape[0]))
        expand_temp = resize(pyrg[i + 1], (pyrg[i].shape[1], pyrg[i].shape[0]), preserve_range=True, anti_aliasing=False)
        temp = pyrg[i] - expand_temp
        expand.append(expand_temp)
        pyr.append(temp)
    pyr.append(pyrg[nlev-1])
    expand.append(pyrg[nlev-1])
    return pyr, expand


def reconstruct_laplacian_pyramid(pyr):
    nlev = len(pyr)
    R = pyr[nlev-1]
    for i in range(nlev-2,-1,-1):
        odd = pyr[i].shape
        R = pyr[i] + cv2.resize(R,(pyr[i].shape[1], pyr[i].shape[0]))

    return R


def Gaussian1D(cen,std,YX1):
    y = np.zeros((1,YX1))
    for i in range(0,YX1):
        y[0][i] = np.exp(-((i - cen)**2) / (2 * (std**2)))
    y = np.round(y * (YX1 - 1))
    return y


def fusion(ldr_ori, multi = True):

    # I_ldr_ori = (ldr_ori[0, :, :, 0:3], ldr_ori[0, :, :, 3:6], ldr_ori[0, :, :, 6:9], ldr_ori[0, :, :, 9:12], ldr_ori[0, :, :, 12:15],
    #              ldr_ori[0, :, :, 15:18], ldr_ori[0, :, :, 18:21], ldr_ori[0, :, :, 21:24], ldr_ori[0, :, :, 24:27])

    img_shape = ldr_ori.shape
    img_rows = img_shape[0]
    img_cols = img_shape[1]
    exposure_num = 9
    r = img_rows
    c = img_cols
    n = exposure_num
    beta = 2
    nlev = round(np.log(min(r, c)) / np.log(2)) - beta
    nlev = int(nlev)
    pyr, expand = laplacian_pyramid(ldr_ori, nlev,mult=multi)
    return pyr, expand

def cfusion(uexp, oexp):
    beta = 2
    vFrTh = 0.16
    RadPr = 3

    I = (uexp, oexp)
    r = uexp.shape[0]
    c = uexp.shape[1]
    n = 2
    nlev = round(np.log(min(r, c)) / np.log(2)) - beta
    nlev = int(nlev)
    RadFr = RadPr * (1 << (nlev - 1))

    W = np.ones((r, c, n))

    W = np.multiply(W, contrast(I, n, r, c))
    W = np.multiply(W, saturation(I, n, r, c))
    W = np.multiply(W, well_exposedness(I, n, r, c))

    W = W + 1e-12
    Norm = np.array([np.sum(W, 2), np.sum(W, 2)])
    Norm = Norm.swapaxes(0, 2)
    Norm = Norm.swapaxes(0, 1)
    W = W / Norm

    II = (uexp / 255.0, oexp / 255.0)

    pyr = gaussian_pyramid(np.zeros((r, c, 3)), nlev,multi = True)
    for i in range(0, n):
        pyrw = gaussian_pyramid(W[:, :, i], nlev, multi = False)
        pyri,content = laplacian_pyramid(II[i], nlev, mult = True)
        for ii in range(0, nlev):
            w = np.array([pyrw[ii], pyrw[ii], pyrw[ii]])
            w = w.swapaxes(0, 2)
            w = w.swapaxes(0, 1)
            pyr[ii] = pyr[ii] + w * pyri[ii]
    R = reconstruct_laplacian_pyramid(pyr)
    # R = cv2.cvtColor(R.astype(np.float32), cv2.COLOR_YCR_CB2BGR)
    # R = ycbcr2rgb(R)

    # R = R * 255
    return R