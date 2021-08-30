import numpy as np
from skimage import transform
import tensorflow as tf

def normalize(input):
    input = np.float32(input)
    xmin = np.amin(input)
    xmax = np.amax(input)
    b = 1.  # max value (17375)
    a = 0.  # min value (0)
    if (xmax - xmin) == 0:
        out = input
    else:
        out = a+(b-a)*(input-xmin)/(xmax-xmin)
    return out


def rescale_img(img, img_size):
    contrast = np.random.uniform(low=0.7, high=1.3)
    brightness = np.random.normal(0, 0.1, 1)
    img = img*contrast + brightness
    r_img = transform.resize(img, img_size, anti_aliasing=True)
    return normalize(r_img).reshape(1, 256, 256, 1)


def rescale_imgs(img, img_size):
    noise = np.random.normal(0, 0.1, (256, 256))
    contrast = np.random.uniform(low=0.7, high=1.3)
    brightness = np.random.normal(0, 0.1, 1)
    img = img*contrast + brightness
    r_img0 = transform.resize(img, img_size, anti_aliasing=True)
    r_img0 = normalize(r_img0.reshape(1, r_img0.shape[0], r_img0.shape[1], 1))
    r_img1 = transform.resize(img, (img_size[0]//2, img_size[1]//2), anti_aliasing=True)
    r_img1 = normalize(r_img1.reshape(1, r_img1.shape[0], r_img1.shape[1], 1))
    r_img2 = transform.resize(img, (img_size[0]//4, img_size[1]//4), anti_aliasing=True)
    r_img2 = normalize(r_img2.reshape(1, r_img2.shape[0], r_img2.shape[1], 1))
    r_img3 = transform.resize(img, (img_size[0]//8, img_size[1]//8), anti_aliasing=True)
    r_img3 = normalize(r_img3.reshape(1, r_img3.shape[0], r_img3.shape[1], 1))
    r_img4 = transform.resize(img, (img_size[0]//16, img_size[1]//16), anti_aliasing=True)
    r_img4 = normalize(r_img4.reshape(1, r_img4.shape[0], r_img4.shape[1], 1))
    r_img5 = transform.resize(img, (img_size[0]//32, img_size[1]//32), anti_aliasing=True)
    r_img5 = normalize(r_img5.reshape(1, r_img5.shape[0], r_img5.shape[1], 1))
    r_img6 = transform.resize(img, (img_size[0]//64, img_size[1]//64), anti_aliasing=True)
    r_img6 = normalize(r_img6.reshape(1, r_img6.shape[0], r_img6.shape[1], 1))

    zero_flow0 = np.float32(np.zeros(r_img0.shape))
    zero_flow1 = np.float32(np.zeros(r_img1.shape))
    zero_flow2 = np.float32(np.zeros(r_img2.shape))
    zero_flow3 = np.float32(np.zeros(r_img3.shape))
    zero_flow4 = np.float32(np.zeros(r_img4.shape))
    zero_flow5 = np.float32(np.zeros(r_img5.shape))
    zero_flow6 = np.float32(np.zeros(r_img6.shape))
    return [r_img0, r_img1, r_img2, r_img3, r_img4, r_img5, r_img6,
            zero_flow0, zero_flow1, zero_flow2, zero_flow3, zero_flow4, zero_flow5, zero_flow6]


def rescale_tensors(img):
    img0 = img
    img1 = tf.image.resize_bicubic(img, [128, 128])
    img2 = tf.image.resize_bicubic(img, [64, 64])
    img3 = tf.image.resize_bicubic(img, [32, 32])
    img4 = tf.image.resize_bicubic(img, [16, 16])
    img5 = tf.image.resize_bicubic(img, [8, 8])
    img6 = tf.image.resize_bicubic(img, [4, 4])
    return [img0, img1, img2, img3, img4, img5, img6]

def warp_tensors(img, flow):
    warped = tf.contrib.image.dense_image_warp(img, flow, name='dense_image_warp')
    return warped