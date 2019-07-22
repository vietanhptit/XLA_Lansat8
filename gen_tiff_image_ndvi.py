import cv2
import numpy as np
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from sklearn import cluster


def prepimage():  # prepare the image stack
    b3 = cv2.imread("data_train/LC08_126052_B3_crop.TIF", 0)  # load red
    b4 = cv2.imread("data_train/LC08_126052_B4_crop.TIF", 0)  # load band 5 - near infrared
    b5 = cv2.imread("data_train/LC08_126052_B5_crop.TIF", 0)

    X3 = b3.reshape((-1, 1))
    X4 = b4.reshape((-1, 1))
    X5 = b5.reshape((-1, 1))

    X = np.concatenate((X5, X4, X3), axis=1)

    ndvi = np.true_divide((b5 - b4), (b5 + b4))  # ndvi
    rgb = np.dstack((b5, b4, b3))  # combine into ordered stack
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)

    rgb_big = skimage.transform.resize(rgb, output_shape=(b5.shape[0], b5.shape[1], 3), order=3, mode='constant',
                                       cval=0.0)  # resize the rgb composite to match the chromatic
    ndvi_big = skimage.transform.resize(ndvi, output_shape=(b5.shape[0], b5.shape[1]), order=3, mode='constant',
                                        cval=0.0)  # resize ndvi to match the rgb

    return rgb, ndvi_big, X

rgb, ndvi, X = prepimage()

# cv2.imshow('ssd', ndvi)
# cv2.waitKey(0)

#kmean ndvi_image
k_means = cluster.KMeans(n_clusters=2)
ndvi_big = ndvi.reshape((rgb.shape[0]*rgb.shape[1],1))
k_means.fit(ndvi_big)

#merge RGB
ndvi_pan_r = np.zeros((rgb.shape[0]*rgb.shape[1]), dtype="uint8")
ndvi_pan_g = np.zeros((rgb.shape[0]*rgb.shape[1]), dtype="uint8")
ndvi_pan_b = np.zeros((rgb.shape[0]*rgb.shape[1]), dtype="uint8")
ndvi_ap = []
X_cluster = k_means.labels_
# binary_image = np.zeros((rgb.shape[0]*rgb.shape[1],1), dtype=np.int32)
print(type(X_cluster[2]))
for i in range(len(X_cluster)):
    if X_cluster[i] == 0:
        ndvi_pan_r[i] = 0
        ndvi_pan_g[i] = 255
        ndvi_pan_b[i] = 0
        # binary_image[i] = 0
    else:
        ndvi_pan_r[i] = 255
        ndvi_pan_g[i] = 0
        ndvi_pan_b[i] = 0
        # binary_image[i] = 255
ndvi_pan_r = ndvi_pan_r.reshape((rgb.shape[0],rgb.shape[1]))
ndvi_pan_g = ndvi_pan_g.reshape((rgb.shape[0],rgb.shape[1]))
ndvi_pan_b = ndvi_pan_b.reshape((rgb.shape[0],rgb.shape[1]))
r = Image.fromarray(ndvi_pan_r, mode=None)
g = Image.fromarray(ndvi_pan_g, mode=None)
b = Image.fromarray(ndvi_pan_b, mode=None)
imga = Image.merge("RGB", (r,g,b))
imga.save("./output/ndvi_result_kmean.TIF")
imga.show()