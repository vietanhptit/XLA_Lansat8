from PIL import Image
import numpy as np

image_predict_path = "/home/boss/AI/project/XLA_LANDSAT8/PycharmCode/WaterRecognition/results/result_ndvi_predict.png"
image_predict = Image.open(image_predict_path)
image_predict_arr = np.array(image_predict)

image_kmean_path = "/home/boss/AI/project/XLA_LANDSAT8/PycharmCode/WaterRecognition/output/ndvi_result_kmean.TIF"
image_kmean = Image.open(image_kmean_path)
image_kmean_arr = np.array(image_kmean)

# print((image_kmean_arr==image_predict_arr).all())
print(np.array_equal(image_predict_arr,image_kmean_arr))
np.mean