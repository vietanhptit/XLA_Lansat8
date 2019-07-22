import pickle
from PIL import Image
import numpy
from math import sqrt

class_tree = 1
class_non_tree = 0

# Load model file
print("Load model file.....")
loaded_svm_model = pickle.load(open('model/tree_detect_model_8bit.sav', 'rb'))
# Red band
print("Read red band....")
red_band_file = 'data_test/LC08_126052_B4_crop_test.TIF'
red_band_tif = Image.open(red_band_file)
red_band_arr = numpy.array(red_band_tif).ravel()

result_arr = numpy.zeros((red_band_arr.size, 3), dtype=numpy.uint8)
print(red_band_arr.size)

# NIR band 1
print("Read NIR1 band....")
nir_band_file_1 = 'data_test/LC08_126052_B5_crop_test.TIF'
nir_band_tif_1 = Image.open(nir_band_file_1)
nir_band_arr_1 = numpy.array(nir_band_tif_1).ravel()


print("Predicting.....")
count = 1
for index, (red, nir_1) in enumerate(
        zip(red_band_arr,
            nir_band_arr_1)):
    count += 1
    predict = loaded_svm_model.predict([[red, nir_1]])
    if predict == class_tree:
        result_arr[index] = [0, 255, 0]
    else:
        result_arr[index] = [255, 0, 0]

image_height = image_width = int(sqrt(red_band_arr.size))
result_img_arr = result_arr.reshape((image_height, image_width, 3))
img = Image.fromarray(result_img_arr, 'RGB')
img.save('results/result_ndvi_predict.png')
img.show()
