from PIL import Image
import numpy

label_file_path = 'LC08_127045/LC08_127045_B5.TIF'
label_file_name = 'LC08_127045/crop_for_ndvi/LC08_127045_B5_crop.TIF'

label_tif = Image.open(label_file_path)
print(label_tif.size)
label_array = numpy.array(label_tif)
print(label_array.shape)
crop_label_array = label_array[3200:3700, 4200:4700]

crop_img = Image.fromarray(crop_label_array)
crop_img.save(label_file_name)