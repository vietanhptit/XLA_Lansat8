from PIL import Image
import numpy
import csv

# Red band
red_band_file = './data_train/LC08_126052_B4_crop.TIF'
red_band_tif = Image.open(red_band_file)
red_band_arr = numpy.array(red_band_tif).ravel()
# NIR band 1
nir_band_file_1 = './data_train/LC08_126052_B5_crop.TIF'
nir_band_tif_1 = Image.open(nir_band_file_1)
nir_band_arr_1 = numpy.array(nir_band_tif_1).ravel()


# CSV file

csv_file = "./data_train/data_training_data_set_ndvi.csv"

with open(csv_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for red, nir_1 in zip(red_band_arr, nir_band_arr_1):
        writer.writerow([red, nir_1])