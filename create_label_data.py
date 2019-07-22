from PIL import Image
import numpy
import csv

class_tree = 1
class_non_tree = 0


image_label_path = './output/ndvi_result_kmean.TIF'

image_label = Image.open(image_label_path)

height, width = image_label.size

label_array = numpy.array(image_label).reshape(width*height, 3)

# CSV file

csv_file = "./data_train/data_training_ndvi_label.csv"

with open(csv_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for row in label_array:
        if (row == [0, 255, 0]).all():
            writer.writerow([1])
        else:
            writer.writerow([0])