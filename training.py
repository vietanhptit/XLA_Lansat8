from sklearn.svm import LinearSVC
from numpy import genfromtxt
import pickle
import numpy

# Label file
label_file = "./data_train/data_training_ndvi_label.csv"
label_data = genfromtxt(label_file, delimiter=',', encoding="utf8")

number_of_points_data = label_data.shape[0]

label_data = label_data.reshape(number_of_points_data, 1)

# Data training file
data_training_file = "./data_train/data_training_data_set_ndvi.csv"
training_data = genfromtxt(data_training_file, delimiter=',', encoding="utf8")

new_arr = numpy.concatenate((training_data, label_data), axis=1)
numpy.random.shuffle(new_arr)

training_data = new_arr[:, :2]
label_data = new_arr[:, 2]

model = LinearSVC()

number_points_for_train = int(0.8*number_of_points_data)

model.fit(training_data[:number_points_for_train, ], label_data[:number_points_for_train, ])

y_predict = model.predict(training_data[number_points_for_train:, ])

print("Accurancy: %.2f %%" % (100 * numpy.mean(label_data[number_points_for_train:, ] == y_predict)))

# Save the model to disk
model_file_name = "model/tree_detect_model_8bit.sav"

pickle.dump(model, open(model_file_name, 'wb'))