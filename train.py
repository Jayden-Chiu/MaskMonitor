from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import os


# initialize initial learning rate for Adam optimizer, epochs, and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 64

# initialize target size for images to use in MobileNetV2
target_size = 224

# get images from dataset directory
print("Loading images from dataset...")
image_paths = list(paths.list_images("dataset"))
data = []
labels = []

for path in image_paths:
	# get label from directory
	label = path.split(os.path.sep)[-3]

	# print(label)
	# preprocess images 
	image = load_img(path, target_size=(target_size, target_size))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update data and label lists
	data.append(image)
	labels.append(label)

# convert the data and labels to np arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# one-hot encode labels
enc = LabelEncoder()
labels = enc.fit_transform(labels)
labels = to_categorical(labels)

# print(labels)

# split data into default 75% testing 25% training and shuffle
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=25,
	zoom_range=0.2,
	width_shift_range=0.25,
	height_shift_range=0.25,
	shear_range=0.2,
	horizontal_flip=True,
	fill_mode="nearest")

input_tensor = Input(shape=(target_size, target_size, 3))

# load MobileNetV2
base_model = MobileNetV2(
	weights="imagenet", 
	include_top=False,
	input_tensor=input_tensor,
	input_shape=(target_size, target_size, 3),
	pooling='avg')

for layer in base_model.layers:
	layer.trainable = False # trainable set to false to freeze layers for fine tuning

op = Dense(200, activation="relu")(base_model.output)
op = Dropout(0.5)(op)
output_tensor = Dense(3, activation="softmax")(op)

model = Model(inputs=base_model.input, outputs=output_tensor)

# compile model
print("Compiling model...")
optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) # Adam optimizer
model.compile(
	loss="categorical_crossentropy", 
	optimizer=optimizer,
	metrics=["categorical_accuracy"])

# train the head of the network
print("Training head...")
H = model.fit(
	aug.flow(x_train, y_train, batch_size=BS),
	steps_per_epoch=len(x_train) // BS,
	validation_data=(x_test, y_test),
	validation_steps=len(x_test) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("Evaluating network...")
predIdxs = model.predict(x_test, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(y_test.argmax(axis=1), predIdxs,
	target_names=enc.classes_))

# save model to model directory
print("Saving model...")
model.save("detector.model", save_format="h5")

# plot the training loss and accuracy and save to plot directory
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")