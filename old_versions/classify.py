import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


batch_size = 32
img_height = 180
img_width = 180
DATA_DIR = "chest_Xray/"
TEST = "test"
TRAIN = "train"
VAL = "val"
NORMAL = "NORMAL"
PNEUMONIA = "PNEUMONIA"

# Créer un ensemble de données
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR +  TRAIN,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR + VAL,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)
# print(class_names)

# Configurer l'ensemble de données pour les performances
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardiser les données
normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# Augmentation des données
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# cree le modele
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Résumé du modèle
model.summary()

# Create a callback allowing to save the best performing model
checkpoint = ModelCheckpoint("saved_model.model.h5", monitor='val_loss', verbose=1, save_best_only=True, min_delta = .002)

# Former le modèle
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[checkpoint]
)

# Visualisez les résultats de l'entraînement
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# test 
img_pneuno1 = DATA_DIR + TEST + "/" + PNEUMONIA + "/person1662_virus_2875.jpeg"
img_pneuno2 = DATA_DIR + TEST + "/" + PNEUMONIA + "/person157_bacteria_739.jpeg"
img_pneuno3 = DATA_DIR + TEST + "/" + PNEUMONIA + "/person94_bacteria_456.jpeg"
img_pneuno4 = DATA_DIR + TEST + "/" + PNEUMONIA + "/pneumonia-right-middle-lobe-1.png"


img_sain1 = DATA_DIR + TEST + "/" + NORMAL + "/IM-0003-0001.jpeg"
img_sain2 = DATA_DIR + TEST + "/" + NORMAL + "/NORMAL2-IM-0051-0001.jpeg"
img_sain3 = DATA_DIR + TEST + "/" + NORMAL + "/NORMAL2-IM-0376-0001.jpeg"
img_sain4 = DATA_DIR + TEST + "/" + NORMAL + "/IM-0043-0001.jpeg"

list_test = [img_pneuno1, img_pneuno2, img_pneuno3, img_pneuno4, img_sain1, img_sain2, img_sain3, img_sain4]

for test in list_test:
  img = tf.keras.utils.load_img(
      test, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )