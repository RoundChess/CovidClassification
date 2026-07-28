import kagglehub
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import IPython.display as display
import ipywidgets.widgets as widgets
import numpy as np

train_path = kagglehub.dataset_download("plameneduardo/sarscov2-ctscan-dataset")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation"
)
classes = train_ds.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  print(f"Images batch shape: {images.shape}")
  print(f"Labels batch shape: {labels.shape}")
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classes[labels[i]])
    plt.axis("off")

SCALE = 1/127.5

model = tf.keras.Sequential(layers=[
    tf.keras.layers.Rescaling(SCALE, offset=-1),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

optimizer = tf.keras.optimizers.AdamW(gradient_accumulation_steps=4)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        keras.metrics.BinaryAccuracy(),
    ]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

train_binary_accuracy = model.history.history["binary_accuracy"]
val_binary_accuracy = model.history.history["val_binary_accuracy"]

plt.plot(train_binary_accuracy, label="Training accuracy")
plt.plot(val_binary_accuracy, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(np.arange(0, 25, 5))
plt.yticks(np.arange(0.5, 1.1, 0.1))
plt.legend()

uploader = widgets.FileUpload(accept=".png", multiple=True, style={
    "button_color": "lightblue",
    "text_color": "black"
})


images = []
def load_images():
  for file in uploader.value.keys():
    image = tf.image.decode_png(uploader.value[file]["content"])
    image = tf.image.resize(image, (256, 256))
    images.append(image)

def show_images():
  plt.figure(figsize=(5, 5))
  for img in images:
    for i in range(len(images)):
      ax = plt.subplot(len(images), 1, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.axis("off")

confirm_dropdown = widgets.Dropdown(
  options=["Yes", "No"],
  description="Are these the correct images?",
  value=None,
  style={
    "description_width": "180px"
  }
)

def classify_images(images):
  for image in images:
    image = np.expand_dims(image, axis=0)
    if model(image) < 0.5:
      print("No covid.")
    elif model(image) > 0.5:
      print("Covid.")

def on_change(change):
  if change["type"] == "change" and change["name"] == "value" and change["new"] == "Yes":
    classify_images(images)

def on_upload(change):
  confirm_dropdown.observe(on_change, names="value", type="change")
  display.display(confirm_dropdown)
  load_images()
  show_images()

uploader.observe(on_upload, names="value", type="change")
display.display(uploader)
