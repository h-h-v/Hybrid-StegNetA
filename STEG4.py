import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("stegware_nn_model.h5")

# Class labels (must match training)
class_labels = {
    0: "cover",
    1: "stegware_family_1",
    2: "stegware_family_2",
    3: "stegware_family_3",
    4: "stegware_family_4"
}

IMG_SIZE = (128, 128)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)
    return class_labels[np.argmax(pred)]

# --------- EVALUATE FAMILY 4 ----------
folder_path = "dataset/stegware_family_4"
true_label = os.path.basename(folder_path)   # ✅ automatic & safe

total, correct = 0, 0

for file in os.listdir(folder_path):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    if file.startswith("._"):
        continue

    pred = predict_image(os.path.join(folder_path, file))
    total += 1
    if pred == true_label:
        correct += 1

accuracy = (correct / total) * 100

print("Class:", true_label)
print("Total images tested :", total)
print("Correct predictions :", correct)
print("Accuracy            :", round(accuracy, 2), "%")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = 128
BATCH_SIZE = 32

model = tf.keras.models.load_model("stegware_batchnorm_model.h5")

datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    "dataset/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(data)
y_pred = np.argmax(predictions, axis=1)
y_true = data.classes
labels = list(data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix – Stegware Classification")
plt.show()
