import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("stegware_nn_model.h5")

class_labels = {
    0: "cover",
    1: "stegware_family_1",
    2: "stegware_family_2",
    3: "stegware_family_3",
    4: "stegware_family_4"
}

IMG_SIZE = (128, 128)

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(preds)
        confidence = np.max(preds)

        return class_labels[predicted_class], confidence

    except:
        return None, None


# --------- BATCH TESTING ----------
folder_path = "dataset/stegware_family_4"
true_label = "stegware_family_4"

total = 0
correct = 0

for file in os.listdir(folder_path):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    if file.startswith("._"):   # skip corrupted hidden files
        continue

    img_path = os.path.join(folder_path, file)
    pred_label, conf = predict_image(img_path)

    if pred_label is None:
        continue

    total += 1
    if pred_label == true_label:
        correct += 1

accuracy = (correct / total) * 100 if total > 0 else 0

print("Total images tested :", total)
print("Correct predictions :", correct)
print("Accuracy            :", round(accuracy, 2), "%")