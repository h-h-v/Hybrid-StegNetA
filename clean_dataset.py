from PIL import Image
import os

dataset_path = "dataset"

bad_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            img = Image.open(file_path)
            img.verify()   # verify image integrity
        except Exception:
            bad_files.append(file_path)

print("Invalid / Corrupted files:")
for f in bad_files:
    print(f)