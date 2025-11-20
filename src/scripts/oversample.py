from pathlib import Path
import shutil

DATASET_ROOT = Path("datasets")
train_images = DATASET_ROOT / "train" / "images"
train_labels = DATASET_ROOT / "train" / "labels"

balanced_images = DATASET_ROOT / "train_balanced" / "images"
balanced_labels = DATASET_ROOT / "train_balanced" / "labels"
balanced_images.mkdir(parents=True, exist_ok=True)
balanced_labels.mkdir(parents=True, exist_ok=True)

for label_path in train_labels.glob("*.txt"):
    img_name = label_path.stem + ".jpg"
    src_img = train_images / img_name
    if not src_img.exists():
        continue

    shutil.copy2(src_img, balanced_images / src_img.name)
    shutil.copy2(label_path, balanced_labels / label_path.name)

minority_class_id = 5
extra_repeats = 3

for label_path in train_labels.glob("*.txt"):
    with open(label_path, "r") as f:
        lines = f.readlines()

    has_minority = any(line.strip().startswith(str(minority_class_id)) for line in lines)
    if not has_minority:
        continue

    img_name = label_path.stem + ".jpg"
    src_img = train_images / img_name
    if not src_img.exists():
        continue

    for k in range(extra_repeats):
        new_img_name = f"{label_path.stem}_yardrep{k}.jpg"
        new_lbl_name = f"{label_path.stem}_yardrep{k}.txt"
        shutil.copy2(src_img, balanced_images / new_img_name)
        shutil.copy2(label_path, balanced_labels / new_lbl_name)

print("Balanced train set created at:", balanced_images.parent)
