import yaml
from pathlib import Path
from ultralytics import YOLO

def create_data_yaml(dataset_root: Path, yaml_path: Path):
    data = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",

        "names": {
            0: "car",
            1: "house",
            2: "road",
            3: "swimming pool",
            4: "tree",
            5: "yard",
        },
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"[INFO] Wrote data.yaml to: {yaml_path}")


def main():
    dataset_root = Path("datasets").resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    data_yaml = dataset_root / "data.yaml"
    create_data_yaml(dataset_root, data_yaml)

    # You can change to yolov8s-seg.pt / yolov8m-seg.pt if you have more GPU.
    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data=str(data_yaml),   # path to data.yaml
        imgsz=640,             # image size (you can try 832 or 1024 if GPU allows)
        epochs=100,            # number of epochs
        batch=16,              # batch size (reduce if you get OOM)
        workers=4,             # dataloader workers
        project="runs_aerial", # folder where results will be saved
        name="yolov8n_seg",    # experiment name
        pretrained=True,       # use pretrained weights
        verbose=True,
    )

    print("[INFO] Training finished.")
    print(f"[INFO] Best weights saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
