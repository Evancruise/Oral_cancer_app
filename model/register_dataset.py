from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

def get_semantic_dataset(split):
    return load_sem_seg(
        gt_root=f"model/dataset/{split}/annotations_mask",
        image_root=f"model/dataset/{split}/images",
        gt_ext="jpg",
        image_ext="png"
    )

def register_dataset():
    """
    Register all datasets
    """
    for d in ["train", "val"]:
        DatasetCatalog.register(
            f"oral_cancer_dataset_{d}", 
            lambda d=d: get_semantic_dataset(d)
        )
        MetadataCatalog.get(f"oral_cancer_dataset_{d}").set(
            stuff_classes=["background", "green", "yellow", "red"]
        )

if __name__ == "__main__":
    register_dataset()