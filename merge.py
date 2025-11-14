from pathlib import Path
import pathlib
from tempfile import TemporaryDirectory
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

# Import the patch to fix extension dtype issues
# import gs_data_io.patch_aggregate  # noqa: F401

datasets_root_path = "/home/yaxiong-zhao/workspace/datasets/20251105"
dataset_paths = list(pathlib.Path(datasets_root_path).glob("*"))
datasets = [LeRobotDataset(repo_id="", root=str(dataset_path)) for dataset_path in dataset_paths]
with TemporaryDirectory(prefix="test_") as temp_dir:
    output_dir = Path(temp_dir) / "merged"
    merge_datasets(datasets, output_repo_id="merged", output_dir=output_dir)
