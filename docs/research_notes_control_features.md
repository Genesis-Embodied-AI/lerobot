# Research Notes: Control Features in LeRobot Writer

## Overview

This document traces how control features like `ctrl.joint_angles` are handled in the LeRobot writer, from action creation to parquet serialization.

## Flow for `ctrl.joint_angles` Feature

### 1. Action to Tensor Dictionary Conversion

When an action (e.g., `Joint6DAction`) is created, it has a `tensors()` method that returns a dictionary with prefixed keys:

```python
def tensors(
    self,
) -> Mapping[
    str,
    AsTensor[
        Batch[Dim["horizon"]],  # noqa: F821
        AsTensor[Batch[Dim["arm"]], JointAngles6D | GripperWidth],  # noqa: F821
    ],
]:
    return {
        f"{ACTION_KEY_PREFIX}{JOINT_ANGLES_KEY}": self.joint_angles_tensor()
            .unsqueeze(0, "horizon")
            .nest(1),
        f"{ACTION_KEY_PREFIX}{GRIPPER_WIDTH_KEY}": self.gripper_width_tensor()
            .unsqueeze(0, "horizon")
            .nest(1),
    }
```

Where:
- `ACTION_KEY_PREFIX = "ctrl."`
- `JOINT_ANGLES_KEY = "joint_angles"`
- This produces the key `"ctrl.joint_angles"`

**Location**: `schemas/gs_schemas/actions/joint.py` (lines 99-115)

### 2. Feature Metadata Definition

In `_build_features()`, action features are built from the `ActionSpace`:

```python
# Add action features
for key, config in self.action_space.tensors.items():
    space = config.to_gymnasium_space()
    features[key] = LeRobotFeature(
        names=space.names,
        shape=config.tensor_decoder.tensor_shape(),
        dtype=space.dtype.name,
    )
```

For `"ctrl.joint_angles"`, this creates a `LeRobotFeature` with:
- `names`: e.g., `["right.j0", "right.j1", ..., "left.j0", ...]`
- `shape`: e.g., `(2, 7)` for 2 arms × 7 joints
- `dtype`: e.g., `"float32"`

**Location**: `data-io/gs_data_io/writers/lerobot.py` (lines 365-372)

### 3. Frame Data Preparation

In `_write_frame()`, actions are converted to numpy arrays:

```python
action_dict = {
    k: v.squeeze(0).tensor.cpu().numpy()  # Remove horizon dimension
    for k, v in action.tensors().items()
}
```

This produces `action_dict` with keys like `"ctrl.joint_angles"` containing numpy arrays (e.g., shape `(2, 7)`).

**Location**: `data-io/gs_data_io/writers/lerobot.py` (lines 337-340)

### 4. Frame Dictionary Assembly

The frame combines all data:

```python
frame = image_dict | state_dict | action_dict | {"task": task}
self._dataset.add_frame(frame)
```

The frame dict includes:
- `"ctrl.joint_angles"`: numpy array `(num_arms, num_joints)`
- `"ctrl.gripper_width"`: numpy array `(num_arms, 1)`
- Other observation/state features
- `"task"`: string

**Location**: `data-io/gs_data_io/writers/lerobot.py` (lines 342-343)

### 5. Parquet Writing by LeRobotDataset

The parquet serialization process involves several steps that convert numpy arrays to parquet files using feature metadata:

#### 5.1. Feature Metadata to HuggingFace Features Conversion

When `LeRobotDataset` is created, the feature metadata is converted to HuggingFace `Features` objects:

```python
def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        # ... Array3D, Array4D, Array5D for higher dimensions
```

For `"ctrl.joint_angles"` with shape `(2, 7)` and dtype `"float32"`, this creates:
- `datasets.Array2D(shape=(2, 7), dtype="float32")`

**Location**: `src/lerobot/datasets/utils.py` (lines 551-586)

#### 5.2. Frame Buffer Accumulation

`add_frame()` accumulates frames in an episode buffer:

```python
def add_frame(self, frame: dict) -> None:
    # Convert torch to numpy if needed
    for name in frame:
        if isinstance(frame[name], torch.Tensor):
            frame[name] = frame[name].numpy()

    validate_frame(frame, self.features)

    # Add frame features to episode_buffer
    for key in frame:
        if self.features[key]["dtype"] in ["image", "video"]:
            # Handle images/videos separately
        else:
            self.episode_buffer[key].append(frame[key])  # Store numpy array
```

**Location**: `src/lerobot/datasets/lerobot_dataset.py` (lines 1059-1100)

#### 5.3. Episode Buffer to HuggingFace Dataset Conversion

When `save_episode()` is called`, the episode buffer is converted to a HuggingFace Dataset:

```python
def _save_episode_data(self, episode_buffer: dict) -> dict:
    # Stack numpy arrays from episode buffer
    for key, ft in self.features.items():
        if key not in ["index", "episode_index", "task_index"] and ft["dtype"] not in ["image", "video"]:
            episode_buffer[key] = np.stack(episode_buffer[key])  # Stack frames into array

    # Convert buffer into HF Dataset
    ep_dict = {key: episode_buffer[key] for key in self.hf_features}
    ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
    ep_dataset = embed_images(ep_dataset)
```

**Key step**: `datasets.Dataset.from_dict()` uses the `features` parameter (HuggingFace Features) to:
- Convert numpy arrays to PyArrow extension types (e.g., `Array2DExtensionType`)
- Preserve shape and dtype information
- Create the correct Arrow schema

For `"ctrl.joint_angles"`:
- Input: numpy array with shape `(num_frames, 2, 7)` and dtype `float32`
- Output: PyArrow ExtensionArray with `Array2DExtensionType(shape=(2, 7), dtype=float32)`

**Location**: `src/lerobot/datasets/lerobot_dataset.py` (lines 1222-1237)

#### 5.4. Arrow Format Conversion and Parquet Writing

The Dataset is converted to Arrow format and written to parquet:

```python
# Convert to Arrow format
table = ep_dataset.with_format("arrow")[:]

# Write to parquet using PyArrow ParquetWriter
if not self.writer:
    self.writer = pq.ParquetWriter(
        path, schema=table.schema, compression="snappy", use_dictionary=True
    )
self.writer.write_table(table)
```

**How it works**:
1. `ep_dataset.with_format("arrow")[:]` converts the Dataset to a PyArrow Table
2. The table schema includes extension types (e.g., `Array2DExtensionType`) that preserve:
   - Shape information: `(2, 7)` for `ctrl.joint_angles`
   - Dtype: `float32`
   - Named dimensions: stored in extension type metadata
3. `ParquetWriter.write_table()` writes the table to parquet, preserving:
   - Extension type metadata in parquet schema
   - Shape and dtype information
   - Named dimensions (via `names` field in feature metadata)

**Location**: `src/lerobot/datasets/lerobot_dataset.py` (lines 1291-1296)

#### 5.5. How HuggingFace Datasets Handles Extension Types

The conversion from numpy arrays to PyArrow extension types happens inside HuggingFace's `datasets` library:

1. **`Dataset.from_dict()` with Features**: When provided with `features=datasets.Array2D(...)`, the library:
   - Detects that the feature is an extension type
   - Converts numpy arrays to PyArrow ListArrays (storage)
   - Wraps them in ExtensionArrays using `pa.ExtensionArray.from_storage()`
   - Preserves extension type metadata

2. **Extension Type Storage**: For `Array2D(shape=(2, 7), dtype=float32)`:
   - Storage: PyArrow ListArray containing flattened arrays
   - Extension type: `Array2DExtensionType` with shape and dtype metadata
   - This metadata is preserved in the parquet file schema

3. **Parquet Schema**: The parquet file stores:
   - Column name: `ctrl.joint_angles`
   - Extension type metadata: shape `(2, 7)`, dtype `float32`
   - Named dimensions: stored in feature metadata (accessed via `info.json`)

**Note**: The actual conversion code is in the HuggingFace `datasets` library (external dependency), specifically in `arrow_writer.py` and `features/features.py`.

## Detailed Code Locations for Parquet Serialization

### 1. Converting Numpy Arrays to PyArrow Extension Types

#### Call Sites in LeRobot Codebase

**Location 1**: `src/lerobot/datasets/lerobot_dataset.py` (line 1237)
```python
ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
```

**Location 2**: `src/lerobot/datasets/dataset_tools.py` (line 942)
```python
ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")
```

#### Implementation in HuggingFace Datasets Library

The conversion happens inside `datasets.Dataset.from_dict()` when provided with `features` containing extension types like `Array2D`.

**GitHub Repository**: https://github.com/huggingface/datasets

**File 1**: `src/datasets/arrow_writer.py`

**Function**: `TypedSequence.__arrow_array__()` (around lines 297-299)
```python
# custom pyarrow types
if isinstance(pa_type, _ArrayXDExtensionType):
    storage = to_pyarrow_listarray(data, pa_type)
    return pa.ExtensionArray.from_storage(pa_type, storage)
```

**File 2**: `src/datasets/features/features.py`

**Function**: `to_pyarrow_listarray()` (around lines 1580-1593)
```python
def to_pyarrow_listarray(data: Any, pa_type: _ArrayXDExtensionType) -> pa.Array:
    """Convert to PyArrow ListArray."""
    if contains_any_np_array(data):
        return any_np_array_to_pyarrow_listarray(data, type=pa_type.value_type)
    else:
        return pa.array(data, pa_type.storage_dtype)
```

**Flow**:
1. `Dataset.from_dict()` creates a `TypedSequence` for each feature
2. When the feature is an extension type (e.g., `Array2D`), `TypedSequence.__arrow_array__()` is called
3. This calls `to_pyarrow_listarray()` to convert numpy arrays to PyArrow ListArrays
4. The ListArray is wrapped in an `ExtensionArray` using `pa.ExtensionArray.from_storage()`
5. This preserves the extension type metadata (shape, dtype)

### 2. Converting Dataset to PyArrow Table using `with_format("arrow")`

#### Call Sites in LeRobot Codebase

**Location 1**: `src/lerobot/datasets/lerobot_dataset.py` (line 1291)
```python
table = ep_dataset.with_format("arrow")[:]
```

**Location 2**: `src/lerobot/datasets/dataset_tools.py` (line 947)
```python
table = ep_dataset.with_format("arrow")[:]
```

**Location 3**: `src/lerobot/datasets/utils.py` (line 236)
```python
dataset = dataset.with_format("arrow")
```

#### Implementation in HuggingFace Datasets Library

**GitHub Repository**: https://github.com/huggingface/datasets

**File**: `src/datasets/arrow_dataset.py`

**Method**: `Dataset.with_format()` (returns a formatted dataset view)

**File**: `src/datasets/arrow_dataset.py`

**Method**: `Dataset.__getitem__()` (when accessing `[:]` on a formatted dataset)

**Flow**:
1. `with_format("arrow")` sets the format to Arrow and returns a view
2. Accessing `[:]` triggers `__getitem__()` which converts the dataset to a PyArrow Table
3. The conversion preserves extension types from the dataset's features
4. Returns a `pyarrow.Table` with the schema containing extension types

**Key Point**: The extension type metadata is preserved because:
- The dataset's `features` attribute contains the extension type definitions
- When converting to Arrow format, these features are used to construct the PyArrow schema
- Extension types are registered with PyArrow and preserved in the table schema

### Summary of Code Paths

**For Operation 1 (numpy → extension types)**:
```
LeRobot: lerobot_dataset.py:1237
  ↓
HuggingFace Datasets: Dataset.from_dict()
  ↓
arrow_writer.py: TypedSequence.__arrow_array__()
  ↓
features/features.py: to_pyarrow_listarray()
  ↓
PyArrow: pa.ExtensionArray.from_storage()
```

**For Operation 2 (Dataset → PyArrow Table)**:
```
LeRobot: lerobot_dataset.py:1291
  ↓
HuggingFace Datasets: Dataset.with_format("arrow")
  ↓
arrow_dataset.py: Dataset.__getitem__()
  ↓
PyArrow: Returns pa.Table with extension types in schema
```

#### 5.6. Feature Metadata Storage

The feature metadata (including `names`) is stored separately in `meta/info.json`:

```python
dataset = LeRobotDataset.create(
    repo_id=config.repo_id,
    fps=config.fps,
    root=config.root,
    robot_type=None,
    features={k: v.model_dump() for k, v in features.items()},  # Includes names, shape, dtype
    # ...
)
```

This creates `meta/info.json` with:
```json
{
  "features": {
    "ctrl.joint_angles": {
      "dtype": "float32",
      "shape": [2, 7],
      "names": ["right.j0", "right.j1", ..., "left.j0", ...]
    }
  }
}
```

**Location**: `data-io/gs_data_io/writers/lerobot.py` (lines 202-213)

## Summary

The complete flow for control features like `ctrl.joint_angles`:

1. **Key generation**: `"ctrl." + "joint_angles"` → `"ctrl.joint_angles"`
2. **Feature metadata**: Built from `ActionSpace` with shape, dtype, and names
3. **Data conversion**: Action tensors → numpy arrays via `action.tensors()`
4. **Frame assembly**: Numpy arrays added to frame dict with the same keys
5. **Parquet serialization**: Multi-step process that preserves shape, dtype, and named dimensions:
   - Feature metadata → HuggingFace Features (e.g., `Array2D`)
   - Frames accumulated in episode buffer
   - Buffer converted to HuggingFace Dataset using Features schema
   - Dataset converted to PyArrow Table with extension types
   - PyArrow ParquetWriter writes table to parquet file
   - Feature metadata (including `names`) stored in `meta/info.json`

**Key insight**: The feature metadata is used to create HuggingFace `Features` objects (like `Array2D`), which guide the conversion of numpy arrays to PyArrow extension types. These extension types preserve shape and dtype information in the parquet file schema, while named dimensions are stored separately in `meta/info.json`.

## Key Components

- **Action classes**: Define `tensors()` method that returns prefixed keys
- **ActionSpace**: Provides tensor configurations for feature metadata
- **LeRobotFeature**: Metadata structure with names, shape, and dtype
- **LeRobotDataset**: Handles parquet serialization using feature metadata

