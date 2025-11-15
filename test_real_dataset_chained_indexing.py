#!/usr/bin/env python
"""Test script to reproduce the chained indexing error with real dataset parquet files."""

from pathlib import Path

import pandas as pd


def test_real_dataset_parquet():
    """Test chained indexing on a real dataset parquet file."""
    dataset_root = Path("/home/yaxiong-zhao/workspace/datasets/20251105/pick_the_apple_and_place_in_the_plate")

    # Find a data parquet file
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return False

    # Find the first parquet file
    parquet_files = list(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return False

    parquet_path = parquet_files[0]
    print(f"Testing with parquet file: {parquet_path}")

    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDataFrame dtypes:")
    for col in df.columns:
        dtype = df[col].dtype
        is_extension = pd.api.types.is_extension_array_dtype(df[col])
        print(f"  {col}: {dtype} (extension: {is_extension})")

    # Check if task_index exists
    if "task_index" not in df.columns:
        print("\n'task_index' column not found in DataFrame")
        return False

    # Create a mask similar to the actual code
    if "episode_index" not in df.columns:
        print("\n'episode_index' column not found in DataFrame")
        return False

    unique_episodes = df["episode_index"].unique()[:2]  # Get first 2 episodes
    mask = df["episode_index"].isin(unique_episodes)
    print(f"\nMask created: {mask.sum()} rows match episodes {unique_episodes.tolist()}")

    # Test chained indexing (the problematic approach)
    print("\n" + "="*60)
    print("Testing chained indexing: df[mask]['task_index']")
    print("="*60)
    try:
        task_series = df[mask]["task_index"]
        print(f"✓ Chained indexing succeeded: {len(task_series)} rows")
        print(f"  Result dtype: {task_series.dtype}")
        print(f"  Unique task indices: {task_series.unique().tolist()}")
        return False  # No error occurred
    except AttributeError as e:
        if "'PandasArrayExtensionDtype' object has no attribute 'v'" in str(e) or "has no attribute 'v'" in str(e):
            print(f"✗ Chained indexing failed with expected error:")
            print(f"  {type(e).__name__}: {e}")
            return True  # Error reproduced
        else:
            print(f"✗ Chained indexing failed with unexpected AttributeError:")
            print(f"  {type(e).__name__}: {e}")
            raise
    except Exception as e:
        print(f"✗ Chained indexing failed with unexpected error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test .loc fix
    print("\n" + "="*60)
    print("Testing fix: df.loc[mask, 'task_index']")
    print("="*60)
    try:
        task_series = df.loc[mask, "task_index"]
        print(f"✓ .loc indexing succeeded: {len(task_series)} rows")
        print(f"  Result dtype: {task_series.dtype}")
        print(f"  Unique task indices: {task_series.unique().tolist()}")
        return True
    except Exception as e:
        print(f"✗ .loc indexing failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("="*60)
    print("Testing Chained Indexing with Real Dataset")
    print("="*60)

    error_occurred = test_real_dataset_parquet()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if error_occurred:
        print("✓ Error reproduced with chained indexing on real dataset")
        print("  This confirms the issue exists and needs the fix.")
    else:
        print("⚠ Error not reproduced (may depend on pandas version or data format)")

