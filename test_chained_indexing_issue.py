#!/usr/bin/env python
"""Test script to reproduce and confirm fix for pandas chained indexing issue.

This script tests the AttributeError that occurs when using chained indexing
df[mask]["task_index"] on parquet dataframes with extension dtypes.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def create_test_parquet_with_extension_dtypes(tmp_path: Path) -> Path:
    """Create a test parquet file that may have extension dtypes when read back."""
    # Create a DataFrame similar to what lerobot datasets use
    n_rows = 100
    df = pd.DataFrame({
        "episode_index": np.arange(n_rows) // 10,  # 10 rows per episode
        "task_index": np.arange(n_rows) // 20,  # 20 rows per task
        "frame_index": np.arange(n_rows),
        "index": np.arange(n_rows),
        "timestamp": np.arange(n_rows, dtype=np.float32) / 30.0,
    })

    # Write to parquet
    parquet_path = tmp_path / "test_data.parquet"
    df.to_parquet(parquet_path, engine="pyarrow")

    return parquet_path


def test_chained_indexing_issue(parquet_path: Path) -> bool:
    """Test if chained indexing causes the AttributeError."""
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"DataFrame dtypes:\n{df.dtypes}")
    print(f"task_index dtype: {df['task_index'].dtype}")
    print(f"task_index dtype type: {type(df['task_index'].dtype)}")

    # Check if we have extension dtypes
    has_extension = False
    for col in df.columns:
        if pd.api.types.is_extension_array_dtype(df[col]):
            print(f"Column {col} has extension dtype: {df[col].dtype}")
            has_extension = True

    if not has_extension:
        print("Note: No extension dtypes detected. The issue may not reproduce with this data.")
        print("The issue typically occurs with parquet files written by HuggingFace datasets library.")

    # Create a boolean mask similar to the actual code
    mask = df["episode_index"].isin([0, 1])
    print(f"\nMask created: {mask.sum()} rows match")

    # Try chained indexing (the problematic approach)
    print("\nTesting chained indexing: df[mask]['task_index']")
    try:
        task_series = df[mask]["task_index"]
        print(f"✓ Chained indexing succeeded: {len(task_series)} rows")
        print(f"  Result dtype: {task_series.dtype}")
        return False  # No error occurred
    except AttributeError as e:
        if "'PandasArrayExtensionDtype' object has no attribute 'v'" in str(e):
            print(f"✗ Chained indexing failed with expected error:")
            print(f"  {type(e).__name__}: {e}")
            return True  # Error reproduced
        else:
            print(f"✗ Chained indexing failed with unexpected error:")
            print(f"  {type(e).__name__}: {e}")
            raise

    except Exception as e:
        print(f"✗ Chained indexing failed with unexpected error:")
        print(f"  {type(e).__name__}: {e}")
        raise


def test_loc_indexing_fix(parquet_path: Path) -> bool:
    """Test if using .loc fixes the issue."""
    print("\n" + "="*60)
    print("Testing fix: df.loc[mask, 'task_index']")
    print("="*60)

    df = pd.read_parquet(parquet_path)
    mask = df["episode_index"].isin([0, 1])

    try:
        task_series = df.loc[mask, "task_index"]
        print(f"✓ .loc indexing succeeded: {len(task_series)} rows")
        print(f"  Result dtype: {task_series.dtype}")
        print(f"  Unique task indices: {task_series.unique().tolist()}")
        return True
    except Exception as e:
        print(f"✗ .loc indexing failed:")
        print(f"  {type(e).__name__}: {e}")
        raise


def main():
    """Run the tests."""
    print("="*60)
    print("Testing Pandas Chained Indexing Issue")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test parquet file
        print("\n1. Creating test parquet file...")
        parquet_path = create_test_parquet_with_extension_dtypes(tmp_path)
        print(f"   Created: {parquet_path}")

        # Test chained indexing (should fail or work depending on pandas version)
        print("\n2. Testing chained indexing (problematic approach)...")
        error_occurred = test_chained_indexing_issue(parquet_path)

        # Test .loc fix
        print("\n3. Testing .loc indexing (fixed approach)...")
        fix_works = test_loc_indexing_fix(parquet_path)

        # Summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        if error_occurred:
            print("✓ Error reproduced with chained indexing")
        else:
            print("⚠ Error not reproduced (may need real dataset with extension dtypes)")

        if fix_works:
            print("✓ Fix confirmed: .loc indexing works correctly")
        else:
            print("✗ Fix failed: .loc indexing still has issues")

        print("\nNote: To fully test with extension dtypes, use a real dataset")
        print("      created by lerobot that contains HuggingFace extension metadata.")


if __name__ == "__main__":
    main()

