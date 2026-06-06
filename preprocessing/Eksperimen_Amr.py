# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMAPSS RUL Prediction - Data Loading & Verification
# This notebook contains the implementation of Fase 1: Data Loading of the development plan.

# %%
import pandas as pd
import numpy as np
import os

# %% [markdown]
# ## 1. Define Paths and Column Names

# %%
# Define paths to data files
RAW_DATA_DIR = "../FD001_raw" # relative to preprocessing directory
TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train_FD001.txt")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test_FD001.txt")
RUL_PATH = os.path.join(RAW_DATA_DIR, "RUL_FD001.txt")

# Define column names based on CMAPSS dataset documentation
COLUMN_NAMES = [
    'unit_nr', 'time_cycles', 'op_1', 'op_2', 'op_3',
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
]

# %% [markdown]
# ## 2. Load Datasets
# Note: The raw data contains trailing spaces at the end of each line, which can result in extra NaN columns if not handled properly.
# We will use `sep=r"\s+"` (regex separator for one or more whitespaces) to avoid extra NaN columns.

# %%
# Load train and test datasets
train_df = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)
test_df = pd.read_csv(TEST_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)

# Load ground truth RUL for test set
rul_df = pd.read_csv(RUL_PATH, sep=r"\s+", header=None, names=['RUL'])

# %% [markdown]
# ## 3. Verify Shape and Columns

# %%
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
print(f"RUL Shape: {rul_df.shape}")

# %% [markdown]
# ## 4. Display Head of Datasets

# %%
print("Train Head:")
print(train_df.head())

# %%
print("\nTest Head:")
print(test_df.head())

# %%
print("\nRUL Head:")
print(rul_df.head())

# %% [markdown]
# ## 5. Check Data Types and Missing Values

# %%
print("Train Data Types and Missing Values:")
train_df.info()

# %%
print("\nTest Data Types and Missing Values:")
test_df.info()

# %%
print("\nMissing values count in Train:")
print(train_df.isnull().sum().sum())

print("\nMissing values count in Test:")
print(test_df.isnull().sum().sum())

# %% [markdown]
# ## 6. Verify Unique Engines (unit_nr)

# %%
num_train_units = train_df['unit_nr'].nunique()
num_test_units = test_df['unit_nr'].nunique()
num_rul_units = len(rul_df)

print(f"Number of unique engines in Train set: {num_train_units}")
print(f"Number of unique engines in Test set: {num_test_units}")
print(f"Number of engines in Ground Truth RUL set: {num_rul_units}")

assert num_train_units == 100, f"Expected 100 train units, found {num_train_units}"
assert num_test_units == 100, f"Expected 100 test units, found {num_test_units}"
assert num_rul_units == 100, f"Expected 100 ground truth units, found {num_rul_units}"
print("Verification successful! All checks passed.")
