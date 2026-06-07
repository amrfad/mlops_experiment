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
# # **1. Perkenalan Dataset**
#
# Dataset yang digunakan dalam eksperimen ini adalah **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset** yang dirilis oleh NASA. Dataset ini berisi data simulasi degradasi mesin jet (turbofan) dari waktu ke waktu hingga terjadi kegagalan (run-to-failure).
#
# **Rincian Dataset (FD001):**
# - **Jumlah Engine (Train/Test):** 100 engine pelatihan dan 100 engine pengujian.
# - **Kondisi Operasi:** Single Operating Condition (Sea Level).
# - **Fault Modes:** Single Fault Mode (HPC Degradation).
# - **Struktur Kolom:** 26 kolom numerik yang dipisahkan oleh spasi:
#   1. `unit_nr` (ID mesin)
#   2. `time_cycles` (Siklus waktu)
#   3. `op_1` (Operational setting 1)
#   4. `op_2` (Operational setting 2)
#   5. `op_3` (Operational setting 3)
#   6. `s1` s.d. `s21` (Sensor measurement 1 s.d. 21)
#
# **Tujuan Eksperimen:**
# Memprediksi **Remaining Useful Life (RUL)** atau sisa siklus operasional mesin jet pada set pengujian berdasarkan data sensor historis.
#
# Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

# %% [markdown]
# # **2. Import Library**
#
# Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning.

# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# %% [markdown]
# # **3. Memuat Dataset**
#
# Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.
#
# Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.
#
# Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan

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

# Load train and test datasets
train_df = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)
test_df = pd.read_csv(TEST_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)

# Load ground truth RUL for test set
rul_df = pd.read_csv(RUL_PATH, sep=r"\s+", header=None, names=['RUL'])

# %%
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
print(f"RUL Shape: {rul_df.shape}")

# %%
print("Train Head:")
print(train_df.head())

# %%
print("\nTest Head:")
print(test_df.head())

# %%
print("\nRUL Head:")
print(rul_df.head())

# %%
print("Train Data Types and Missing Values:")
train_df.info()

# %%
print("\nMissing values count in Train:")
print(train_df.isnull().sum().sum())

print("\nMissing values count in Test:")
print(test_df.isnull().sum().sum())

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

# %% [markdown]
# # **4. Exploratory Data Analysis (EDA)**
#
# Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.
#
# Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.

# %%
# Describe sensor columns
sensor_cols = [col for col in train_df.columns if col.startswith('s')]
desc = train_df[sensor_cols].describe()
print(desc.transpose()[['mean', 'std', 'min', 'max']])

# Identify sensors with std near 0 (std < 0.01)
constant_sensors = desc.columns[desc.loc['std'] < 0.01].tolist()
print(f"\nSensors with std near 0 (almost constant): {constant_sensors}")

# %% [markdown]
# **Temuan Sensor Konstan:**
# Secara empiris pada dataset FD001, sensor berikut adalah konstan atau hampir konstan (std ≈ 0): `s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19`. Sensor-sensor ini tidak membawa informasi tentang proses degradasi mesin, sehingga perlu di-drop pada tahap preprocessing.

# %%
# Select informative sensors to visualize
visualize_sensors = ['s2', 's3', 's4', 's11', 's12', 's15']
selected_units = [1, 2, 3, 4, 5]

plt.figure(figsize=(15, 10))
for i, sensor in enumerate(visualize_sensors, 1):
    plt.subplot(3, 2, i)
    for unit in selected_units:
        unit_data = train_df[train_df['unit_nr'] == unit]
        plt.plot(unit_data['time_cycles'], unit_data[sensor], label=f"Unit {unit}")
    plt.title(f"Degradation Trend of {sensor}")
    plt.xlabel("Time Cycles")
    plt.ylabel("Value")
    if i == 1:
        plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Temuan Tren Degradasi:**
# Visualisasi menunjukkan tren degradasi yang monoton seiring bertambahnya siklus (time_cycles).
# - Beberapa sensor menunjukkan tren **meningkat** (seperti `s2`, `s3`, `s4`, `s11`, `s15`).
# - Beberapa sensor menunjukkan tren **menurun** (seperti `s12`).
# Ini mengonfirmasi bahwa pola sensor berkorelasi dengan degradasi kesehatan mesin, yang memungkinkan model untuk memprediksi sisa umur mesin (RUL).

# %%
# Group by unit_nr to get lifetime of each engine
lifetimes = train_df.groupby('unit_nr')['time_cycles'].max()

plt.figure(figsize=(8, 5))
sns.histplot(lifetimes, bins=20, kde=True, color='skyblue')
plt.axvline(lifetimes.mean(), color='red', linestyle='--', label=f"Mean: {lifetimes.mean():.1f}")
plt.axvline(lifetimes.median(), color='green', linestyle='-', label=f"Median: {lifetimes.median():.1f}")
plt.title("Distribution of Engine Lifetimes in Training Set")
plt.xlabel("Maximum Time Cycles before Failure")
plt.ylabel("Count")
plt.legend()
plt.show()

print(f"Minimum engine lifetime: {lifetimes.min()} cycles")
print(f"Maximum engine lifetime: {lifetimes.max()} cycles")

# %% [markdown]
# **Justifikasi Teknis untuk Clipping RUL (Piecewise RUL):**
# Grafik distribusi panjang time series menunjukkan bahwa umur mesin berkisar dari 128 hingga 362 siklus. 
# Pada masa awal operasi (misal ketika mesin baru beroperasi < 100 siklus), kondisi mesin masih prima dan sensor menunjukkan nilai yang stabil tanpa sinyal degradasi yang signifikan. 
# Jika RUL didefinisikan secara linear penuh (misal RUL = 300 di awal), model regresi akan kesulitan mempelajari hubungan sensor dengan RUL karena sensor di awal (sehat) terlihat sama baik untuk mesin yang berumur 300 siklus maupun 150 siklus.
# 
# Oleh karena itu, kita menggunakan **clipping RUL** (misalnya di batas 125 siklus). RUL di awal operasi di-clip pada nilai 125. Saat mesin mulai terdegradasi (di bawah 125 siklus sebelum gagal), barulah RUL berkurang secara linear hingga mencapai 0. Ini sangat meningkatkan akurasi estimasi RUL.

# %%
# Filter out constant sensors
informative_sensors = [s for s in sensor_cols if s not in constant_sensors]

# Correlation heatmap
corr = train_df[informative_sensors].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Informative Sensors")
plt.show()

# %%
op_cols = ['op_1', 'op_2', 'op_3']
plt.figure(figsize=(12, 4))
for i, op in enumerate(op_cols, 1):
    plt.subplot(1, 3, i)
    sns.histplot(train_df[op], kde=True, bins=15)
    plt.title(f"Distribution of {op}")
plt.tight_layout()
plt.show()

for op in op_cols:
    print(f"Unique values/Stats for {op}:")
    print(train_df[op].describe())
    print("-" * 30)

# %% [markdown]
# **Temuan Operational Settings:**
# - Untuk dataset FD001, operational settings hampir konstan (hanya ada variasi kecil pada `op_1` dan `op_2`, sedangkan `op_3` adalah konstan 100.0). Hal ini mengonfirmasi bahwa FD001 beroperasi pada kondisi tunggal (single operating condition).

# %% [markdown]
# # **5. Data Preprocessing**
#
# Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.
#
# Proses preprocessing pada dataset C-MAPSS ini meliputi:
# 1. Menghapus sensor konstan (non-informative).
# 2. Menghitung Remaining Useful Life (RUL) target.
# 3. Melakukan Clipping RUL (Piecewise Linear RUL).
# 4. Feature Engineering: Rolling Features (Mean dan Std) dengan window size 5 dan 10 untuk menangkap tren waktu.
# 5. Normalisasi fitur menggunakan MinMaxScaler.
# 6. Menyiapkan label RUL untuk test set menggunakan ground truth.
# 7. Mengekspor hasil preprocessed data ke CSV.

# %%
# Step 1 — Drop Constant/Non-informative Sensors
SENSOR_COLS_TO_DROP = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
train_processed = train_df.drop(columns=SENSOR_COLS_TO_DROP)
test_processed = test_df.drop(columns=SENSOR_COLS_TO_DROP)

# Remaining sensor columns
remaining_sensors = [col for col in train_processed.columns if col.startswith('s')]
print(f"Remaining sensors: {remaining_sensors}")

# %%
# Step 2 — Derive RUL target for Training Set
max_cycles = train_processed.groupby('unit_nr')['time_cycles'].max()
train_processed = train_processed.merge(max_cycles.rename('max_cycle'), on='unit_nr')
train_processed['RUL'] = train_processed['max_cycle'] - train_processed['time_cycles']
train_processed.drop(columns=['max_cycle'], inplace=True)

print("Sample RUL derivation:")
print(train_processed[['unit_nr', 'time_cycles', 'RUL']].head())

# %%
# Step 3 — Clipping RUL
RUL_CLIP = 125
train_processed['RUL'] = train_processed['RUL'].clip(upper=RUL_CLIP)

print("\nSample clipped RUL:")
print(train_processed[['unit_nr', 'time_cycles', 'RUL']].head())

# %%
# Step 4 — Feature Engineering: Rolling Features
def add_rolling_features(df, sensor_cols, windows=[5, 10]):
    features_df = df.copy()
    for window in windows:
        # Rolling mean
        rolling_mean = df.groupby('unit_nr')[sensor_cols].rolling(window=window).mean().reset_index(level=0, drop=True)
        rolling_mean.columns = [f"{col}_roll_mean_{window}" for col in sensor_cols]
        
        # Rolling std
        rolling_std = df.groupby('unit_nr')[sensor_cols].rolling(window=window).std().reset_index(level=0, drop=True)
        rolling_std.columns = [f"{col}_roll_std_{window}" for col in sensor_cols]
        
        # Merge
        features_df = pd.concat([features_df, rolling_mean, rolling_std], axis=1)
    return features_df

# Apply rolling features
train_processed = add_rolling_features(train_processed, remaining_sensors)
test_processed = add_rolling_features(test_processed, remaining_sensors)

# Drop any rows containing NaN values due to the rolling window calculation
train_processed.dropna(inplace=True)
test_processed.dropna(inplace=True)

# Reset index
train_processed.reset_index(drop=True, inplace=True)
test_processed.reset_index(drop=True, inplace=True)

print(f"Shape of Train with rolling features (after dropping NaNs): {train_processed.shape}")
print(f"Shape of Test with rolling features (after dropping NaNs): {test_processed.shape}")

# %%
# Step 5 — Normalization using MinMaxScaler
exclude_cols = ['unit_nr', 'time_cycles', 'RUL']
feature_cols = [col for col in train_processed.columns if col not in exclude_cols]

scaler = MinMaxScaler()

# Fit scaler on train features and transform both
train_processed[feature_cols] = scaler.fit_transform(train_processed[feature_cols])
test_processed[feature_cols] = scaler.transform(test_processed[feature_cols])

print("Sample scaled training features:")
print(train_processed[feature_cols].head())

# %%
# Step 6 — Prepare Test Labels (Ground Truth RUL)
test_last_cycle = test_processed.groupby('unit_nr').last().reset_index()
rul_df['unit_nr'] = rul_df.index + 1
test_final = test_last_cycle.merge(rul_df, on='unit_nr')

print("Final test dataframe structure with ground truth RUL:")
print(test_final[['unit_nr', 'time_cycles', 'RUL']].head())

# %%
# Step 7 — Export Dataset
OUTPUT_DIR = "FD001_preprocessing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "FD001_preprocessing_train.csv")
TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "FD001_preprocessing_test.csv")

# Export to CSV
train_processed.to_csv(TRAIN_OUTPUT_PATH, index=False)
test_final.to_csv(TEST_OUTPUT_PATH, index=False)

print(f"Train dataset exported to: {TRAIN_OUTPUT_PATH}")
print(f"Test dataset exported to: {TEST_OUTPUT_PATH}")

# %%
# Final Verification Checks
verify_train = pd.read_csv(TRAIN_OUTPUT_PATH)
verify_test = pd.read_csv(TEST_OUTPUT_PATH)

print(f"Verify Train shape: {verify_train.shape}")
print(f"Verify Test shape: {verify_test.shape}")
print(f"Total NaNs in exported Train: {verify_train.isnull().sum().sum()}")
print(f"Total NaNs in exported Test: {verify_test.isnull().sum().sum()}")

assert verify_train.isnull().sum().sum() == 0, "Error: NaNs found in exported Train set!"
assert verify_test.isnull().sum().sum() == 0, "Error: NaNs found in exported Test set!"
print("Verification successful! Preprocessed datasets are clean and ready.")
