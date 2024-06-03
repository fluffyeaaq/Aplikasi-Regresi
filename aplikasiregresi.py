import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Membaca data dari CSV
data = pd.read_csv("student_performance.csv")

# Memilih kolom yang dibutuhkan
TB = data["Hours Studied"]
NL = data["Sample Question Papers Practiced"]
NT = data["Performance Index"]

# Model Linear (Metode 1)

# Membangun model regresi linear
model_linear = LinearRegression()
model_linear.fit(np.array([TB, NL]).T, NT)

# Mendapatkan koefisien regresi
b0_linear = model_linear.intercept_
b1_linear = model_linear.coef_[0]
b2_linear = model_linear.coef_[1]

# Persamaan model linear
print("Model Linear:")
print(f"NT = {b0_linear:.3f} + {b1_linear:.3f} * TB + {b2_linear:.3f} * NL")

# Prediksi nilai NT menggunakan model linear
NT_pred_linear = model_linear.predict(np.array([TB, NL]).T)

# Menghitung galat RMS (Root Mean Squared Error)
rmse_linear = np.sqrt(np.mean((NT - NT_pred_linear)**2))
print(f"Galat RMS (Linear): {rmse_linear:.3f}")

# Plot grafik titik data dan hasil regresi linear
plt.scatter(TB, NT, label="Data Aktual")
plt.plot(TB, NT_pred_linear, label="Prediksi Linear")
plt.xlabel("Durasi Waktu Belajar (Jam)")
plt.ylabel("Nilai Ujian")
plt.title("Hubungan Durasi Waktu Belajar dan Nilai Ujian (Model Linear)")
plt.legend()
plt.show()

# Model Pangkat Sederhana (Metode 2)

# Transformasi log pada variabel TB dan NL
TB_log = np.log(TB + 1)  # Menambahkan 1 untuk menghindari logaritma dari 0
NL_log = np.log(NL + 1)  # Menambahkan 1 untuk menghindari logaritma dari 0

# Membangun model regresi linear dengan variabel tertransformasi
model_log = LinearRegression()
model_log.fit(np.array([TB_log, NL_log]).T, NT)

# Mendapatkan koefisien regresi
b0_log = model_log.intercept_
b1_log = model_log.coef_[0]
b2_log = model_log.coef_[1]

# Persamaan model pangkat sederhana
print("\nModel Pangkat Sederhana:")
print(f"log(NT) = {b0_log:.3f} + {b1_log:.3f} * log(TB) + {b2_log:.3f} * log(NL)")

# Transformasi kembali ke skala asli
NT_pred_log = np.exp(b0_log + b1_log * TB_log + b2_log * NL_log)

# Menghitung galat RMS
rmse_log = np.sqrt(np.mean((NT - NT_pred_log)**2))
print(f"Galat RMS (Pangkat Sederhana): {rmse_log:.3f}")

# Plot grafik titik data dan hasil regresi pangkat sederhana
plt.scatter(TB, NT, label="Data Aktual")
plt.plot(TB, NT_pred_log, label="Prediksi Pangkat Sederhana")
plt.xlabel("Durasi Waktu Belajar (Jam)")
plt.ylabel("Nilai Ujian")
plt.title("Hubungan Durasi Waktu Belajar dan Nilai Ujian (Model Pangkat Sederhana)")
plt.legend()
plt.show()
