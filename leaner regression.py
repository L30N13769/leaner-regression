# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 23:04:46 2026

@author: bilge
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 23:04:46 2026

@author: bilge
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv("employee_income (1).csv")


# İlk 30 satırı göster
print(data.head(30))


# Toplam satır sayısı
print("Toplam veri sayısı:", len(data))

# Satır ve sütun sayısı
print("Boyut:", data.shape)
print(data.head(20))
# Tüm tablo için eksik değer sayısı
print(data.isnull().sum())

# Toplam eksik hücre sayısı
print("Toplam eksik değer:", data.isnull().sum().sum())
print("Tekrarlanan satır sayısı:", data.duplicated().sum())

# Bağımsız değişken: experience
X = data[['experience']]
# Bağımlı değişken: income (maaş)
y = data['income']

# Deneyim ve maaş ortalamaları
mean_experience = data['experience'].mean()
mean_income = data['income'].mean()

print("Ortalama deneyim:", mean_experience)
print("Ortalama maaş:", mean_income)

x = data['experience']
y = data['income']

x_mean = x.mean()
y_mean = y.mean()

# m = eğim
numerator = ((x - x_mean) * (y - y_mean)).sum()
denominator = ((x - x_mean)**2).sum()
m = numerator / denominator

# b = kesişim
b = y_mean - m * x_mean

print("m (eğim):", m)
print("b (kesişim):", b)

# Manuel hesaplama denklemi
def predict_income(experience):
    return m * experience + b

# Örnek: 10 yıl deneyim için maaş tahmini
print("10 yıl deneyim için tahmini maaş:", predict_income(10))

# Linear regresyon denklemi: y = m*x + b
def predict_income(x_pred):
    return m * x_pred + b

# Örnek tahminler
print("5 yıl deneyim:", predict_income(5))
print("9 yıl deneyim:", predict_income(9))
print("12 yıl deneyim:", predict_income(12))

# Grafik çizimi
plt.figure(figsize=(8,6))
plt.scatter(x, y, color='blue', label='Gerçek veriler')

# Regresyon doğrusu
y_pred = m * x + b
plt.plot(x, y_pred, color='red', label=f'y = {m:.2f}x + {b:.2f}')

plt.title("Deneyim vs Maaş (Linear Regression)")
plt.xlabel("Deneyim (yıl)")
plt.ylabel("Maaş (TL)")
plt.legend()
plt.show()






















































