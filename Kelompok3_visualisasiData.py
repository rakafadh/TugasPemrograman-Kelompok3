import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# Biar plotnya keliatan lebih kece
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = [14, 10]

# Baca data dari file CSV
data = pd.read_csv('Data Tugas Pemrograman A.csv')

# Ambil data yang dibutuhin
years = data['Year'].values
population = data['Population'].values
internet_pct = data['Percentage_Internet_User'].values

# Tahun yang datanya bolong
missing_years = [2005, 2006, 2015, 2016]
all_years = np.arange(1960, 2024)
years_for_interpolation = np.array([y for y in all_years if y not in missing_years])

# Biar hitungannya lebih stabil, normalisasi tahun
base_year = 1960
normalized_years = years - base_year
normalized_all_years = all_years - base_year
normalized_missing_years = np.array(missing_years) - base_year

# Fungsi buat fitting polinomial
def fit_polynomial(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coeffs)
    return polynomial, coeffs

# Ambil data internet mulai dari 1994 (sebelum itu nol semua)
internet_start_idx = np.where(years >= 1994)[0][0]
internet_years = normalized_years[internet_start_idx:]
internet_values = internet_pct[internet_start_idx:]

# Fitting polinomial buat populasi dan internet
pop_poly_degree = 2
int_poly_degree = 3

pop_poly, pop_coeffs = fit_polynomial(normalized_years, population, pop_poly_degree)
int_poly, int_coeffs = fit_polynomial(internet_years, internet_values, int_poly_degree)

# Bikin data buat plot kurva fitting
x_pop = np.linspace(normalized_years.min(), normalized_years.max() + 15, 1000)  # +15 buat proyeksi
y_pop = pop_poly(x_pop)

x_int = np.linspace(internet_years.min(), internet_years.max() + 15, 1000)  # +15 buat proyeksi
y_int = int_poly(x_int)

# Estimasi data yang bolong
pop_estimates = {
    2005: pop_poly(2005 - base_year),
    2006: pop_poly(2006 - base_year),
    2015: pop_poly(2015 - base_year),
    2016: pop_poly(2016 - base_year)
}

int_estimates = {
    2005: int_poly(2005 - base_year),
    2006: int_poly(2006 - base_year),
    2015: int_poly(2015 - base_year),
    2016: int_poly(2016 - base_year)
}

# Proyeksi buat tahun 2030 dan 2035
pop_2030 = pop_poly(2030 - base_year)
pop_2035 = pop_poly(2035 - base_year)
int_2035 = int_poly(2035 - base_year)

# Cetak persamaan polinomial biar gampang dibaca
def print_polynomial_equation(coeffs):
    terms = []
    for i, coef in enumerate(reversed(coeffs)):
        power = len(coeffs) - i - 1
        if abs(coef) < 1e-10:
            continue
        if power == 0:
            terms.append(f"{coef:.6f}")
        elif power == 1:
            terms.append(f"{coef:.6f}x")
        else:
            terms.append(f"{coef:.6f}x^{power}")
    
    equation = " + ".join(terms).replace("+ -", "- ")
    return f"y = {equation} (dimana x = tahun - {base_year})"

# Bikin plot buat visualisasi
fig = plt.figure(figsize=(16, 20))
gs = GridSpec(3, 2, figure=fig)

# Plot 1: Pertumbuhan populasi + kurva fitting
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(years, population, color='blue', label='Data Aktual')
ax1.plot(x_pop + base_year, y_pop, color='red', linestyle='-', label=f'Polinomial (derajat {pop_poly_degree})')

# Tambahin estimasi buat tahun yang bolong
for year, value in pop_estimates.items():
    ax1.scatter(year, value, color='green', s=80, marker='s', label=f'Estimasi {year}' if year == 2005 else "")

# Tambahin proyeksi buat 2030
ax1.scatter(2030, pop_2030, color='purple', s=100, marker='*', label='Proyeksi 2030')

ax1.set_title('Pertumbuhan Populasi Indonesia (1960-2023) dengan Estimasi', fontsize=16)
ax1.set_xlabel('Tahun', fontsize=12)
ax1.set_ylabel('Populasi', fontsize=12)
ax1.ticklabel_format(style='plain', axis='y')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Plot 2: Persentase pengguna internet + kurva fitting
ax2 = fig.add_subplot(gs[1, :])
internet_start_year = years[internet_start_idx]
ax2.scatter(years[internet_start_idx:], internet_pct[internet_start_idx:], color='blue', label='Data Aktual')
ax2.plot(x_int + base_year, y_int, color='red', linestyle='-', label=f'Polinomial (derajat {int_poly_degree})')

# Tambahin estimasi buat tahun yang bolong
for year, value in int_estimates.items():
    ax2.scatter(year, value, color='green', s=80, marker='s', label=f'Estimasi {year}' if year == 2005 else "")

# Tambahin proyeksi buat 2035
ax2.scatter(2035, int_2035, color='purple', s=100, marker='*', label='Proyeksi 2035')

ax2.set_title('Persentase Pengguna Internet di Indonesia (1994-2023) dengan Estimasi', fontsize=16)
ax2.set_xlabel('Tahun', fontsize=12)
ax2.set_ylabel('Persentase Pengguna Internet (%)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

# Plot 3: Residual plot buat populasi
ax3 = fig.add_subplot(gs[2, 0])
residuals_pop = population - pop_poly(normalized_years)
ax3.scatter(years, residuals_pop, color='orange')
ax3.axhline(y=0, color='red', linestyle='-')
ax3.set_title('Residual Plot - Model Populasi', fontsize=16)
ax3.set_xlabel('Tahun', fontsize=12)
ax3.set_ylabel('Residual', fontsize=12)
ax3.grid(True, alpha=0.3)

# Plot 4: Residual plot buat internet
ax4 = fig.add_subplot(gs[2, 1])
residuals_int = internet_values - int_poly(internet_years)
ax4.scatter(years[internet_start_idx:], residuals_int, color='orange')
ax4.axhline(y=0, color='red', linestyle='-')
ax4.set_title('Residual Plot - Model Persentase Internet', fontsize=16)
ax4.set_xlabel('Tahun', fontsize=12)
ax4.set_ylabel('Residual', fontsize=12)
ax4.grid(True, alpha=0.3)

# Tampilkan persamaan polinomial di bawah plot
pop_equation = print_polynomial_equation(pop_coeffs)
int_equation = print_polynomial_equation(int_coeffs)

plt.figtext(0.1, 0.05, f"Persamaan Populasi: {pop_equation}", fontsize=12, ha='left')
plt.figtext(0.1, 0.03, f"Persamaan Internet: {int_equation}", fontsize=12, ha='left')

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig('visualisasi_fitting_kurva.png', dpi=300, bbox_inches='tight')
plt.show()

# Cetak hasil estimasi
print("\nHASIL ESTIMASI DATA YANG HILANG:")
print("-" * 40)
print(f"Populasi 2005: {pop_estimates[2005]:,.0f} jiwa")
print(f"Populasi 2006: {pop_estimates[2006]:,.0f} jiwa")
print(f"Populasi 2015: {pop_estimates[2015]:,.0f} jiwa")
print(f"Populasi 2016: {pop_estimates[2016]:,.0f} jiwa")
print()
print(f"Internet 2005: {int_estimates[2005]:.4f}%")
print(f"Internet 2006: {int_estimates[2006]:.4f}%")
print(f"Internet 2015: {int_estimates[2015]:.4f}%")
print(f"Internet 2016: {int_estimates[2016]:.4f}%")
print()
print("\nPROYEKSI MASA DEPAN:")
print("-" * 40)
print(f"Populasi 2030: {pop_2030:,.0f} jiwa")
print(f"Persentase Internet 2035: {int_2035:.4f}%")
print(f"Jumlah Pengguna Internet 2035: {(int_2035/100 * pop_2035):,.0f} jiwa")