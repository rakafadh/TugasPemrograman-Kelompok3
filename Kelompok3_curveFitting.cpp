/**
 * Program buat nyocokin kurva buat estimasi data populasi dan pengguna internet di Indonesia
 * 
 * Program ini pake metode regresi polinomial buat:
 * 1. Ngeprediksi data yang ilang (tahun 2005, 2006, 2015, 2016)
 * 2. Bikin persamaan polinomial buat pertumbuhan populasi dan pengguna internet
 * 3. Ngeprediksi data di masa depan (populasi 2030 dan pengguna internet 2035)
 * 
 * Mata Kuliah: Komputasi Numerik
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace std;

/**
 * Struktur buat nyimpen data tahunan
 */
struct YearData {
    int year;
    double internet_percentage;
    double population;
};

/**
 * Fungsi buat baca file CSV dan ambil datanya
 * @param filename Nama file CSV
 * @return Vector yang isinya data tahunan
 */
vector<YearData> readCSV(const string& filename) {
    vector<YearData> data;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Gagal buka file " << filename << endl;
        return data;
    }
    
    // Skip header
    string line;
    getline(file, line);
    
    // Baca data
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        YearData yearData;
        
        // Baca tahun
        getline(ss, token, ',');
        yearData.year = stoi(token);
        
        // Baca persentase internet
        getline(ss, token, ',');
        yearData.internet_percentage = stod(token);
        
        // Baca populasi
        getline(ss, token, ',');
        yearData.population = stod(token);
        
        data.push_back(yearData);
    }
    
    file.close();
    return data;
}

/**
 * Fungsi buat eliminasi Gauss di matriks augmented
 * @param A Matriks augmented
 * @param n Dimensi matriks
 * @return Solusi persamaan linear
 */
vector<double> gaussElimination(vector<vector<double>> A, int n) {
    vector<double> x(n);

    // Forward elimination
    for (int i = 0; i < n - 1; i++) {
        // Partial pivoting
        int maxRow = i;
        double maxVal = abs(A[i][i]);
        
        for (int k = i + 1; k < n; k++) {
            if (abs(A[k][i]) > maxVal) {
                maxVal = abs(A[k][i]);
                maxRow = k;
            }
        }
        
        if (maxRow != i) {
            swap(A[i], A[maxRow]);
        }
        
        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j <= n; j++) {
                A[k][j] -= factor * A[i][j];
            }
        }
    }
    
    // Back substitution
    for (int i = n - 1; i >= 0; i--) {
        x[i] = A[i][n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    
    return x;
}

/**
 * Fungsi buat hitung koefisien polinomial pake metode regresi polinomial
 * @param x Vektor data x
 * @param y Vektor data y
 * @param degree Derajat polinomial
 * @return Vektor koefisien polinomial
 */
vector<double> polynomialRegression(const vector<double>& x, const vector<double>& y, int degree) {
    int n = x.size();
    int terms = degree + 1;
    
    // Inisialisasi matriks augmented [A|b]
    vector<vector<double>> A(terms, vector<double>(terms + 1, 0.0));
    
    // Hitung matriks normal equation
    for (int i = 0; i < terms; i++) {
        for (int j = 0; j < terms; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += pow(x[k], i + j);
            }
            A[i][j] = sum;
        }
        
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += y[k] * pow(x[k], i);
        }
        A[i][terms] = sum;
    }
    
    // Pake eliminasi Gauss buat dapetin koefisien
    return gaussElimination(A, terms);
}

/**
 * Fungsi buat evaluasi nilai polinomial di titik x
 * @param coeffs Vektor koefisien polinomial
 * @param x Nilai x yang mau dievaluasi
 * @return Nilai polinomial di titik x
 */
double evaluatePolynomial(const vector<double>& coeffs, double x) {
    double result = 0.0;
    for (size_t i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

/**
 * Fungsi buat ubah tahun jadi nilai relatif (biar stabil secara numerik)
 * @param year Tahun asli
 * @return Tahun relatif (mulai dari 0)
 */
double normalizeYear(int year, int baseYear) {
    return year - baseYear;
}

/**
 * Fungsi buat cetak persamaan polinomial
 * @param coeffs Vektor koefisien polinomial
 * @param baseYear Tahun dasar buat normalisasi
 */
string printPolynomialEquation(const vector<double>& coeffs, int baseYear) {
    stringstream equation;
    equation << fixed << setprecision(6);
    
    equation << "y = ";
    bool firstTerm = true;
    
    for (int i = coeffs.size() - 1; i >= 0; i--) {
        double coeff = coeffs[i];
        
        if (abs(coeff) < 1e-10) continue; // Skip koefisien yang kecil banget
        
        if (coeff > 0 && !firstTerm) {
            equation << " + ";
        } else if (coeff < 0 && !firstTerm) {
            equation << " - ";
            coeff = -coeff;
        }
        
        if (i == 0 || abs(coeff - 1.0) > 1e-10 || i == coeffs.size() - 1) {
            equation << coeff;
        }
        
        if (i >= 1) {
            equation << "x";
            if (i > 1) {
                equation << "^" << i;
            }
        }
        
        firstTerm = false;
    }
    
    equation << " (dimana x = tahun - " << baseYear << ")";
    return equation.str();
}

/**
 * Fungsi utama buat fitting kurva dan estimasi
 */
int main() {
    // Baca data dari file CSV (ganti sama nama file yang sesuai)
    vector<YearData> allData = readCSV("Data Tugas Pemrograman A.csv");
    
    // Pisahin data buat populasi dan persentase internet
    vector<double> years, population, internet;
    int baseYear = 1960; // Tahun dasar buat normalisasi
    
    for (const auto& data : allData) {
        double normalizedYear = normalizeYear(data.year, baseYear);
        years.push_back(normalizedYear);
        population.push_back(data.population);
        internet.push_back(data.internet_percentage);
    }
    
    // Interpolasi buat data internet perlu pendekatan khusus karena nilai 0 sebelum 1994
    vector<double> internetYears, internetValues;
    for (size_t i = 0; i < years.size(); i++) {
        if (allData[i].year >= 1994) { // Cuma pake data setelah 1994 buat internet
            internetYears.push_back(years[i]);
            internetValues.push_back(internet[i]);
        }
    }
    
    // Tentuin derajat polinomial yang cocok berdasarkan data
    int populationDegree = 2; // Derajat buat populasi (model kuadratik)
    int internetDegree = 3;   // Derajat buat internet (model kubik)
    
    // Lakuin regresi polinomial
    vector<double> populationCoeffs = polynomialRegression(years, population, populationDegree);
    vector<double> internetCoeffs = polynomialRegression(internetYears, internetValues, internetDegree);
    
    // Estimasi nilai yang ilang
    double pop2005 = evaluatePolynomial(populationCoeffs, normalizeYear(2005, baseYear));
    double pop2006 = evaluatePolynomial(populationCoeffs, normalizeYear(2006, baseYear));
    double pop2015 = evaluatePolynomial(populationCoeffs, normalizeYear(2015, baseYear));
    double pop2016 = evaluatePolynomial(populationCoeffs, normalizeYear(2016, baseYear));
    
    double internet2005 = evaluatePolynomial(internetCoeffs, normalizeYear(2005, baseYear));
    double internet2006 = evaluatePolynomial(internetCoeffs, normalizeYear(2006, baseYear));
    double internet2015 = evaluatePolynomial(internetCoeffs, normalizeYear(2015, baseYear));
    double internet2016 = evaluatePolynomial(internetCoeffs, normalizeYear(2016, baseYear));
    
    // Estimasi buat tahun 2030 dan 2035
    double pop2030 = evaluatePolynomial(populationCoeffs, normalizeYear(2030, baseYear));
    double internet2035 = evaluatePolynomial(internetCoeffs, normalizeYear(2035, baseYear));
    double pop2035 = evaluatePolynomial(populationCoeffs, normalizeYear(2035, baseYear));
    
    // Cetak hasil
    cout << "=============================================================" << endl;
    cout << "       ESTIMASI DATA POPULASI DAN PENGGUNA INTERNET INDONESIA" << endl;
    cout << "=============================================================" << endl;
    cout << endl;
    
    cout << "1. ESTIMASI DATA YANG HILANG:" << endl;
    cout << "-----------------------------" << endl;
    cout << "a. Populasi Indonesia tahun 2005: " << fixed << setprecision(0) << pop2005 << " jiwa" << endl;
    cout << "b. Populasi Indonesia tahun 2006: " << fixed << setprecision(0) << pop2006 << " jiwa" << endl;
    cout << "c. Populasi Indonesia tahun 2015: " << fixed << setprecision(0) << pop2015 << " jiwa" << endl;
    cout << "d. Populasi Indonesia tahun 2016: " << fixed << setprecision(0) << pop2016 << " jiwa" << endl;
    cout << endl;
    
    cout << "e. Persentase pengguna Internet Indonesia tahun 2005: " << fixed << setprecision(4) << internet2005 << "%" << endl;
    cout << "f. Persentase pengguna Internet Indonesia tahun 2006: " << fixed << setprecision(4) << internet2006 << "%" << endl;
    cout << "g. Persentase pengguna Internet Indonesia tahun 2015: " << fixed << setprecision(4) << internet2015 << "%" << endl;
    cout << "h. Persentase pengguna Internet Indonesia tahun 2016: " << fixed << setprecision(4) << internet2016 << "%" << endl;
    cout << endl;
    
    cout << "2. PERSAMAAN POLINOMIAL:" << endl;
    cout << "------------------------" << endl;
    cout << "a. Persentase pengguna Internet Indonesia:" << endl;
    cout << "   " << printPolynomialEquation(internetCoeffs, baseYear) << endl;
    cout << endl;
    
    cout << "b. Pertumbuhan populasi Indonesia:" << endl;
    cout << "   " << printPolynomialEquation(populationCoeffs, baseYear) << endl;
    cout << endl;
    
    cout << "3. ESTIMASI MASA DEPAN:" << endl;
    cout << "----------------------" << endl;
    cout << "a. Populasi Indonesia tahun 2030: " << fixed << setprecision(0) << pop2030 << " jiwa" << endl;
    cout << "b. Pengguna Internet Indonesia tahun 2035: " << fixed << setprecision(0) << internet2035 / 100 * pop2035 << " jiwa" << endl;
    cout << "   (Persentase: " << fixed << setprecision(4) << internet2035 << "% dari populasi " << fixed << setprecision(0) << pop2035 << " jiwa)" << endl;
    
    return 0;
}