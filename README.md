# Eksperimen SML - Idlofi-Zahir-Rajaba

## 📊 Project Overview

Proyek ini adalah submission untuk **Kriteria 1: Melakukan Eksperimen terhadap Dataset Pelatihan** pada kelas Membangun Sistem Machine Learning (MSML) Dicoding.

Proyek ini bertujuan untuk melakukan preprocessing data secara otomatis pada **Customer Shopping Mall Transaction Dataset** menggunakan Python dan GitHub Actions.

## 👨‍💻 Author

- **Nama:** Idlofi-Zahir-Rajaba
- **Proyek:** Eksperimen_SML_Idlofi-Zahir-Rajaba
- **Kriteria:** 1 (Basic/Skilled/Advance)

## 📁 Dataset Information

### Sumber Dataset
Dataset ini diperoleh dari Kaggle: Customer Shopping Dataset

### Deskripsi Dataset
Dataset ini berisi informasi transaksi pelanggan di berbagai pusat perbelanjaan (mall) di Turki.

### Fitur-fitur Dataset
| Fitur | Tipe | Deskripsi |
|-------|------|------------|
| invoice_no | Object | Nomor invoice transaksi |
| customer_id | Object | ID unik pelanggan |
| gender | Object | Jenis kelamin (Female/Male) |
| age | Integer | Usia pelanggan (18-69 tahun) |
| category | Object | Kategori produk |
| quantity | Integer | Jumlah item yang dibeli (1-5) |
| price | Float | Harga per item |
| payment_method | Object | Metode pembayaran |
| invoice_date | Object | Tanggal transaksi |
| shopping_mall | Object | Nama mall tempat transaksi |

### Target Variable
- **total_amount** = quantity × price (total belanja)

## 📁 Project Structure

```text
Eksperimen_SML_Idlofi-Zahir-Rajaba/
├── .github/
│   └── workflows/
│       └── preprocessing.yml               # GitHub Actions workflow (CI Pipeline)
│
├── customer_shopping_data_raw/
│   └── customer_shopping_data.csv          # Raw dataset dari Kaggle (99,457 baris)
│
├── preprocessing/
│   ├── Eksperimen_Idlofi-Zahir-Rajaba.ipynb # Tahapan manual EDA & Preprocessing (Template MSML)
│   ├── automate_Idlofi-Zahir-Rajaba.py     # Skrip otomatisasi preprocessing (Skilled/Advance)
│   └── customer_shopping_data_preprocessing.csv # Hasil akhir dataset yang siap dilatih
│
└── README.md                               # Dokumentasi proyek akhir

## 🔧 Preprocessing Steps

Script `automate_Idlofi-Zahir-Rajaba.py` melakukan langkah-langkah berikut:

### 1. Data Loading
- Memuat dataset dari `customer_shopping_data_raw/customer_shopping_data.csv`

### 2. Target Variable Creation
- Membuat `total_amount = quantity × price`

### 3. Drop Unnecessary Columns
- Menghapus: `invoice_no`, `customer_id`, `invoice_date`

### 4. Handle Missing Values
- Cek dan isi missing values (jika ada)
- Numerik: diisi dengan median
- Kategorikal: diisi dengan modus

### 5. Remove Duplicates
- Menghapus data duplikat
- Hasil: dari 99,457 menjadi 54,200 baris

### 6. Handle Outliers (IQR Capping)
- Deteksi outliers pada `price` dan `total_amount`
- Menggunakan metode IQR (Interquartile Range)
- Capping outliers ke batas atas/bawah

### 7. Feature Engineering
- Membuat `price_vs_category_avg` = price / category_average_price

### 8. Encoding Categorical Variables
- Label Encoding untuk semua kolom kategorikal:
  - `gender` (2 categories)
  - `category` (8 categories)
  - `payment_method` (3 categories)
  - `shopping_mall` (10 categories)

### 9. Feature Scaling
- Menggunakan `StandardScaler`
- Mean = 0, Std = 1 untuk semua fitur

## 📊 Dataset Statistics

### Before Preprocessing
- **Total rows:** 99,457
- **Total columns:** 10
- **Missing values:** 0
- **Duplicates:** 45,257

### After Preprocessing
- **Total rows:** 54,200
- **Total columns:** 8 (7 features + 1 target)
- **Missing values:** 0
- **Duplicates:** 0

### Features After Preprocessing
1. `gender` (encoded: 0=Female, 1=Male)
2. `age` (scaled)
3. `category` (encoded: 0-7)
4. `quantity` (scaled)
5. `price` (scaled, outliers capped)
6. `payment_method` (encoded: 0-2)
7. `shopping_mall` (encoded: 0-9)
8. `price_vs_category_avg` (new feature, scaled)
9. **`total_amount`** (target variable, scaled)

## 🚀 How to Run

### Method 1: Manual Run (Local)

```bash
# Clone repository
git clone https://github.com/your-username/Eksperimen_SML_Idlofi-Zahir-Rajaba.git
cd Eksperimen_SML_Idlofi-Zahir-Rajaba

# Install dependencies
pip install pandas numpy scikit-learn

# Run preprocessing script
cd preprocessing
python automate_Idlofi-Zahir-Rajaba.py
