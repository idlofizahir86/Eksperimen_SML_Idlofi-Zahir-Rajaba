#!/usr/bin/env python3
"""
Automated Data Preprocessing Script for Shopping Mall Transaction Dataset
Author: Idlofi-Zahir-Rajaba
Project: Eksperimen_SML_Idlofi-Zahir-Rajaba

Usage: 
    python automate_Idlofi-Zahir-Rajaba.py

Atau dengan parameter:
    python automate_Idlofi-Zahir-Rajaba.py --input ../customer_shopping_data_raw/customer_shopping_data.csv --output customer_shopping_data_preprocessing.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

class ShoppingDataPreprocessor:
    """
    Kelas untuk melakukan preprocessing data transaksi mall secara otomatis
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'total_amount'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Memuat dataset dari file path
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Data berhasil dimuat dari {file_path}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            sys.exit(1)
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Membuat target variable total_amount = quantity * price
        """
        print("\n" + "="*60)
        print("💰 MEMBUAT TARGET VARIABLE")
        print("="*60)
        df['total_amount'] = df['quantity'] * df['price']
        print(f"  Total Amount range: {df['total_amount'].min():.2f} - {df['total_amount'].max():.2f}")
        print(f"  Average Total Amount: {df['total_amount'].mean():.2f}")
        print(f"  Median Total Amount: {df['total_amount'].median():.2f}")
        return df
    
    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghapus kolom yang tidak diperlukan untuk modeling
        """
        print("\n" + "="*60)
        print("🗑️ MENGHAPUS KOLOM TIDAK PERLU")
        print("="*60)
        columns_to_drop = ['invoice_no', 'customer_id', 'invoice_date']
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"  Kolom dihapus: {existing_cols}")
        else:
            print(f"  Tidak ada kolom yang perlu dihapus")
        
        print(f"  Kolom tersisa: {list(df.columns)}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menangani missing values
        """
        print("\n" + "="*60)
        print("📊 MENANGANI MISSING VALUES")
        print("="*60)
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing > 0:
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                        print(f"  {col}: diisi dengan median = {df[col].median()}")
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        print(f"  {col}: diisi dengan modus = {df[col].mode()[0]}")
            
            final_missing = df.isnull().sum().sum()
            print(f"\n  Missing values awal: {initial_missing}")
            print(f"  Missing values akhir: {final_missing}")
        else:
            print("  Tidak ada missing values!")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghapus data duplikat
        """
        print("\n" + "="*60)
        print("🔄 MENGHAPUS DATA DUPLIKAT")
        print("="*60)
        initial_shape = df.shape
        duplicates_count = df.duplicated().sum()
        
        if duplicates_count > 0:
            df = df.drop_duplicates()
            final_shape = df.shape
            print(f"  Data duplikat ditemukan: {duplicates_count:,}")
            print(f"  Shape sebelum: {initial_shape}")
            print(f"  Shape setelah: {final_shape}")
            print(f"  Data yang tersisa: {final_shape[0]:,} baris")
        else:
            print(f"  Tidak ada data duplikat!")
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menangani outliers menggunakan metode IQR (capping)
        """
        print("\n" + "="*60)
        print("📈 MENANGANI OUTLIERS (IQR CAPPING)")
        print("="*60)
        
        outlier_cols = ['price', 'total_amount']
        
        for col in outlier_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Hitung outliers sebelum capping
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_count = len(outliers)
                outliers_percentage = (outliers_count / len(df)) * 100
                
                # Simpan nilai sebelum capping
                before_max = df[col].max()
                
                # Capping outliers
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                # Nilai setelah capping
                after_max = df[col].max()
                
                print(f"\n  {col}:")
                print(f"    Outliers: {outliers_count:,} ({outliers_percentage:.2f}%)")
                print(f"    IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"    Max value: {before_max:.2f} → {after_max:.2f}")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Membuat fitur-fitur baru yang informatif
        """
        print("\n" + "="*60)
        print("🔧 FEATURE ENGINEERING")
        print("="*60)
        
        # Feature: Price relative to category average
        if 'category' in df.columns and 'price' in df.columns:
            category_price_mean = df.groupby('category')['price'].transform('mean')
            df['price_vs_category_avg'] = df['price'] / category_price_mean
            
            # Handle infinite or NaN values
            df['price_vs_category_avg'] = df['price_vs_category_avg'].replace([np.inf, -np.inf], np.nan)
            df['price_vs_category_avg'].fillna(1.0, inplace=True)
            
            print(f"\n  ✓ price_vs_category_avg (relative price to category average)")
            print(f"    Range: [{df['price_vs_category_avg'].min():.3f}, {df['price_vs_category_avg'].max():.3f}]")
            print(f"    Mean: {df['price_vs_category_avg'].mean():.3f}")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mengencode semua fitur kategorikal menggunakan Label Encoding
        """
        print("\n" + "="*60)
        print("🏷️ ENCODING KATEGORIKAL")
        print("="*60)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            print("  Tidak ada kolom kategorikal!")
            return df
        
        print(f"  Kolom kategorikal ditemukan: {categorical_cols}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
            # Tampilkan mapping
            mapping = dict(enumerate(le.classes_))
            print(f"\n  ✓ {col}: {len(le.classes_)} unique values")
            # Hanya tampilkan 5 mapping pertama jika terlalu banyak
            if len(mapping) <= 10:
                print(f"    Mapping: {mapping}")
            else:
                print(f"    Mapping: {dict(list(mapping.items())[:5])} ...")
        
        return df
    
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Melakukan standard scaling pada fitur
        """
        print("\n" + "="*60)
        print("📏 FEATURE SCALING (StandardScaler)")
        print("="*60)
        
        # Simpan feature names
        feature_names = X.columns.tolist()
        
        # Perform scaling
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        print(f"  Scaler type: StandardScaler (mean=0, std=1)")
        print(f"  Features scaled: {len(feature_names)} kolom")
        print(f"  Mean after scaling: {X_scaled_df.mean().mean():.6f}")
        print(f"  Std after scaling: {X_scaled_df.std().mean():.6f}")
        
        return X_scaled_df
    
    def save_preprocessing_info(self, output_path: str, df_final: pd.DataFrame, feature_cols: list):
        """
        Menyimpan informasi preprocessing ke file
        """
        # Buat file info dengan nama yang sama tapi ekstensi .txt
        info_path = output_path.replace('.csv', '_info.txt')
        
        with open(info_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PREPROCESSING INFORMATION\n")
            f.write(f"Author: Idlofi-Zahir-Rajaba\n")
            f.write(f"Project: Eksperimen_SML_Idlofi-Zahir-Rajaba\n")
            f.write("="*60 + "\n\n")
            
            f.write("TARGET VARIABLE:\n")
            f.write(f"  Name: {self.target_column}\n")
            f.write(f"  Formula: quantity * price\n\n")
            
            f.write("FEATURES:\n")
            for i, col in enumerate(feature_cols, 1):
                f.write(f"  {i}. {col}\n")
            f.write(f"\nTotal features: {len(feature_cols)}\n\n")
            
            f.write("LABEL ENCODERS MAPPING:\n")
            for col, le in self.label_encoders.items():
                f.write(f"\n  {col}:\n")
                for code, label in enumerate(le.classes_):
                    f.write(f"    {code} -> {label}\n")
            
            f.write("\nSCALER INFORMATION:\n")
            f.write(f"  Type: StandardScaler\n")
            f.write(f"  Mean (per feature): {self.scaler.mean_.tolist()}\n")
            f.write(f"  Scale (per feature): {self.scaler.scale_.tolist()}\n")
            
            f.write("\nFINAL DATASET STATISTICS:\n")
            f.write(f"  Shape: {df_final.shape}\n")
            f.write(f"  Target mean: {df_final[self.target_column].mean():.4f}\n")
            f.write(f"  Target std: {df_final[self.target_column].std():.4f}\n")
        
        print(f"\n  ✓ Info preprocessing disimpan ke: {info_path}")
        return info_path
    
    def preprocess(self, input_path: str, output_path: str):
        """
        Pipeline lengkap preprocessing data
        """
        print("\n" + "="*70)
        print("🚀 MEMULAI PREPROCESSING DATA")
        print(f"   Dataset: Shopping Mall Transaction")
        print(f"   Author: Idlofi-Zahir-Rajaba")
        print("="*70)
        
        # 1. Load data
        df = self.load_data(input_path)
        
        # 2. Create target variable
        df = self.create_target_variable(df)
        
        # 3. Drop unnecessary columns
        df = self.drop_unnecessary_columns(df)
        
        # 4. Handle missing values
        df = self.handle_missing_values(df)
        
        # 5. Remove duplicates
        df = self.remove_duplicates(df)
        
        # 6. Handle outliers
        df = self.handle_outliers(df)
        
        # 7. Feature engineering
        df = self.feature_engineering(df)
        
        # 8. Encode categorical (PENTING: dilakukan SEBELUM scaling)
        df = self.encode_categorical(df)
        
        # 9. Pisahkan fitur dan target
        feature_cols = [col for col in df.columns if col != self.target_column]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[self.target_column]
        
        # 10. Scale features
        X_scaled = self.scale_features(X)
        
        # 11. Gabungkan kembali untuk output lengkap
        df_final = X_scaled.copy()
        df_final[self.target_column] = y.values
        
        # 12. Save hasil preprocessing
        output_dir = os.path.dirname(output_path)
        if output_dir and output_dir != '':
            os.makedirs(output_dir, exist_ok=True)
        
        df_final.to_csv(output_path, index=False)
        
        # 13. Save preprocessing info
        info_path = self.save_preprocessing_info(output_path, df_final, feature_cols)
        
        # 14. Final report
        print("\n" + "="*70)
        print("✅ PREPROCESSING SELESAI!")
        print("="*70)
        print(f"\n📁 Output file: {output_path}")
        print(f"📁 Info file: {info_path}")
        print(f"📊 Final dataset shape: {df_final.shape}")
        print(f"🎯 Target column: {self.target_column}")
        print(f"📝 Total features: {len(feature_cols)}")
        print(f"\nFeature list:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2}. {col}")
        
        print(f"\n📊 Target statistics (scaled):")
        print(f"  Mean: {df_final[self.target_column].mean():.6f}")
        print(f"  Std: {df_final[self.target_column].std():.6f}")
        
        print("\n✨ Data siap untuk digunakan dalam training model (Kriteria 2)!")
        print("="*70)
        
        return df_final

def main():
    # Set default paths sesuai struktur direktori
    default_input = "../customer_shopping_data_raw/customer_shopping_data.csv"
    default_output = "customer_shopping_data_preprocessing.csv"
    
    parser = argparse.ArgumentParser(
        description='Automated Data Preprocessing untuk Shopping Mall Transaction Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Contoh penggunaan:

  # Tanpa parameter (menggunakan default path)
  python automate_Idlofi-Zahir-Rajaba.py

  # Dengan parameter lengkap
  python automate_Idlofi-Zahir-Rajaba.py --input {default_input} --output {default_output}

  # Dari root project
  cd Eksperimen_SML_Idlofi-Zahir-Rajaba
  python preprocessing/automate_Idlofi-Zahir-Rajaba.py
        """
    )
    
    parser.add_argument('--input', '-i', type=str, 
                       default=default_input,
                       help=f'Path ke file CSV input (default: {default_input})')
    parser.add_argument('--output', '-o', type=str, 
                       default=default_output,
                       help=f'Path untuk menyimpan hasil preprocessing (default: {default_output})')
    
    args = parser.parse_args()
    
    # Validasi input file
    if not os.path.exists(args.input):
        print(f"✗ Error: File '{args.input}' tidak ditemukan!")
        print("\nPastikan struktur direktori Anda:")
        print("  Eksperimen_SML_Idlofi-Zahir-Rajaba/")
        print("  ├── customer_shopping_data_raw/")
        print("  │   └── customer_shopping_data.csv")
        print("  └── preprocessing/")
        print("      └── automate_Idlofi-Zahir-Rajaba.py (file ini)")
        sys.exit(1)
    
    # Run preprocessing
    preprocessor = ShoppingDataPreprocessor()
    df_final = preprocessor.preprocess(
        input_path=args.input,
        output_path=args.output
    )
    
    print("\n🎉 Preprocessing berhasil! Silakan lanjut ke Kriteria 2.")
    print(f"\n📌 File hasil preprocessing: {args.output}")
    print(f"📌 File ini akan digunakan untuk modelling.py di Kriteria 2")

if __name__ == "__main__":
    main()