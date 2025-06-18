1. Memahami Gambaran Besar (Big Picture)

    Tujuan Bisnis: Pertanyaan pertama adalah apa tujuan bisnisnya. Dalam kasus ini, output model (prediksi harga median sebuah distrik) akan dimasukkan ke sistem ML lain yang akan menentukan kelayakan investasi di suatu area. Ini disebut data pipeline, di mana beberapa komponen pemrosesan data berjalan secara berurutan.

Pembingkaian Masalah:

    Ini adalah tugas supervised learning karena data pelatihan memiliki label (harga median rumah).

Ini adalah tugas regresi karena kita diminta untuk memprediksi sebuah nilai. Secara spesifik, ini adalah masalah multiple regression (karena menggunakan banyak fitur untuk membuat prediksi) dan univariate regression (karena kita hanya memprediksi satu nilai per distrik).
Data cukup kecil untuk muat di memori, sehingga batch learning adalah pilihan yang tepat.

Ukuran Kinerja (Performance Measure):

    Ukuran kinerja yang umum untuk masalah regresi adalah Root Mean Square Error (RMSE). Ini memberikan gambaran tentang seberapa banyak kesalahan yang dibuat sistem, dengan bobot yang lebih tinggi untuk kesalahan besar. RMSE(X,h)=m1​i=1∑m​(h(x(i))−y(i))2​

Alternatif lain adalah Mean Absolute Error (MAE), yang lebih tidak sensitif terhadap outlier. MAE(X,h)=m1​i=1∑m​​h(x(i))−y(i)​

2. Mendapatkan Data

Langkah ini mencakup pengaturan lingkungan kerja dan mengunduh data.

Membuat Ruang Kerja & Mengunduh Data
Kita akan menulis fungsi Python untuk mengambil data secara otomatis. Ini berguna jika data sering berubah atau jika kita perlu menginstal dataset di beberapa mesin.

Python

import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Kemudian kita muat datanya menggunakan pandas
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

Melihat Struktur Data
Setelah data dimuat, kita perlu melakukan inspeksi cepat.

    housing.head(): Menampilkan lima baris pertama data.

housing.info(): Memberikan deskripsi singkat data, termasuk jumlah baris, tipe setiap atribut, dan jumlah nilai non-null. Dari sini kita tahu ada 20.640 instance dan atribut total_bedrooms memiliki 207 nilai yang hilang.
housing["ocean_proximity"].value_counts(): Menunjukkan bahwa ocean_proximity adalah atribut kategorikal.
housing.describe(): Menampilkan ringkasan atribut numerik seperti rata-rata, standar deviasi, dan persentil.
housing.hist(bins=50, figsize=(20,15)): Membuat histogram untuk setiap atribut numerik untuk mendapatkan gambaran distribusi data.

Membuat Test Set
Penting untuk menyisihkan test set sejak awal untuk menghindari data snooping bias, yaitu bias yang terjadi jika kita melihat data uji dan secara tidak sadar memilih model yang berkinerja baik pada data tersebut.

Untuk memastikan test set tetap konsisten di setiap run, bahkan setelah memperbarui dataset, kita dapat menggunakan identifier unik setiap instance untuk memutuskan apakah instance tersebut masuk ke dalam test set atau tidak.

Metode yang lebih baik adalah stratified sampling. Jika kita tahu bahwa suatu fitur (misalnya, median_income) sangat penting, kita ingin memastikan bahwa test set kita representatif terhadap berbagai kategori dari fitur tersebut.

Python

from sklearn.model_selection import train_test_split

# Membagi data secara acak
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Menggunakan stratified sampling berdasarkan kategori pendapatan
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

3. Menjelajahi dan Memvisualisasikan Data

Sekarang kita akan menjelajahi training set untuk mendapatkan wawasan lebih dalam.

    Visualisasi Data Geografis: Karena data memiliki informasi lintang (latitude) dan bujur (longitude), kita bisa membuat scatterplot untuk melihat distribusi geografisnya.

Python

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

Mencari Korelasi: Kita bisa menghitung koefisien korelasi standar (Pearson's r) antara setiap pasangan atribut menggunakan metode corr(). Fungsi scatter_matrix() dari pandas juga sangat berguna untuk memvisualisasikan korelasi antara beberapa atribut.

Eksperimen dengan Kombinasi Atribut: Langkah terakhir dalam eksplorasi adalah mencoba berbagai kombinasi atribut. Misalnya, rooms_per_household atau bedrooms_per_room mungkin lebih informatif daripada jumlah total kamar atau kamar tidur.

4. Mempersiapkan Data

Kita akan menulis fungsi untuk setiap transformasi data agar mudah direproduksi, digunakan kembali, dan diuji.

    Pembersihan Data: Sebagian besar algoritma ML tidak bisa bekerja dengan fitur yang hilang. Kita punya tiga pilihan: (1) menyingkirkan distrik yang sesuai, (2) menyingkirkan seluruh atribut, atau (3) mengisi nilai yang hilang (dengan nol, rata-rata, median, dll.). Scikit-Learn menyediakan SimpleImputer untuk menangani nilai yang hilang.

Menangani Atribut Teks dan Kategorikal: Kita perlu mengubah kategori dari teks menjadi angka. Untuk ini, kita bisa menggunakan OneHotEncoder dari Scikit-Learn untuk membuat satu atribut biner per kategori.
Feature Scaling: Algoritma ML tidak berkinerja baik ketika atribut numerik input memiliki skala yang sangat berbeda. Dua cara umum untuk menskalakan fitur adalah min-max scaling dan standardization. Scikit-Learn menyediakan MinMaxScaler dan StandardScaler untuk ini.
Transformation Pipelines: Scikit-Learn menyediakan kelas Pipeline untuk membantu dengan urutan transformasi. Untuk menangani kolom numerik dan kategorikal secara terpisah, kita dapat menggunakan ColumnTransformer.

5. Memilih dan Melatih Model

Setelah data dipersiapkan, kita siap untuk melatih model.

    Melatih dan Mengevaluasi di Training Set: Kita akan mulai dengan melatih beberapa model seperti LinearRegression dan DecisionTreeRegressor.

Evaluasi yang Lebih Baik Menggunakan Cross-Validation: Untuk mendapatkan estimasi kinerja yang lebih baik, kita gunakan K-fold cross-validation dari Scikit-Learn. Ini membantu kita melihat apakah model overfitting.
Kita juga akan mencoba model yang lebih kuat seperti RandomForestRegressor, yang merupakan contoh Ensemble Learning.

6. Menyempurnakan (Fine-Tune) Model

Setelah memiliki daftar pendek model yang menjanjikan, kita perlu menyempurnakannya.

    Grid Search: Scikit-Learn menyediakan GridSearchCV untuk mencari kombinasi hyperparameter terbaik menggunakan cross-validation.

Randomized Search: Ketika ruang pencarian hyperparameter besar, RandomizedSearchCV seringkali lebih disukai daripada grid search karena lebih efisien dalam menjelajahi berbagai nilai.
Menganalisis Model Terbaik dan Kesalahannya: Kita bisa melihat pentingnya setiap fitur (feature_importances_) untuk mendapatkan wawasan lebih lanjut.
Mengevaluasi di Test Set: Setelah kita yakin dengan model final kita, kita mengevaluasinya di test set untuk mendapatkan estimasi generalization error.

7. & 8. Presentasi, Peluncuran, dan Pemeliharaan

Langkah terakhir adalah mempresentasikan solusi Anda, menyoroti apa yang telah Anda pelajari dan bagaimana solusi tersebut mencapai tujuan bisnis. Setelah mendapat persetujuan, Anda akan meluncurkan model ke lingkungan produksi. Ini melibatkan penulisan kode pemantauan untuk memeriksa kinerja sistem secara berkala dan memicu peringatan jika kinerjanya menurun, serta melatih kembali model secara teratur dengan data baru.

