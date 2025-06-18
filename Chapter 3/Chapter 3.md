Chapter 3: Notebook Klasifikasi

Chapter ini akan membahas sistem klasifikasi. Kita akan menggunakan dataset MNIST, membangun classifier, mengevaluasi kinerjanya, dan menjelajahi berbagai aspek penting dalam tugas klasifikasi.
1. Dataset MNIST

MNIST adalah dataset berisi 70.000 gambar kecil tulisan tangan angka oleh siswa sekolah menengah dan pegawai Biro Sensus AS. Setiap gambar diberi label sesuai dengan angka yang diwakilinya.

Mengunduh Dataset
Scikit-Learn menyediakan fungsi untuk mengunduh dataset populer, termasuk MNIST.

Setiap gambar berukuran 28x28 piksel, yang diratakan menjadi vektor 1D dengan 784 fitur.

Membagi Data
Dataset MNIST sudah dibagi menjadi training set (60.000 gambar pertama) dan test set (10.000 gambar terakhir).
2. Melatih Classifier Biner

Kita akan menyederhanakan masalah dengan mencoba mengidentifikasi hanya satu digit, misalnya angka 5. Ini adalah contoh binary classifier, yang mampu membedakan antara dua kelas: "5" dan "bukan-5".
3. Ukuran Kinerja (Performance Measures)

Mengevaluasi classifier lebih rumit daripada regressor.

Akurasi dengan Cross-Validation
Akurasi seringkali bukan ukuran kinerja yang baik untuk classifier, terutama jika datasetnya skewed (tidak seimbang).
Classifier bodoh tersebut mencapai akurasi lebih dari 90% hanya karena sekitar 10% dari gambar adalah angka 5.

Confusion Matrix
Cara yang jauh lebih baik untuk mengevaluasi kinerja adalah dengan melihat confusion matrix. Ini menghitung berapa kali instance kelas A salah diklasifikasikan sebagai kelas B.

Setiap baris mewakili kelas aktual, sementara setiap kolom mewakili kelas yang diprediksi. Baris pertama adalah kelas negatif ("bukan-5") dan baris kedua adalah kelas positif ("5").

Precision dan Recall

    Precision: Akurasi dari prediksi positif. precision = TP / (TP + FP).

Recall (Sensitivity): Rasio instance positif yang terdeteksi dengan benar. recall = TP / (TP + FN).

Precision/Recall Trade-off
Meningkatkan precision cenderung mengurangi recall, dan sebaliknya. Ini disebut precision/recall trade-off. Kita bisa mengubah threshold keputusan untuk memilih trade-off yang kita inginkan.

Kurva ROC
Kurva Receiver Operating Characteristic (ROC) adalah alat umum lainnya untuk classifier biner. Kurva ini memplot true positive rate (recall) terhadap false positive rate (FPR). Semakin dekat kurva ke sudut kiri atas, semakin baik classifiernya. Area di bawah kurva (AUC) adalah ukuran kinerja yang baik: AUC=1 untuk classifier sempurna, dan AUC=0.5 untuk classifier acak.
4. Klasifikasi Multiclass

Classifier multiclass dapat membedakan lebih dari dua kelas. Beberapa algoritma (seperti SGDClassifier dan RandomForestClassifier) dapat menangani banyak kelas secara native. Algoritma lain (seperti SVC) adalah classifier biner, tetapi Scikit-Learn secara otomatis menggunakan strategi one-versus-the-rest (OvR) atau one-versus-one (OvO) untuk melakukan klasifikasi multiclass.
5. Analisis Kesalahan

Setelah menemukan model yang menjanjikan, cara untuk memperbaikinya adalah dengan menganalisis jenis kesalahan yang dibuatnya. Kita bisa memplot confusion matrix untuk melihat kelas mana yang sering salah diklasifikasikan.
Dengan menormalisasi confusion matrix berdasarkan jumlah gambar di setiap kelas, kita bisa fokus pada tingkat kesalahan.

6. Klasifikasi Multilabel dan Multioutput

    Klasifikasi Multilabel: Sebuah sistem klasifikasi yang dapat menghasilkan beberapa label biner untuk setiap instance. Misalnya, mengidentifikasi beberapa orang dalam satu foto.

Klasifikasi Multioutput: Generalisasi dari klasifikasi multilabel di mana setiap label bisa berupa multiclass. Contohnya adalah sistem yang membersihkan noise dari gambar, di mana setiap piksel adalah sebuah label dengan nilai dari 0 hingga 255.
