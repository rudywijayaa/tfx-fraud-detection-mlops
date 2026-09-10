<<<<<<< HEAD
# Submission 1: Deteksi Fraud Transaksi Kartu Kredit

Nama: Rudy Wijaya

Username Dicoding: `rudy_wijaya_7Xnd`

## Struktur Project

```text
data/                         Dataset CSV
src/                          Module Transform, Tuner, dan Trainer
notebooks/                    Notebook pipeline dan pengujian serving
deployment/                   Docker Compose dan konfigurasi TensorFlow Serving
schema_pipeline/              Artefak pipeline dan ML Metadata
serving_model/                SavedModel hasil Pusher
submission/                   Arsip submission
requirements.txt              Dependensi Python
```

|                         | Deskripsi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                 | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Dataset berisi transaksi kartu kredit dengan 284.807 baris dan 31 kolom. Fitur terdiri dari `Time`, `Amount`, `V1` sampai `V28`, serta label `Class`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Masalah                 | Transaksi fraud jumlahnya sangat sedikit dibandingkan transaksi normal. Dari seluruh data, terdapat 492 transaksi fraud dan 284.315 transaksi normal, sehingga proporsi fraud hanya sekitar 0,17%. Tantangannya adalah menemukan transaksi fraud sebanyak mungkin tanpa menganggap terlalu banyak transaksi normal sebagai fraud.                                                                                                                                                                                                                                                                                                                                                                                    |
| Solusi machine learning | Solusi yang dibuat adalah model klasifikasi biner berbasis neural network. Pipeline menggunakan TFX Tuner untuk memilih hidden units, dropout, dan learning rate berdasarkan PR-AUC, lalu model dideploy melalui TensorFlow Serving.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Target                  | Target utama adalah memperoleh recall minimal 0,90 agar sebagian besar transaksi fraud dapat ditemukan. Model juga diharapkan memiliki PR-AUC yang baik karena metrik ini lebih sesuai untuk dataset dengan ketidakseimbangan kelas yang besar.                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Metode pengolahan       | Data dibaca menggunakan `CsvExampleGen`, kemudian dianalisis dengan `StatisticsGen`, dibuatkan skema menggunakan `SchemaGen`, dan diperiksa anomali dengan `ExampleValidator`. Seluruh fitur numerik dinormalisasi menggunakan z-score melalui TensorFlow Transform. Data hasil transformasi disimpan dalam TFRecord terkompresi dan digunakan oleh Trainer. Karena jumlah fraud jauh lebih sedikit, kelas fraud diberi bobot 578 sedangkan kelas normal diberi bobot 1.                                                                                                                                                                                                                                             |
| Arsitektur model        | Model menerima 30 fitur numerik, yaitu `V1` sampai `V28`, `Amount`, dan `Time`. Semua input digabungkan, lalu diproses oleh Dense layer berukuran 64 neuron dengan aktivasi ReLU, Dropout 20%, Dense layer berukuran 32 neuron dengan aktivasi ReLU, Dropout 20%, dan output satu neuron dengan aktivasi sigmoid. Model dilatih menggunakan optimizer Adam dengan learning rate 0,001 dan binary cross-entropy sebagai loss.                                                                                                                                                                                                                                                                                         |
| Metrik evaluasi         | Metrik yang digunakan adalah recall, precision, PR-AUC, binary accuracy, loss, dan jumlah contoh evaluasi. Recall dipakai untuk memastikan transaksi fraud tidak banyak terlewat, sedangkan precision dan PR-AUC membantu melihat kualitas prediksi pada kondisi kelas yang tidak seimbang. TFX Evaluator menetapkan batas minimal recall sebesar 0,90 sebelum model dapat berstatus blessed.                                                                                                                                                                                                                                                                                                                        |
| Performa model          | Berdasarkan hasil evaluasi TFMA pada 95.286 data evaluasi, model memperoleh recall **0,9500**, PR-AUC **0,7146**, precision **0,0553**, binary accuracy **0,9692**, dan loss **0,1184**. Recall sebesar 0,95 berhasil melewati target minimal 0,90, sehingga model memenuhi kriteria utama untuk menemukan transaksi fraud. Namun, precision yang masih sekitar 5,53% menunjukkan bahwa banyak transaksi yang diprediksi fraud ternyata bukan fraud atau terjadi false positive. Model sudah cocok sebagai penyaring awal untuk memberi tanda pada transaksi mencurigakan, tetapi masih perlu penyempurnaan threshold dan analisis lanjutan sebelum digunakan sebagai keputusan otomatis tanpa pemeriksaan tambahan. |

## Alur Pipeline

Pipeline dibuat dengan komponen TFX berikut:

1. `CsvExampleGen` membaca dataset CSV.
2. `StatisticsGen` menghasilkan ringkasan statistik data.
3. `SchemaGen` menyusun skema fitur dan label.
4. `ExampleValidator` memeriksa anomali pada dataset.
5. `Transform` menormalisasi fitur numerik dan menyimpan transform graph.
6. `Tuner` mencari hyperparameter terbaik berdasarkan PR-AUC.
7. `Trainer` melatih model neural network menggunakan artefak `best_hyperparameters`.
8. `Resolver` mencari model blessed sebelumnya sebagai baseline.
9. `Evaluator` membandingkan performa model dengan kriteria yang ditentukan.
10. `Pusher` mengekspor model yang lolos evaluasi ke folder serving.
11. TensorFlow Serving memuat model berversi melalui Docker Compose.

## Kesimpulan

Model berhasil mencapai recall 95% dan melewati target yang ditentukan. Hasil ini menunjukkan bahwa model cukup baik dalam menangkap transaksi fraud, tetapi precision yang rendah masih menjadi perhatian utama. Pengembangan berikutnya dapat berfokus pada pemilihan threshold yang lebih sesuai, penanganan false positive, dan evaluasi tambahan pada data transaksi terbaru.

## Deployment TensorFlow Serving

Setelah pipeline selesai dan Pusher menghasilkan SavedModel, jalankan:

```powershell
docker compose -f deployment/docker-compose.serving.yml up -d
```

REST API tersedia di `http://localhost:8501/v1/models/fraud_detection:predict`. Notebook `notebooks/rudy_wijaya_7Xnd-testing.ipynb` menyiapkan payload `instances` dan menguji endpoint tersebut. Deployment ini adalah TensorFlow Serving, bukan custom Flask server.

Untuk menghentikan service:

```powershell
docker compose -f deployment/docker-compose.serving.yml down
```
=======
# tfx-fraud-detection-mlops
>>>>>>> 596f62f687465b3d65a965ce57bd8184b6bbe6e2
