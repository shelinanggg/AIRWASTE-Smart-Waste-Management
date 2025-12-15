# AIRWASTE (Airlangga Integrated and Revolutionary Waste System)

AIRWASTE adalah implementasi **Smart Waste Management System** berbasis *Computer Vision* yang bertujuan untuk membantu optimalisasi pengelolaan sampah di lingkungan **Universitas Airlangga**. Sistem ini menggunakan model **Convolutional Neural Network (CNN)** untuk melakukan klasifikasi jenis sampah dan disajikan dalam bentuk aplikasi web interaktif menggunakan **Streamlit**.

---

## Fitur Utama
- Klasifikasi sampah berbasis citra menggunakan CNN
- Antarmuka web interaktif dengan Streamlit
- Mendukung pengelolaan model dan dataset eksternal
- Mudah dijalankan secara lokal

---

## Teknologi yang Digunakan
- Python 3.x
- TensorFlow / Keras
- Streamlit
- NumPy, OpenCV, Pillow

---

## Struktur Proyek
```
AIRWASTE-Smart-Waste-Management/
│── app.py                 # File utama aplikasi Streamlit
│── requirements.txt       # Daftar dependensi
│── README.md              # Dokumentasi proyek
│── LICENSE                # Lisensi proyek
│── .gitignore             # Konfigurasi file yang diabaikan Git
```

Dataset dan model **tidak disertakan** di dalam repository GitHub karena memiliki ukuran besar dan tersedia secara publik melalui Kaggle.

---

## Dataset dan Model
Dataset dan model CNN dapat diperoleh melalui Kaggle:
- Dataset: [Plastic Waste Classification Dataset](https://www.kaggle.com/models/hardikksankhla/waste-classification-cnn-model/)
- Model: [Pretrained CNN Plastic Waste Classification Model](https://www.kaggle.com/datasets/techsash/waste-classification-data)

Setelah diunduh, sesuaikan path dataset dan model di dalam file `app.py`.

---

## Cara Menjalankan Aplikasi

### 1. Clone Repository
```bash
git clone https://github.com/shelinanggg/AIRWASTE-Smart-Waste-Management.git
cd AIRWASTE-Smart-Waste-Management
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser melalui `http://localhost:8501`.

---

## Catatan Penting
- Pastikan dataset dan model sudah tersedia secara lokal sebelum menjalankan aplikasi.
- Repository ini hanya berisi **source code**, bukan file dataset atau model.

---

## Lisensi
Proyek ini dilisensikan di bawah lisensi yang tercantum pada file `LICENSE`.

---

## Penulis
- Maria Shelina Angie (187231006)
- Cokorda Istri Trisna Shanti Maharani Pemayun	(187231011)
- Jasmine Mumtaz (502510310010)

Proyek ini dikembangkan sebagai bagian dari penerapan sistem cerdas untuk mendukung pengelolaan lingkungan kampus yang berkelanjutan.
