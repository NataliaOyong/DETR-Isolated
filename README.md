# DETR Isolated Mathematical Expression Detector

Aplikasi web untuk mendeteksi ekspresi matematika terisolasi dalam dokumen digital menggunakan model DETR (Detection Transformer).

## Fitur

- 📤 Upload gambar dokumen dalam format JPEG
- 🔍 Deteksi otomatis ekspresi matematika terisolasi
- 📊 Tampilan hasil deteksi (crop bagian yang terdeteksi)
- ⬇️ Unduh formula matematika hasil deteksi dalam format JPEG
- ⚙️ Pengaturan confidence threshold

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

### Cara 1: Menggunakan run.py (Disarankan)
```bash
python run.py
```

### Cara 2: Langsung menggunakan Streamlit
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## Cara Penggunaan

1. Upload gambar dokumen yang berisi ekspresi matematika (format JPEG)
2. Klik tombol "Deteksi Ekspresi Matematika"
3. Hasil deteksi akan ditampilkan sebagai crop dari setiap formula yang terdeteksi
4. Gunakan tombol unduh untuk menyimpan formula individual atau semua formula sekaligus

## Struktur File

- `run.py`: Script untuk menjalankan aplikasi (entry point)
- `app.py`: Aplikasi Streamlit utama
- `model_utils.py`: Utility functions untuk loading model dan inference
- `detr_isolated_model_100_weights.pth`: Model weights
- `requirements.txt`: Dependencies Python

## Catatan

Pastikan model weights (`detr_isolated_model_100_weights.pth`) berada di direktori yang sama dengan aplikasi.

## Troubleshooting

### Error: "transformers library not available"

Jika Anda mendapatkan error ini, install library transformers:
```bash
pip install transformers
```

### Model Architecture

Model ini menggunakan **Hugging Face Transformers DETR** (`DetrForObjectDetection`). 
- Base checkpoint: `facebook/detr-resnet-50`
- Model dibungkus dalam PyTorch Lightning untuk training
- Model weights disimpan dengan `torch.save(model.state_dict(), ...)`

Aplikasi akan otomatis:
1. Load model dari Hugging Face checkpoint
2. Load custom weights dari file `.pth`
3. Menggunakan `DetrImageProcessor` untuk preprocessing

