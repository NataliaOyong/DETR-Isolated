# DETR Isolated Mathematical Expression Detector

Aplikasi web untuk mendeteksi ekspresi matematika terisolasi dalam dokumen digital menggunakan model DETR (Detection Transformer) dengan arsitektur **Model-View-Controller (MVC)**.

## 🎯 Fitur

### Detection (Deteksi Ekspresi)
- 📤 Upload dokumen dalam format **JPG, PNG, atau PDF**
- 🔍 Deteksi otomatis ekspresi matematika terisolasi
- 📊 Tampilan hasil deteksi dengan bounding box
- 📑 Crop per formula yang terdeteksi
- ⬇️ Unduh hasil deteksi:
  - Gambar hasil deteksi dengan bounding box
  - Crop per formula (individual)
  - Semua formula dalam format ZIP + metadata.json

### Training Model
- 🏋️ Fine-tuning model DETR menggunakan dataset dari Roboflow
- ⚙️ Konfigurasi hyperparameter:
  - Jumlah Epochs
  - Batch Size
  - Learning Rate (Head & Backbone) - **Fixed: 1e-4 dan 1e-5**
- 📈 Visualisasi loss training
- 💾 Download model weights hasil training

## 📋 Requirements & Libraries

- Python 3.8 atau lebih tinggi
- Dependencies (lihat `requirements.txt`)

### Penjelasan Library Utama

Berikut adalah library utama yang digunakan dalam project ini beserta fungsinya:

| Library | Kegunaan |
| :--- | :--- |
| **Streamlit** | Framework utama untuk membuat antarmuka web (UI) yang interaktif tanpa perlu coding HTML/CSS. |
| **PyTorch (`torch`)** | Library Deep Learning untuk komputasi tensor dan pelatihan model neural network. |
| **Torchvision** | Menyediakan dataset, arsitektur model, dan transformasi gambar untuk computer vision. |
| **Transformers** | Menyediakan model pre-trained DETR (Detection Transformer) dari Hugging Face. |
| **PyTorch Lightning** | Wrapper untuk PyTorch yang merapikan struktur kode training dan validasi. |
| **Roboflow** | Mengelola dan mendownload dataset gambar yang sudah dianotasi. |
| **Pillow (PIL)** | Library standar untuk membuka, memanipulasi, dan menyimpan file gambar. |
| **Matplotlib** | Membuat visualisasi grafik, seperti kurva loss training. |
| **Supervision** | Membantu visualisasi hasil deteksi (bounding box) agar terlihat lebih rapi dan estetis. |
| **PyMuPDF** | Menangani pemrosesan file PDF, memungkinkan ekstraksi gambar dari dokumen. |
| **OpenCV** (`headless`) | Library computer vision untuk manipulasi gambar tingkat lanjut. |

## 🚀 Instalasi

1. Clone repository atau download project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pastikan file model weights ada:
   - File `detr_isolated100_weights.pth` harus ada di direktori root

## 💻 Menjalankan Aplikasi

### Cara 1: Menggunakan run.py (Disarankan)
```bash
python run.py
```

Script ini akan:
- ✅ Mengecek semua dependencies
- ✅ Mengecek keberadaan model weights
- ✅ Menjalankan aplikasi Streamlit secara otomatis
- ✅ Membuka browser otomatis

### Cara 2: Langsung menggunakan Streamlit
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📖 Cara Penggunaan

### 1. Deteksi Ekspresi Matematika

1. Pilih halaman **"Deteksi Ekspresi"** di sidebar
2. Upload gambar atau PDF:
   - Format yang didukung: **JPG, PNG, PDF**
   - Untuk PDF, pilih halaman yang ingin diproses (jika multi-page)
3. Klik tombol **"Jalankan Deteksi"**
4. Hasil deteksi akan ditampilkan:
   - Gambar dengan bounding box
   - Crop per formula yang terdeteksi
   - Confidence score untuk setiap deteksi
5. Unduh hasil:
   - Unduh gambar hasil deteksi
   - Unduh formula individual
   - Unduh semua formula dalam format ZIP (dengan metadata.json)

### 2. Training Model

1. Pilih halaman **"Training Model"** di sidebar
2. Isi konfigurasi Roboflow:
   - API Key Roboflow
   - Workspace Name
   - Project Name
   - Project Version
3. Atur hyperparameter:
   - **Jumlah Epochs**: 1-200 (default: 10)
   - **Batch Size**: 2, 4, 8, 16, atau 32 (default: 16)
   - **Learning Rate**: Fixed (Head = 1e-4, Backbone = 1e-5)
4. Klik tombol **"Mulai Training"**
5. Tunggu proses training selesai
6. Unduh model weights hasil training (.pth)
7. Lihat grafik loss training

## 📁 Struktur Project (MVC Architecture)

```
DETR-Isolated/
├── app.py                    # Entry point utama (Streamlit app)
├── run.py                    # Script untuk running aplikasi
├── requirements.txt          # Dependencies Python
├── README.md                 # Dokumentasi
├── detr_isolated100_weights.pth  # Model weights
│
├── models/                   # Models (Data & Business Logic)
│   ├── __init__.py
│   ├── detr_model.py         # Detr, CocoDetection, LossPlotCallback
│   └── training_config.py    # TrainingConfig dataclass
│
├── views/                    # Views (UI Components)
│   ├── __init__.py
│   ├── detection_view.py     # DetectionView - UI untuk detection
│   └── training_view.py      # TrainingView - UI untuk training
│
├── controllers/              # Controllers (Business Logic)
│   ├── __init__.py
│   ├── model_controller.py       # ModelController - Load/save model
│   ├── detection_controller.py   # DetectionController - Detection logic
│   └── training_controller.py    # TrainingController - Training orchestration
│
└── utils/                    # Utilities (Helper Functions)
    ├── __init__.py
    └── model_utils.py        # load_model, predict_isolated_expressions
```

### Arsitektur MVC

- **Models** (`models/`): Data structures dan model classes
  - `Detr`: PyTorch Lightning module untuk model DETR
  - `CocoDetection`: Dataset class untuk COCO format
  - `TrainingConfig`: Configuration dataclass
  
- **Views** (`views/`): UI components untuk Streamlit
  - `DetectionView`: UI untuk halaman detection
  - `TrainingView`: UI untuk halaman training
  
- **Controllers** (`controllers/`): Business logic dan orchestration
  - `ModelController`: Mengelola loading/saving model
  - `DetectionController`: Mengelola detection/inference
  - `TrainingController`: Mengelola training process
  
- **Utils** (`utils/`): Helper functions
  - `model_utils`: Utility functions untuk model operations

## ⚙️ Konfigurasi

### Default Settings

- **Model Weights**: `detr_isolated100_weights.pth`
- **Checkpoint**: `facebook/detr-resnet-50`
- **Confidence Threshold**: 0.5
- **Learning Rate (Head)**: 1e-4 (fixed)
- **Learning Rate (Backbone)**: 1e-5 (fixed)
- **Weight Decay**: 1e-4
- **Freeze Backbone**: True

### File Types yang Didukung

- **Images**: JPG, JPEG, PNG
- **Documents**: PDF (jika PyMuPDF terinstall)

## 🐛 Troubleshooting

### Error: "Model belum berhasil dimuat"

**Penyebab**: File model weights tidak ditemukan atau corrupt

**Solusi**:
- Pastikan file `detr_isolated100_weights.pth` ada di direktori root
- Periksa ukuran file (harus > 0 bytes)
- Pastikan file tidak corrupt

### Error: "transformers library not available"

**Solusi**:
```bash
pip install transformers
```

### Error: "PyMuPDF tidak ditemukan" (untuk PDF support)

**Solusi**:
```bash
pip install PyMuPDF
```

**Note**: Aplikasi tetap berfungsi tanpa PDF support, hanya tidak bisa upload PDF.

### Error: ModuleNotFoundError

**Penyebab**: Package tidak terinstall atau struktur direktori salah

**Solusi**:
```bash
pip install -r requirements.txt
```

Pastikan struktur direktori sesuai dengan dokumentasi di atas.

### Training Error: "API key tidak boleh kosong"

**Solusi**: Pastikan API Key Roboflow sudah diisi dengan benar.

## 📊 Model Architecture

Model menggunakan **Hugging Face Transformers DETR**:
- Base checkpoint: `facebook/detr-resnet-50`
- Model dibungkus dalam PyTorch Lightning untuk training
- Image processor: `DetrImageProcessor`
- Post-processing: Object detection dengan confidence thresholding

### Model Loading Process

1. Load image processor dari checkpoint Hugging Face
2. Load model weights dari file `.pth`
3. Handle format lama (state_dict only) dan format baru (dengan config)
4. Process state_dict keys untuk compatibility
5. Load ke model dan set ke eval mode

## 🔧 Development

### Menambahkan Fitur Baru

Dengan struktur MVC, menambahkan fitur baru lebih mudah:

1. **Model**: Tambahkan dataclass atau model class di `models/`
2. **View**: Tambahkan UI component di `views/`
3. **Controller**: Tambahkan business logic di `controllers/`
4. **Utils**: Tambahkan helper functions di `utils/` jika diperlukan

### Testing

Untuk testing individual components:

```python
# Test Model Controller
from controllers.model_controller import ModelController
model_ctrl = ModelController("weights.pth", device="cpu")
model, processor = model_ctrl.load()

# Test Detection Controller
from controllers.detection_controller import DetectionController
detection_ctrl = DetectionController(model, processor, device="cpu")
detections = detection_ctrl.detect(image)

# Test Training Controller
from models.training_config import TrainingConfig
from controllers.training_controller import TrainingController
config = TrainingConfig(api_key="...", workspace="...", project_name="...", version=1)
training_ctrl = TrainingController(config)
final_path, plot_path = training_ctrl.train()
```

## 📝 Notes

- Model weights harus ada sebelum menjalankan aplikasi
- Training memerlukan API Key Roboflow yang valid
- File hasil training akan disimpan di direktori root
- Logs training tersimpan di `lightning_logs/`
- Plot loss training tersimpan di `training_plots/`

## 🙏 Acknowledgments

- Hugging Face Transformers untuk model DETR
- PyTorch Lightning untuk training framework
- Streamlit untuk UI framework
- Roboflow untuk dataset management
