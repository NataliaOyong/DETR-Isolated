"""
Main application entry point untuk DETR Isolated Mathematical Expression Detector
Menggunakan MVC (Model-View-Controller) architecture
"""
import streamlit as st
import torch

from controllers.model_controller import ModelController
from controllers.detection_controller import DetectionController
from views.detection_view import DetectionView
from views.training_view import TrainingView

# --- Constants ---
DEFAULT_MODEL_WEIGHTS = "detr_isolated100_weights.pth"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Try to import PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# --- Page Configuration ---
st.set_page_config(
    page_title="DETR Isolated Mathematical Expression Detector",
    layout="wide"
)

# --- PDF Support Check ---
if not PDF_SUPPORT:
    st.error(
        "Peringatan: Pustaka `PyMuPDF` tidak ditemukan. Fitur upload PDF dinonaktifkan. "
        "Untuk mengaktifkan, jalankan: `pip install PyMuPDF`",
        icon="⚠️"
    )

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #2ca02c;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown('<h1 class="main-header">DETR Isolated Mathematical Expression Detector</h1>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Navigasi")
    app_mode = st.radio("Pilih Halaman", ["Deteksi Ekspresi", "Training Model"])
    
    st.header("📋 Informasi")
    st.info("""
    - **Deteksi Ekspresi:** Upload gambar (JPG, PNG) atau PDF untuk dideteksi.
    - **Training Model:** Lakukan fine-tuning pada model DETR menggunakan dataset Anda.
    """)

# --- Model Loading (Singleton Pattern) ---
@st.cache_resource
def load_model_cached(weights_path: str, device: str):
    """
    Cache model loading untuk menghindari reload setiap refresh
    
    Args:
        weights_path: Path ke model weights
        device: Device untuk inference
        
    Returns:
        tuple: (model, image_processor)
    """
    model_controller = ModelController(weights_path=weights_path, device=device)
    return model_controller.load()

# Initialize model loading
if 'model_loaded' not in st.session_state:
    with st.spinner("Memuat model DETR..."):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            st.session_state.device = device
            
            model, image_processor = load_model_cached(DEFAULT_MODEL_WEIGHTS, device)
            
            # Initialize controllers
            detection_controller = DetectionController(
                model=model,
                image_processor=image_processor,
                device=device,
                confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
            )
            
            st.session_state.model = model
            st.session_state.image_processor = image_processor
            st.session_state.detection_controller = detection_controller
            st.session_state.model_loaded = True
            
            st.sidebar.success(f"Model berhasil dimuat! (Device: {device})")
        except Exception as e:
            st.session_state.model_loaded = False
            st.sidebar.error(f"Error memuat model: {str(e)}")

# --- Page Routing ---
if app_mode == "Deteksi Ekspresi":
    if st.session_state.get('model_loaded', False):
        detection_view = DetectionView()
        detection_controller = st.session_state.detection_controller
        detection_view.render(detection_controller)
    else:
        st.error("❌ Model belum berhasil dimuat. Silakan refresh halaman atau periksa file weights.")

elif app_mode == "Training Model":
    training_view = TrainingView()
    training_view.render()