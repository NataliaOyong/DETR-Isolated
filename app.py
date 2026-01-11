"""
Main application entry point untuk DETR Isolated Mathematical Expression Detector
Menggunakan MVC (Model-View-Controller) architecture
"""
import streamlit as st
import torch
import os
import glob

from controllers.model_controller import ModelController
from controllers.detection_controller import DetectionController
from views.detection_view import DetectionView
from views.training_view import TrainingView

# --- Constants ---
DEFAULT_MODEL_WEIGHTS = "detr_isolated100_weights.pth"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def get_available_weight_files():
    """
    Mendapatkan semua file .pth yang tersedia di direktori root
    
    Returns:
        list: List of weight file names
    """
    weight_files = glob.glob("*.pth")
    return sorted(weight_files)


def get_default_weight_file():
    """
    Mendapatkan default weight file (detr_isolated100_weights.pth jika ada)
    
    Returns:
        str: Default weight file name atau None jika tidak ada
    """
    if os.path.exists(DEFAULT_MODEL_WEIGHTS):
        return DEFAULT_MODEL_WEIGHTS
    return None

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
    
    # Weight file selection (only show on Detection page)
    if app_mode == "Deteksi Ekspresi":
        st.header("⚙️ Model Weights")
        available_weights = get_available_weight_files()
        
        if not available_weights:
            st.warning("Tidak ada file weight (.pth) yang ditemukan!")
            if 'current_weight_file' in st.session_state:
                del st.session_state.current_weight_file
        else:
            # Initialize current_weight_file if not exists
            if 'current_weight_file' not in st.session_state:
                default_weight = get_default_weight_file()
                if default_weight and default_weight in available_weights:
                    st.session_state.current_weight_file = default_weight
                else:
                    st.session_state.current_weight_file = available_weights[0]
            
            # Determine default index for selectbox
            current_weight = st.session_state.current_weight_file
            if current_weight in available_weights:
                default_index = available_weights.index(current_weight)
            else:
                # If current weight not in list, use default
                default_weight = get_default_weight_file()
                if default_weight and default_weight in available_weights:
                    default_index = available_weights.index(default_weight)
                    st.session_state.current_weight_file = default_weight
                else:
                    default_index = 0
                    st.session_state.current_weight_file = available_weights[0]
            
            selected_weight = st.selectbox(
                "Pilih Model Weight",
                options=available_weights,
                index=default_index,
                help="Pilih file weight model yang akan digunakan untuk deteksi. Default: detr_isolated100_weights.pth",
                key="selected_weight_file"
            )
            
            # Store selected weight in session state and mark for reload if changed
            if st.session_state.current_weight_file != selected_weight:
                st.session_state.current_weight_file = selected_weight
                # Mark model for reload when weight changes
                if 'model_loaded' in st.session_state:
                    st.session_state.model_loaded = False
    
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

# Get selected weight file (from sidebar selection or default)
if app_mode == "Deteksi Ekspresi":
    # Get selected weight from session state
    if 'current_weight_file' in st.session_state:
        selected_weight = st.session_state.current_weight_file
    else:
        selected_weight = get_default_weight_file() or DEFAULT_MODEL_WEIGHTS
        st.session_state.current_weight_file = selected_weight
    
    # Initialize model loading
    # Check if model needs to be loaded/reloaded
    should_load_model = False
    if 'model_loaded' not in st.session_state:
        should_load_model = True
    elif not st.session_state.model_loaded:
        should_load_model = True
    elif 'current_weight_file' in st.session_state:
        # Check if the loaded model matches current weight file
        if st.session_state.get('loaded_weight_file') != st.session_state.current_weight_file:
            should_load_model = True
    
    if should_load_model and selected_weight and os.path.exists(selected_weight):
        with st.spinner(f"Memuat model DETR dari {selected_weight}..."):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                st.session_state.device = device
                
                model, image_processor = load_model_cached(selected_weight, device)
                
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
                st.session_state.loaded_weight_file = selected_weight
                
                st.sidebar.success(f"Model berhasil dimuat dari {selected_weight}! (Device: {device})")
            except Exception as e:
                st.session_state.model_loaded = False
                st.sidebar.error(f"Error memuat model: {str(e)}")
    elif selected_weight and not os.path.exists(selected_weight):
        st.session_state.model_loaded = False
        st.sidebar.error(f"File weight tidak ditemukan: {selected_weight}")

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