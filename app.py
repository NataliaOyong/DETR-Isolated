import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import io
from model_utils import load_model, predict_isolated_expressions

# Page configuration
st.set_page_config(
    page_title="DETR Isolated Mathematical Expression Detector",
    layout="wide"
)

# Custom CSS for better UI
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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">DETR Isolated Mathematical Expression Detector</h1>', unsafe_allow_html=True)

# Sidebar for model info
with st.sidebar:
    st.header("📋 Informasi")
    st.info("""
    **Cara Penggunaan:**
    1. Upload gambar dokumen dalam format JPEG
    2. Sistem akan mendeteksi ekspresi matematika terisolasi
    3. Hasil deteksi akan ditampilkan sebagai crop
    4. Unduh formula yang terdeteksi
    """)
    
    # Threshold di-hardcode ke 0.5
    confidence_threshold = 0.5

# Initialize session state
if 'model' not in st.session_state:
    with st.spinner("Memuat model DETR..."):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            st.session_state.device = device
            model, image_processor = load_model("detr_isolated100_weights.pth", num_classes=2, device=device)
            st.session_state.model = model
            st.session_state.image_processor = image_processor
            st.session_state.model_loaded = True
            st.sidebar.success(f"Model berhasil dimuat! (Device: {device})")
        except Exception as e:
            st.session_state.model_loaded = False
            st.sidebar.error(f"Error memuat model: {str(e)}")
            with st.sidebar.expander("Bantuan Troubleshooting", expanded=True):
                st.markdown("""
                **Kemungkinan solusi:**
                1. Pastikan file weights ada di direktori yang sama
                2. Jika model adalah custom DETR, letakkan model class definition di `model_definition.py`
                3. Pastikan model disimpan dengan format yang benar
                4. Lihat README.md untuk informasi lebih lanjut
                """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar dokumen (JPEG)",
        type=['jpg', 'jpeg'],
        help="Upload gambar dokumen yang berisi ekspresi matematika"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar Original", use_container_width=True)
        
        # Process button
        if st.button("Deteksi Ekspresi Matematika", type="primary"):
            if st.session_state.model_loaded:
                with st.spinner("Memproses gambar..."):
                    try:
                        # Run prediction
                        device = st.session_state.get('device', 'cpu')
                        detections = predict_isolated_expressions(
                            st.session_state.model,
                            st.session_state.image_processor,
                            image,
                            confidence_threshold,
                            device=device
                        )
                        
                        # Store results in session state
                        st.session_state.detections = detections
                        st.session_state.original_image = image
                        st.success(f"✅ Ditemukan {len(detections)} ekspresi matematika!")
                    except Exception as e:
                        st.error(f"❌ Error saat deteksi: {str(e)}")
            else:
                st.error("Model belum dimuat. Silakan refresh halaman.")

with col2:
    st.header("📊 Hasil Deteksi")
    
    if 'detections' in st.session_state and len(st.session_state.detections) > 0:
        detections = st.session_state.detections
        original_image = st.session_state.original_image
        
        st.info(f"**Total deteksi:** {len(detections)} ekspresi matematika")
        
        # Display all cropped detections
        for idx, detection in enumerate(detections):
            with st.expander(f"Formula #{idx + 1} (Confidence: {detection['confidence']:.2%})", expanded=True):
                # Display cropped image
                cropped_img = detection['cropped_image']
                st.image(cropped_img, caption=f"Formula #{idx + 1}", use_container_width=True)
                
                # Download button for individual formula
                img_bytes = io.BytesIO()
                cropped_img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                st.download_button(
                    label=f"Unduh Formula #{idx + 1}",
                    data=img_bytes,
                    file_name=f"formula_{idx + 1}.jpg",
                    mime="image/jpeg",
                    key=f"download_{idx}"
                )
        
        # Download all formulas as zip
        if len(detections) > 1:
            import zipfile
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for idx, detection in enumerate(detections):
                    img_bytes = io.BytesIO()
                    detection['cropped_image'].save(img_bytes, format='JPEG')
                    img_bytes.seek(0)
                    zip_file.writestr(f"formula_{idx + 1}.jpg", img_bytes.read())
            zip_buffer.seek(0)
            
            st.download_button(
                label="⬇️ Unduh Semua Formula (ZIP)",
                data=zip_buffer,
                file_name="all_formulas.zip",
                mime="application/zip",
                key="download_all"
            )
    else:
        st.info("👆 Upload gambar dan klik tombol deteksi untuk melihat hasil")

