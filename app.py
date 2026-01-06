import streamlit as st
import torch
from PIL import Image, ImageDraw
import io
import time
import os
from model_utils import load_model, predict_isolated_expressions
from training_ui import render_training_page

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

# --- Helper Function ---
def draw_boxes_on_image(image, detections):
    """Draws bounding boxes on a copy of the image."""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    for det in detections:
        box = det['bbox']
        confidence = det['confidence']
        
        # Draw bounding box
        draw.rectangle(box, outline="red", width=2)
        
        # Prepare text and position
        label = f"{confidence:.2f}"
        text_position = (box[0], box[1] - 10)
        
        # Draw text background
        text_bbox = draw.textbbox(text_position, label)
        draw.rectangle(text_bbox, fill="red")
        
        # Draw text
        draw.text(text_position, label, fill="white")
        
    return img_with_boxes

def render_detection_page():
    # --- Main Content ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 1. Upload Dokumen")
        
        # Define allowed file types
        file_types = ['jpg', 'jpeg', 'png', 'tiff']
        if PDF_SUPPORT:
            file_types.append('pdf')

        uploaded_file = st.file_uploader(
            "Pilih gambar atau PDF",
            type=file_types,
            help="Upload gambar atau PDF yang berisi ekspresi matematika",
            on_change=lambda: st.session_state.pop('detections', None) # Clear old results on new upload
        )
        
        image_to_process = None
        if uploaded_file:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extension == ".pdf":
                if not PDF_SUPPORT:
                     st.error("Gagal memuat library `PyMuPDF`. Silakan install dengan `pip install PyMuPDF`.")
                else:
                    try:
                        with st.spinner("Mengubah PDF menjadi gambar..."):
                            pdf_bytes = uploaded_file.read()
                            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                            pdf_pages = []
                            for page_num in range(len(pdf_doc)):
                                page = pdf_doc.load_page(page_num)
                                pix = page.get_pixmap()
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                pdf_pages.append(img)
                            
                        st.session_state.pdf_pages = pdf_pages
                        
                        if len(pdf_pages) > 1:
                            page_num_selected = st.selectbox(
                                f"Pilih halaman PDF (Total {len(pdf_pages)} halaman)", 
                                options=range(1, len(pdf_pages) + 1),
                                format_func=lambda x: f"Halaman {x}"
                            )
                            image_to_process = pdf_pages[page_num_selected - 1]
                        elif pdf_pages:
                            image_to_process = pdf_pages[0]
                    
                    except Exception as e:
                        st.error(f"Gagal memproses file PDF: {e}")

            else: # It's an image file
                image_to_process = Image.open(uploaded_file).convert('RGB')
            
            if image_to_process:
                st.image(image_to_process, caption="Dokumen yang akan diproses")
                st.session_state.original_image = image_to_process

            st.header("🚀 2. Deteksi Ekspresi")
            if st.button("Jalankan Deteksi", type="primary", disabled=(image_to_process is None)):
                if st.session_state.get('model_loaded', False):
                    with st.status("Memulai proses deteksi...", expanded=True) as status:
                        try:
                            status.update(label="Langkah 1/3: Preprocessing gambar...")
                            time.sleep(1)

                            detections = predict_isolated_expressions(
                                st.session_state.model,
                                st.session_state.image_processor,
                                image_to_process,
                                st.session_state.confidence_threshold,
                                device=st.session_state.device
                            )
                            st.session_state.detections = detections
                            
                            status.update(label="Langkah 2/3: Menjalankan inferensi model...")
                            time.sleep(1)

                            if detections:
                                status.update(label="Langkah 3/3: Memproses hasil deteksi...")
                                annotated_image = draw_boxes_on_image(image_to_process, detections)
                                st.session_state.annotated_image = annotated_image
                                time.sleep(1)
                            
                            status.update(label="Deteksi selesai!", state="complete", expanded=False)

                        except Exception as e:
                            status.update(label="Deteksi Gagal", state="error")
                            st.error(f"❌ Terjadi error saat proses deteksi: {str(e)}")
                            st.session_state.detections = None
                else:
                    st.error("Model belum berhasil dimuat. Coba refresh halaman.")

    with col2:
        st.header("📊 3. Hasil Deteksi")
        
        if 'detections' in st.session_state:
            detections = st.session_state.detections
            
            if detections is None:
                st.warning("Proses deteksi gagal. Silakan coba lagi.")
            elif not detections:
                st.info("Tidak ditemukan ekspresi matematika yang dapat diekstraksi.")
            else:
                st.success(f"✅ Berhasil! Ditemukan {len(detections)} ekspresi matematika.")
                
                annotated_image = st.session_state.annotated_image
                st.image(annotated_image, caption="Hasil Deteksi dengan Bounding Box")
                
                st.header("💾 4. Ekspor Hasil")
                img_bytes = io.BytesIO()
                annotated_image.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                # Use original filename without extension for the download
                original_filename = os.path.splitext(uploaded_file.name)[0]
                st.download_button(
                    label="⬇️ Unduh Gambar Hasil Deteksi",
                    data=img_bytes,
                    file_name=f"hasil_deteksi_{original_filename}.jpg",
                    mime="image/jpeg"
                )
        else:
            st.info("👆 Upload gambar dan klik tombol 'Jalankan Deteksi' untuk melihat hasilnya di sini.")


# --- Main App ---
st.markdown('<h1 class="main-header">DETR Isolated Mathematical Expression Detector</h1>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🧭 Navigasi")
    app_mode = st.radio("Pilih Halaman", ["Deteksi Ekspresi", "Training Model"])
    
    st.header("📋 Informasi")
    st.info("""
    - **Deteksi Ekspresi:** Upload gambar (JPG, PNG) atau PDF untuk dideteksi.
    - **Training Model:** Lakukan fine-tuning pada model DETR menggunakan dataset Anda.
    """)
    
    if app_mode == "Deteksi Ekspresi":
        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.05,
            help="Hanya deteksi dengan skor di atas ambang batas ini yang akan ditampilkan."
        )

# --- Model Loading ---
if 'model' not in st.session_state:
    with st.spinner("Memuat model DETR..."):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            st.session_state.device = device
            model, image_processor = load_model("detr_isolated100_weights.pth", device=device)
            st.session_state.model = model
            st.session_state.image_processor = image_processor
            st.session_state.model_loaded = True
            st.sidebar.success(f"Model berhasil dimuat! (Device: {device})")
        except Exception as e:
            st.session_state.model_loaded = False
            st.sidebar.error(f"Error memuat model: {str(e)}")


# --- Page Routing ---
if app_mode == "Deteksi Ekspresi":
    render_detection_page()
elif app_mode == "Training Model":
    render_training_page()

