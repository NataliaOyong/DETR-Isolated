"""
View untuk detection page di Streamlit
"""
import streamlit as st
import io
import os
import json
import zipfile
from PIL import Image
from typing import Optional

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class DetectionView:
    """
    View class untuk halaman detection
    """
    
    def __init__(self):
        """Initialize DetectionView"""
        self.default_confidence = 0.5
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = self.default_confidence
    
    @staticmethod
    def draw_boxes_on_image(image: Image.Image, detections: list) -> Image.Image:
        """
        Draw bounding boxes pada gambar
        
        Args:
            image: PIL Image object
            detections: List of detection dictionaries
            
        Returns:
            PIL Image dengan bounding boxes
        """
        from PIL import ImageDraw
        
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
    
    @staticmethod
    def process_uploaded_file(uploaded_file) -> Optional[Image.Image]:
        """
        Process uploaded file (PDF atau image)
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            PIL Image atau None jika error
        """
        if not uploaded_file:
            return None
        
        # Save filename
        st.session_state.uploaded_filename = uploaded_file.name
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == ".pdf":
            if not PDF_SUPPORT:
                st.error("Gagal memuat library `PyMuPDF`. Silakan install dengan `pip install PyMuPDF`.")
                return None
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
                        return pdf_pages[page_num_selected - 1]
                    elif pdf_pages:
                        return pdf_pages[0]
            
            except Exception as e:
                st.error(f"Gagal memproses file PDF: {e}")
                return None
        else:
            # Image file
            try:
                return Image.open(uploaded_file).convert('RGB')
            except Exception as e:
                st.error(f"Gagal memproses gambar: {e}")
                return None
    
    def render_upload_section(self) -> Optional[Image.Image]:
        """
        Render upload section
        
        Returns:
            PIL Image jika berhasil, None jika belum ada upload
        """
        st.header("📤 Upload Dokumen")
        
        # Define allowed file types - hanya JPG, PNG, dan PDF
        file_types = ['jpg', 'jpeg', 'png']
        if PDF_SUPPORT:
            file_types.append('pdf')
        
        uploaded_file = st.file_uploader(
            "Pilih gambar atau PDF",
            type=file_types,
            help="Upload gambar (JPG, PNG) atau PDF yang berisi ekspresi matematika",
            on_change=lambda: st.session_state.pop('detections', None)
        )
        
        if uploaded_file:
            image_to_process = self.process_uploaded_file(uploaded_file)
            if image_to_process:
                st.image(image_to_process, caption="Dokumen yang akan diproses")
                st.session_state.original_image = image_to_process
                return image_to_process
        
        return None
    
    def render_detection_section(self, image_to_process: Optional[Image.Image], 
                                detection_controller) -> Optional[list]:
        """
        Render detection section dengan button untuk menjalankan deteksi
        
        Args:
            image_to_process: PIL Image untuk diproses
            detection_controller: DetectionController instance
            
        Returns:
            List of detections atau None
        """
        st.header("🚀 Deteksi Ekspresi")
        
        if st.button("Jalankan Deteksi", type="primary", disabled=(image_to_process is None)):
            if st.session_state.get('model_loaded', False):
                with st.status("Memulai proses deteksi...", expanded=True) as status:
                    try:
                        status.update(label="Langkah 1/3: Preprocessing gambar...")
                        import time
                        time.sleep(1)
                        
                        detections = detection_controller.detect(image_to_process)
                        st.session_state.detections = detections
                        
                        status.update(label="Langkah 2/3: Menjalankan inferensi model...")
                        time.sleep(1)
                        
                        if detections:
                            status.update(label="Langkah 3/3: Memproses hasil deteksi...")
                            annotated_image = self.draw_boxes_on_image(image_to_process, detections)
                            st.session_state.annotated_image = annotated_image
                            time.sleep(1)
                        
                        status.update(label="Deteksi selesai!", state="complete", expanded=False)
                        return detections
                    
                    except Exception as e:
                        status.update(label="Deteksi Gagal", state="error")
                        st.error(f"❌ Terjadi error saat proses deteksi: {str(e)}")
                        st.session_state.detections = None
                        return None
            else:
                st.error("Model belum berhasil dimuat. Coba refresh halaman.")
                return None
        
        return st.session_state.get('detections', None)
    
    def render_results_section(self, detections: Optional[list]):
        """
        Render results section dengan detections
        
        Args:
            detections: List of detection dictionaries
        """
        st.header("📊 Hasil Deteksi")
        
        if detections is None:
            if 'detections' in st.session_state and st.session_state.detections is None:
                st.warning("Proses deteksi gagal. Silakan coba lagi.")
            else:
                st.info("Upload gambar dan klik tombol 'Jalankan Deteksi' untuk melihat hasilnya.")
            return
        
        if not detections:
            st.info("Tidak ditemukan ekspresi matematika yang dapat diekstraksi.")
            return
        
        st.success(f"✅ Berhasil! Ditemukan {len(detections)} ekspresi matematika.")
        
        annotated_image = st.session_state.get('annotated_image')
        if annotated_image:
            st.image(annotated_image, caption="Hasil Deteksi dengan Bounding Box")
        
        self.render_export_section(detections)
    
    def render_export_section(self, detections: list):
        """
        Render export section dengan download buttons
        
        Args:
            detections: List of detection dictionaries
        """
        st.header("💾 Ekspor Hasil")
        
        annotated_image = st.session_state.get('annotated_image')
        if not annotated_image:
            return
        
        # Download annotated image
        img_bytes = io.BytesIO()
        annotated_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        original_filename = os.path.splitext(
            st.session_state.get('uploaded_filename', 'dokumen')
        )[0]
        
        st.download_button(
            label="⬇️ Unduh Gambar Hasil Deteksi",
            data=img_bytes,
            file_name=f"hasil_deteksi_{original_filename}.jpg",
            mime="image/jpeg"
        )
        
        # Crop per formula
        st.subheader("📑 Crop Per Formula")
        for idx, detection in enumerate(detections):
            with st.expander(f"Formula #{idx + 1} (Confidence: {detection['confidence']:.2%})", expanded=True):
                cropped_img = detection['cropped_image']
                st.image(cropped_img, caption=f"Formula #{idx + 1}", use_container_width=True)
                
                img_bytes = io.BytesIO()
                cropped_img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                st.download_button(
                    label=f"Unduh Formula #{idx + 1}",
                    data=img_bytes,
                    file_name=f"formula_{idx + 1}.jpg",
                    mime="image/jpeg",
                    key=f"download_formula_{idx}"
                )
        
        # Download all as ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            metadata = []
            for idx, detection in enumerate(detections):
                img_bytes = io.BytesIO()
                detection['cropped_image'].save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                zip_file.writestr(f"formula_{idx + 1}.jpg", img_bytes.read())
                
                metadata.append({
                    "index": idx + 1,
                    "bbox": detection['bbox'],
                    "confidence": detection['confidence'],
                    "label": detection['label']
                })
            
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        zip_buffer.seek(0)
        st.download_button(
            label="⬇️ Unduh Semua (ZIP + metadata.json)",
            data=zip_buffer,
            file_name=f"semua_formula_{original_filename}.zip",
            mime="application/zip",
            key="download_all_zip"
        )
    
    def render(self, detection_controller):
        """
        Render seluruh detection page
        
        Args:
            detection_controller: DetectionController instance
        """
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image_to_process = self.render_upload_section()
            detections = self.render_detection_section(image_to_process, detection_controller)
        
        with col2:
            self.render_results_section(detections)
