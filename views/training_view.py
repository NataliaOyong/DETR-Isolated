"""
View untuk training page di Streamlit
"""
import streamlit as st
import os
from models.training_config import TrainingConfig
from controllers.training_controller import TrainingController


class TrainingView:
    """
    View class untuk halaman training
    """
    
    def render(self):
        """Render training page"""
        st.header("🏋️ Model Training")
        
        st.info("""
        **Halaman ini digunakan untuk melakukan training (fine-tuning) model DETR.**
        - Masukkan informasi dataset dari Roboflow.
        - Atur hyperparameter training.
        - Klik 'Mulai Training' untuk memulai proses.
        """)
        
        with st.form("training_form"):
            st.subheader("📦 Roboflow Dataset Configuration")
            api_key = st.text_input(
                "Roboflow API Key", 
                type="password", 
                help="Gunakan API key Roboflow Anda."
            )
            workspace = st.text_input(
                "Workspace Name", 
                value="runxy", 
                help="Nama workspace di Roboflow."
            )
            project_name = st.text_input(
                "Project Name", 
                value="isolated-6chwu", 
                help="Nama project di Roboflow."
            )
            version = st.number_input(
                "Project Version", 
                min_value=1, 
                value=2, 
                help="Nomor versi dataset."
            )
            
            st.subheader("⚙️ Hyperparameter Training")
            epochs = st.number_input("Jumlah Epochs", min_value=1, max_value=200, value=10)
            batch_size = st.select_slider(
                "Batch Size", 
                options=[2, 4, 8, 16, 32], 
                value=16
            )
            
            # Learning rate hardcoded dengan nilai default
            st.info("💡 **Learning Rate:** Head = 1e-4, Backbone = 1e-5 (fixed)")
            
            submitted = st.form_submit_button("Mulai Training")
        
        if submitted:
            if not api_key:
                st.error("API Key Roboflow tidak boleh kosong.")
            else:
                try:
                    # Create config dengan learning rate default (hardcoded)
                    config = TrainingConfig(
                        api_key=api_key,
                        workspace=workspace,
                        project_name=project_name,
                        version=version,
                        epochs=epochs,
                        lr=1e-4,  # Default: 1e-4 (hardcoded)
                        lr_backbone=1e-5,  # Default: 1e-5 (hardcoded)
                        weight_decay=1e-4,
                        batch_size=batch_size,
                        num_workers=2  # Keep it low for Streamlit
                    )
                    
                    with st.spinner("Sedang mempersiapkan training... Proses ini mungkin memakan waktu beberapa menit."):
                        st.session_state.training_in_progress = True
                        st.session_state.training_log = ""
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Memulai training... Log akan muncul di konsol.")
                        
                        # Create training controller dan jalankan training
                        training_controller = TrainingController(config)
                        final_model_path, loss_plot_path = training_controller.train()
                        
                        st.session_state.training_in_progress = False
                        st.success("🎉 Training Selesai!")
                        st.balloons()
                        
                        st.subheader("Hasil Training")
                        st.markdown(f"**Model baru disimpan di:** `{final_model_path}`")
                        
                        # Download button untuk model
                        if os.path.exists(final_model_path):
                            with open(final_model_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Unduh Model Weights (.pth)",
                                    data=f,
                                    file_name=os.path.basename(final_model_path),
                                    mime="application/octet-stream"
                                )
                        
                        if os.path.exists(loss_plot_path):
                            st.markdown("**Grafik Loss Training:**")
                            st.image(loss_plot_path)
                
                except ValueError as e:
                    st.session_state.training_in_progress = False
                    st.error(f"❌ Konfigurasi tidak valid: {str(e)}")
                except Exception as e:
                    st.session_state.training_in_progress = False
                    st.error(f"❌ Terjadi error saat training: {str(e)}")
