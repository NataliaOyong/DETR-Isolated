import streamlit as st
from training import train_model
import os

def render_training_page():
    st.header("🏋️ Model Training")

    st.info("""
    **Halaman ini digunakan untuk melakukan training (fine-tuning) model DETR.**
    - Masukkan informasi dataset dari Roboflow.
    - Atur hyperparameter training.
    - Klik 'Mulai Training' untuk memulai proses.
    """)

    with st.form("training_form"):
        st.subheader("📦 Roboflow Dataset Configuration")
        # TODO: Handle API key securely
        api_key = st.text_input("Roboflow API Key", type="password", help="Gunakan API key Roboflow Anda.")
        workspace = st.text_input("Workspace Name", value="runxy", help="Nama workspace di Roboflow.")
        project_name = st.text_input("Project Name", value="isolated-6chwu", help="Nama project di Roboflow.")
        version = st.number_input("Project Version", min_value=1, value=2, help="Nomor versi dataset.")

        st.subheader("⚙️ Hyperparameter Training")
        epochs = st.number_input("Jumlah Epochs", min_value=1, max_value=200, value=10)
        batch_size = st.select_slider("Batch Size", options=[2, 4, 8, 16, 32], value=16)
        lr = st.select_slider("Learning Rate (Head)", options=[1e-3, 1e-4, 1e-5, 1e-6], value=1e-4, format_func=lambda x: f"{x:.0e}")
        lr_backbone = st.select_slider("Learning Rate (Backbone)", options=[1e-4, 1e-5, 1e-6, 1e-7], value=1e-5, format_func=lambda x: f"{x:.0e}")
        
        submitted = st.form_submit_button("Mulai Training")

    if submitted:
        if not api_key:
            st.error("API Key Roboflow tidak boleh kosong.")
        else:
            with st.spinner("Sedang mempersiapkan training... Proses ini mungkin memakan waktu beberapa menit."):
                try:
                    st.session_state.training_in_progress = True
                    st.session_state.training_log = ""
                    
                    # Placeholder for progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # We can't directly capture stdout from the training function easily in streamlit.
                    # A better approach would be to use a custom callback to write progress to a file 
                    # or use a queue, but for now, we will just show a spinner.
                    
                    status_text.text("Memulai training... Log akan muncul di konsol.")

                    final_model_path, loss_plot_path = train_model(
                        api_key=api_key,
                        workspace=workspace,
                        project_name=project_name,
                        version=version,
                        epochs=epochs,
                        lr=lr,
                        lr_backbone=lr_backbone,
                        weight_decay=1e-4,
                        batch_size=batch_size,
                        num_workers=2 # Keep it low to avoid issues in Streamlit environment
                    )

                    st.session_state.training_in_progress = False
                    st.success(f"🎉 Training Selesai!")
                    st.balloons()
                    
                    st.subheader("Hasil Training")
                    st.markdown(f"**Model baru disimpan di:** `{final_model_path}`")

                    # Add download button for the model
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
                    
                except Exception as e:
                    st.session_state.training_in_progress = False
                    st.error(f"❌ Terjadi error saat training: {str(e)}")