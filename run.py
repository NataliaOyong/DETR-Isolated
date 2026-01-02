"""
Script untuk menjalankan aplikasi DETR Isolated Mathematical Expression Detector
"""
import subprocess
import sys
import os
import platform

def check_dependencies():
    """Check if required packages are installed"""
    # Map package names to their import names
    package_map = {
        'streamlit': 'streamlit',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'transformers': 'transformers',
        'timm': 'timm',
        'Pillow': 'PIL',  # Pillow is imported as PIL
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_map.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Package yang belum terinstall:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install dengan perintah:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_model_weights():
    """Check if model weights file exists"""
    weights_file = "detr_isolated_model_100_weights.pth"
    if not os.path.exists(weights_file):
        print(f"⚠️  Warning: File {weights_file} tidak ditemukan!")
        print("   Pastikan file weights berada di direktori yang sama dengan aplikasi.")
        return False
    return True

def main():
    """Main function to run the Streamlit app"""
    print("=" * 60)
    print("🔢 DETR Isolated Mathematical Expression Detector")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("📦 Memeriksa dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Semua dependencies terinstall")
    print()
    
    # Check model weights
    print("🔍 Memeriksa model weights...")
    if not check_model_weights():
        print("⚠️  Aplikasi tetap akan dijalankan, namun mungkin akan error saat loading model.")
    else:
        print("✅ Model weights ditemukan")
    print()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, "app.py")
    
    if not os.path.exists(app_file):
        print(f"❌ Error: File {app_file} tidak ditemukan!")
        sys.exit(1)
    
    print("🚀 Menjalankan aplikasi Streamlit...")
    print("   Aplikasi akan terbuka di browser secara otomatis")
    print("   Tekan Ctrl+C untuk menghentikan aplikasi")
    print()
    print("=" * 60)
    print()
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Aplikasi dihentikan oleh user")
    except Exception as e:
        print(f"\n❌ Error menjalankan aplikasi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

