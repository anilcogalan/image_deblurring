import streamlit as st
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import time
import io
from streamlit_image_comparison import image_comparison
import pandas as pd

# Distopik tema özelleştirmesi
st.markdown("""
<style>
    /* Ana tema renkleri ve arka plan */
    .main {
        background-color: #0b0b13;
        color: #e0e0e0;
    }
    
    /* Başlık stilini özelleştir */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        color: #36f9f6;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(54, 249, 246, 0.5);
        border-bottom: 1px solid #36f9f6;
        padding-bottom: 10px;
    }
    
    /* Widget stilini özelleştir */
    .stSlider, .stNumberInput {
        background-color: #161b22;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #364156;
    }
    
    /* Button stilini özelleştir */
    .stButton > button {
        background: linear-gradient(45deg, #36f9f6, #3b28cc);
        color: white;
        border: none;
        border-radius: 3px;
        padding: 10px 25px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 0 15px rgba(54, 249, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(54, 249, 246, 0.5);
    }
    
    /* Metrik stilleri */
    .stMetric {
        background: #161b22;
        border-left: 3px solid #36f9f6;
        padding: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar stili */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #161b22;
        border-right: 1px solid #364156;
    }
    
    .sidebar .sidebar-content {
        background-image: linear-gradient(180deg, #161b22 0%, #0b0b13 100%);
    }
    
    /* Sidebar başlığı */
    .sidebar .sidebar-content h2 {
        color: #36f9f6;
        text-align: center;
        letter-spacing: 1px;
    }
    
    /* Progress bar stili */
    .stProgress > div > div {
        background-color: #36f9f6;
    }
    
    /* Success mesajı */
    .element-container .stAlert {
        background-color: #162447;
        border: 1px solid #36f9f6;
        color: white;
    }
    
    /* Araç çıktılarını geliştir */
    .stImage {
        border: 1px solid #364156;
        border-radius: 5px;
        padding: 5px;
        background-color: #161b22;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Font import */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600&display=swap');
    
    body {
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* İndirme butonları */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #1f4068, #1a1a2e);
        color: #36f9f6;
        border: 1px solid #36f9f6;
    }
    
    /* Karşılaştırma aracı özelleştirme */
    .comparison-slider {
        border: 2px solid #36f9f6;
        box-shadow: 0 0 30px rgba(54, 249, 246, 0.2);
    }
    
    /* Logo/başlık alanı için özel konteyner */
    .header-container {
        background: linear-gradient(90deg, #0b0b13, #161b22);
        padding: 20px;
        margin-bottom: 30px;
        border-bottom: 1px solid #36f9f6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Function to load PyTorch and related modules
def load_pytorch_modules():
    import torch
    from blind_deconv import blind_deconv
    from ringing_artifacts_removal import ringing_artifacts_removal
    from misc import visualize_rgb, visualize_image, gray_image, process_image, PSNR
    return torch, blind_deconv, ringing_artifacts_removal, visualize_rgb, visualize_image, gray_image, process_image, PSNR

# Function to perform deblur operation
def perform_deblur(image_path, kernel_size, lambda_ftr, lambda_dark, lambda_grad, lambda_tv, lambda_l0, weight_ring):
    # Load PyTorch modules
    torch, blind_deconv, ringing_artifacts_removal, visualize_rgb, visualize_image, gray_image, process_image, PSNR = load_pytorch_modules()
    
    # Parameters
    opts = {
        'prescale': 1,
        'xk_iter': 5,
        'gamma_correct': 1.0,
        'k_thresh': 20,
        'kernel_size': kernel_size,
    }
    
    # Start processing
    inpt = Image.open(image_path)
    yg = gray_image(inpt)
    
    # Blind deconvolution
    kernel, interim_latent = blind_deconv(yg, lambda_ftr, lambda_dark, lambda_grad, opts)
    
    # Ringing artifacts removal
    y = process_image(inpt)
    y = y.permute(1, 2, 0)
    y = y / 255.0
    
    Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)
    
    # Process result image
    Latent[Latent > 1.0] = 1.0
    Latent[Latent < 0.0] = 0.0
    
    # Convert images to numpy arrays
    latent_np = Latent.squeeze().cpu().numpy()
    if latent_np.ndim == 3:
        latent_np = (latent_np * 255).astype(np.uint8)
    else:
        latent_np = (latent_np * 255).astype(np.uint8)
        latent_np = np.stack([latent_np] * 3, axis=2)
    
    # Convert kernel to numpy array
    kmn = kernel.min()
    kmx = kernel.max()
    kernel_np = ((kernel - kmn) / (kmx - kmn) * 255).numpy().astype(np.uint8)
    
    # Prepare interim results with matplotlib
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(kernel.numpy(), cmap='gray')
    plt.title('Estimated Kernel')
    plt.subplot(122)
    plt.imshow(interim_latent.numpy(), cmap='gray')
    plt.title('Interim Result Image')
    
    # Convert figure to byte array
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return latent_np, kernel_np, buf

def main():
    # Özel başlık
    st.markdown("""
    <div class="header-container">
        <h1>NEXUS DEBLUR</h1>
        <p style="color:#888;text-transform:uppercase;letter-spacing:3px;">Advanced Visual Restoration System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Image Deblurring Application")
    
    # Initialize session state (if not exists)
    if 'psnr_history' not in st.session_state:
        st.session_state.psnr_history = []
    if 'mse_history' not in st.session_state:
        st.session_state.mse_history = []
    if 'run_count' not in st.session_state:
        st.session_state.run_count = 0
    
    st.sidebar.header("Parameters")
    kernel_size = st.sidebar.slider("Kernel Size", 5, 101, 29, step=2)  # Always needs to be odd number
    
    # Lambda value adjustment options
    lambda_ftr = st.sidebar.number_input("Lambda Feature", value=3e-4, format="%.6f")
    lambda_dark = st.sidebar.number_input("Lambda Dark", value=0.0, format="%.6f")
    lambda_grad = st.sidebar.number_input("Lambda Grad", value=4e-3, format="%.6f")
    lambda_tv = st.sidebar.number_input("Lambda TV", value=0.001, format="%.6f")
    lambda_l0 = st.sidebar.number_input("Lambda L0", value=5e-4, format="%.6f")
    weight_ring = st.sidebar.number_input("Weight Ring", value=1.0)
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a blurry image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Blurry Image", use_column_width=True)
        
        # Create temporary file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(temp_path)
        
        # Start processing button
        if st.button("⚡ INITIATE DEBLUR SEQUENCE"):
            with st.spinner("Performing deblurring... This may take a while."):
                # Progress bar
                progress_bar = st.progress(0)
                
                # Process start
                start_time = time.time()
                
                try:
                    # Perform the process
                    progress_bar.progress(10)
                    
                    st.text("Estimating kernel and performing deblurring...")
                    latent_np, kernel_np, figure_bytes = perform_deblur(
                        temp_path, kernel_size, lambda_ftr, lambda_dark, 
                        lambda_grad, lambda_tv, lambda_l0, weight_ring
                    )
                    
                    progress_bar.progress(90)
                    
                    # Save results
                    results_dir = 'results'
                    os.makedirs(results_dir, exist_ok=True)
                    
                    result_path = os.path.join(results_dir, f'deblurred_{uploaded_file.name}')
                    kernel_path = os.path.join(results_dir, f'kernel_{uploaded_file.name}')
                    
                    latent_img = Image.fromarray(latent_np)
                    latent_img.save(result_path)
                    
                    kernel_img = Image.fromarray(kernel_np)
                    kernel_img.save(kernel_path)
                    
                    end_time = time.time()
                    progress_bar.progress(100)
                    
                    # Show interim results
                    st.image(figure_bytes, caption="Kernel and Interim Result", use_column_width=True)
                    
                    # Show results
                    st.success(f"Process completed in {end_time - start_time:.2f} seconds!")
                    
                    # Show result image
                    st.subheader("Deblurred Image")
                    st.image(latent_img, use_column_width=True)
                    
                    # Resize for image comparison tool
                    orig_width, orig_height = image.size
                    # Resize deblurred image to original size for comparison and metrics
                    latent_img_resized = latent_img.resize((orig_width, orig_height))
                    
                    st.subheader("Image Comparison")
                    st.write("Move the slider to compare results")
                    image_comparison(
                        img1=image,
                        img2=latent_img_resized, # Use the resized one
                        label1="Original Blurry Image",
                        label2="Deblurred Image",
                        width=700,
                        starting_position=50,
                    )
                    
                    # Show kernel
                    st.subheader("Estimated Kernel")
                    st.image(kernel_np, width=200)

                    # --- METRIC CALCULATION AND DISPLAY ---
                    st.subheader("Performance Metrics")

                    # Convert PIL images to numpy arrays for metrics
                    original_pil_for_metric = image # Original uploaded PIL
                    deblurred_pil_for_metric = latent_img_resized # Resized deblurred PIL

                    # Ensure channel consistency (convert to grayscale if modes differ)
                    if original_pil_for_metric.mode != deblurred_pil_for_metric.mode:
                        st.warning(
                            f"Original image mode ({original_pil_for_metric.mode}) and "
                            f"deblurred image mode ({deblurred_pil_for_metric.mode}) differ. "
                            "Both are being converted to grayscale for metrics."
                        )
                        if original_pil_for_metric.mode != 'L':
                            original_pil_for_metric = original_pil_for_metric.convert('L')
                        if deblurred_pil_for_metric.mode != 'L':
                            deblurred_pil_for_metric = deblurred_pil_for_metric.convert('L')

                    original_np_for_metric = np.array(original_pil_for_metric).astype(np.float32)
                    deblurred_np_for_metric = np.array(deblurred_pil_for_metric).astype(np.float32)

                    # Calculate MSE
                    if original_np_for_metric.shape == deblurred_np_for_metric.shape:
                        mse_value = np.mean((original_np_for_metric - deblurred_np_for_metric) ** 2)
                        st.metric(label="Mean Square Error (MSE)", value=f"{mse_value:.2f}")

                        # Calculate PSNR (uses PSNR function from misc.py)
                        # misc.PSNR function expects normalized torch tensors in [0,1] range
                        torch_module, _, _, _, _, _, _, PSNR_func = load_pytorch_modules()
                        
                        original_tensor_norm = torch_module.from_numpy(original_np_for_metric / 255.0)
                        deblurred_tensor_norm = torch_module.from_numpy(deblurred_np_for_metric / 255.0)
                        
                        # Match dimensions (if one is (H,W) and the other is (H,W,C))
                        if original_tensor_norm.ndim == 2 and deblurred_tensor_norm.ndim == 3:
                            # If deblurred is RGB and original is grayscale, convert deblurred to grayscale
                            if deblurred_tensor_norm.shape[2] == 3: # Simple grayscale conversion
                                deblurred_tensor_norm = deblurred_tensor_norm[:,:,0]*0.2989 + deblurred_tensor_norm[:,:,1]*0.587 + deblurred_tensor_norm[:,:,2]*0.114
                        elif original_tensor_norm.ndim == 3 and deblurred_tensor_norm.ndim == 2:
                            # If original is RGB and deblurred is grayscale, convert original to grayscale
                            if original_tensor_norm.shape[2] == 3:
                                original_tensor_norm = original_tensor_norm[:,:,0]*0.2989 + original_tensor_norm[:,:,1]*0.587 + original_tensor_norm[:,:,2]*0.114

                        if original_tensor_norm.shape == deblurred_tensor_norm.shape:
                            psnr_value = PSNR_func(deblurred_tensor_norm, original_tensor_norm)
                            psnr_value_item = psnr_value.item() if torch_module.is_tensor(psnr_value) else psnr_value
                            st.metric(label="PSNR (Peak Signal-to-Noise Ratio)", value=f"{psnr_value_item:.2f} dB")

                            # Add to history
                            st.session_state.run_count += 1
                            st.session_state.psnr_history.append({'run': st.session_state.run_count, 'PSNR': psnr_value_item})
                            st.session_state.mse_history.append({'run': st.session_state.run_count, 'MSE': mse_value})
                        else:
                            st.warning(f"PSNR calculation failed: Dimensions don't match. Original shape: {original_tensor_norm.shape}, Deblurred shape: {deblurred_tensor_norm.shape}")
                            psnr_value_item = None # For graph
                    else:
                        st.warning("MSE and PSNR calculation failed: Image dimensions don't match.")
                        mse_value = None
                        psnr_value_item = None

                    # Draw Graphs
                    if st.session_state.psnr_history:
                        st.subheader("PSNR History")
                        psnr_df = pd.DataFrame(st.session_state.psnr_history)
                        st.line_chart(psnr_df.set_index('run')['PSNR'])

                    if st.session_state.mse_history:
                        st.subheader("MSE History")
                        mse_df = pd.DataFrame(st.session_state.mse_history)
                        st.line_chart(mse_df.set_index('run')['MSE'])
                    
                    # Download links
                    st.subheader("Download Results")
                    with open(result_path, "rb") as file:
                        st.download_button(
                            label="Download Deblurred Image",
                            data=file,
                            file_name=f'deblurred_{uploaded_file.name}',
                            mime="image/png"
                        )
                    
                    with open(kernel_path, "rb") as file:
                        st.download_button(
                            label="Download Kernel",
                            data=file,
                            file_name=f'kernel_{uploaded_file.name}',
                            mime="image/png"
                        )
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.error("Error details:")
                    st.exception(e)

if __name__ == "__main__":
    main() 