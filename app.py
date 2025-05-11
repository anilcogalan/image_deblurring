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

# Apply dystopian theme customization
st.markdown("""
<style>
    /* Main theme colors and background */
    .main {
        background-color: #0b0b13;
        color: #e0e0e0;
    }
    /* Customize heading styles */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        color: #36f9f6;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(54, 249, 246, 0.5);
        border-bottom: 1px solid #36f9f6;
        padding-bottom: 10px;
    }
    /* Customize slider and number input widgets */
    .stSlider, .stNumberInput {
        background-color: #161b22;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #364156;
    }
    /* Customize button style */
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
    /* Style for metric components */
    .stMetric {
        background: #161b22;
        border-left: 3px solid #36f9f6;
        padding: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #161b22;
        border-right: 1px solid #364156;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(180deg, #161b22 0%, #0b0b13 100%);
    }
    /* Sidebar heading */
    .sidebar .sidebar-content h2 {
        color: #36f9f6;
        text-align: center;
        letter-spacing: 1px;
    }
    /* Customize progress bar */
    .stProgress > div > div {
        background-color: #36f9f6;
    }
    /* Success alert styling */
    .element-container .stAlert {
        background-color: #162447;
        border: 1px solid #36f9f6;
        color: white;
    }
    /* Improve tool output styling */
    .stImage {
        border: 1px solid #364156;
        border-radius: 5px;
        padding: 5px;
        background-color: #161b22;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
    }
    /* Import custom font */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600&display=swap');
    body {
        font-family: 'Rajdhani', sans-serif;
    }
    /* Download button customization */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #1f4068, #1a1a2e);
        color: #36f9f6;
        border: 1px solid #36f9f6;
    }
    /* Comparison slider customization */
    .comparison-slider {
        border: 2px solid #36f9f6;
        box-shadow: 0 0 30px rgba(54, 249, 246, 0.2);
    }
    /* Custom header container */
    .header-container {
        background: linear-gradient(90deg, #0b0b13, #161b22);
        padding: 20px;
        margin-bottom: 30px;
        border-bottom: 1px solid #36f9f6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Function to ensure image format is compatible with saving
def ensure_compatible_format(img, path):
    # Get file extension
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    
    # Convert RGBA to RGB if saving as JPEG
    if ext in ['.jpg', '.jpeg'] and img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # If no extension or unsupported, default to PNG
    if not ext or ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        path = f"{path}.png"
    
    return img, path

# Function to load PyTorch and related modules
def load_pytorch_modules():
    import torch
    from blind_deconv import blind_deconv
    from ringing_artifacts_removal import ringing_artifacts_removal
    from misc import visualize_rgb, visualize_image, gray_image, process_image, PSNR
    return torch, blind_deconv, ringing_artifacts_removal, visualize_rgb, visualize_image, gray_image, process_image, PSNR

# Function to perform deblurring operation
def perform_deblur(image_path, kernel_size, lambda_ftr, lambda_dark, lambda_grad, lambda_tv, lambda_l0, weight_ring):
    torch, blind_deconv, ringing_artifacts_removal, visualize_rgb, visualize_image, gray_image, process_image, PSNR = load_pytorch_modules()
    opts = {
        'prescale': 1,
        'xk_iter': 5,
        'gamma_correct': 1.0,
        'k_thresh': 20,
        'kernel_size': kernel_size,
    }
    # Open input image
    inpt = Image.open(image_path)
    # Convert to grayscale
    yg = gray_image(inpt)
    # Blind deconvolution step
    kernel, interim_latent = blind_deconv(yg, lambda_ftr, lambda_dark, lambda_grad, opts)
    # Prepare for ringing artifact removal
    y = process_image(inpt).permute(1,2,0) / 255.0
    Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)
    # Clamp output between 0 and 1
    Latent = torch.clamp(Latent, 0.0, 1.0)
    # Convert to uint8 numpy
    latent_np = (Latent.squeeze().cpu().numpy() * 255).astype(np.uint8)
    if latent_np.ndim == 2:
        latent_np = np.stack([latent_np]*3, axis=2)
    # Normalize kernel for display
    kmn, kmx = kernel.min(), kernel.max()
    kernel_np = (((kernel - kmn)/(kmx - kmn))*255).numpy().astype(np.uint8)
    # Create interim results figure
    fig = plt.figure(figsize=(10,5))
    plt.subplot(121); plt.imshow(kernel.numpy(), cmap='gray'); plt.title('Estimated Kernel')
    plt.subplot(122); plt.imshow(interim_latent.numpy(), cmap='gray'); plt.title('Interim Result')
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    return latent_np, kernel_np, buf

# Function to display results and metrics
def display_results(image, latent_img, kernel_img, interim_fig=None, processing_time=None):
    # Display interim figure if available
    if interim_fig is not None:
        st.image(interim_fig, caption="Kernel & Interim Result", use_column_width=True)
    
    # Show processing time if available
    if processing_time is not None:
        st.success(f"Completed in {processing_time:.2f} seconds")

    # Show deblurred image
    st.subheader("Deblurred Image")
    st.image(latent_img, use_column_width=True)

    # Image comparison slider
    st.subheader("Image Comparison")
    latent_resized = latent_img.resize(image.size)
    image_comparison(
        img1=image, img2=latent_resized,
        label1="Original Blurry", label2="Deblurred",
        width=700, starting_position=50
    )

    # Display kernel
    st.subheader("Estimated Kernel")
    st.image(kernel_img, width=200)

    # Display performance metrics
    st.subheader("Performance Metrics")
    orig = image
    deb = latent_resized
    if orig.mode != deb.mode:
        orig = orig.convert('L')
        deb = deb.convert('L')
    orig_np = np.array(orig).astype(np.float32)
    deb_np  = np.array(deb).astype(np.float32)
    if orig_np.shape == deb_np.shape:
        mse = np.mean((orig_np - deb_np)**2)
        st.metric("MSE", f"{mse:.2f}")
        torch_mod, *_ , PSNR_func = load_pytorch_modules()
        t_orig = torch_mod.from_numpy(orig_np/255.0)
        t_deb  = torch_mod.from_numpy(deb_np/255.0)
        if t_orig.ndim == 2 and t_deb.ndim == 3:
            t_deb = t_deb[:,:,0]*0.2989 + t_deb[:,:,1]*0.587 + t_deb[:,:,2]*0.114
        elif t_orig.ndim == 3 and t_deb.ndim == 2:
            t_orig = t_orig[:,:,0]*0.2989 + t_orig[:,:,1]*0.587 + t_orig[:,:,2]*0.114
        if t_orig.shape == t_deb.shape:
            psnr = PSNR_func(t_deb, t_orig)
            st.metric("PSNR (dB)", f"{psnr:.2f}")
            st.session_state.run_count += 1
            st.session_state.psnr_history.append({'run': st.session_state.run_count, 'PSNR': psnr})
            st.session_state.mse_history.append({'run': st.session_state.run_count, 'MSE': mse})
        else:
            st.warning("PSNR calculation failed: dimension mismatch.")
    else:
        st.warning("MSE/PSNR calculation failed: dimension mismatch.")
    
    # Plot history charts
    if st.session_state.psnr_history:
        st.subheader("PSNR History")
        psnr_df = pd.DataFrame(st.session_state.psnr_history)
        st.line_chart(psnr_df.set_index('run')['PSNR'])
    if st.session_state.mse_history:
        st.subheader("MSE History")
        mse_df = pd.DataFrame(st.session_state.mse_history)
        st.line_chart(mse_df.set_index('run')['MSE'])
    
    return latent_resized

# Main application entry point
def main():
    # Display custom header
    st.markdown("""
    <div class="header-container">
    <p style="color:#888;text-transform:uppercase;letter-spacing:3px;">Advanced Visual Restoration System</p>
    </div>
    """, unsafe_allow_html=True)
    st.title("Image Deblurring Application")

    # Initialize session state
    if 'psnr_history' not in st.session_state:
        st.session_state.psnr_history = []
        st.session_state.mse_history = []
        st.session_state.run_count = 0

    # Sidebar: sample image selection
    st.sidebar.subheader("🖼️ Sample Image Selection")
    test_images = {
        "Car🚘": "test_images/car.png",
        "City🌇": "test_images/city.png",
        "Face☺️": "test_images/face.png",
        "House🏠": "test_images/house.jpg",
        "Street🌆":"test_images/street.png",
        "Book📘":"test_images/book.jpg",
        "Flag🚩":"test_images/flag.png",
        "Flower🌷":"test_images/flower.jpg",


    }

    # Create map for pre-processed results
    deblurred_images = {
        "Car🚘": "deblurred_images/car.png",
        "City🌇": "deblurred_images/city.png",
        "Face☺️": "deblurred_images/face.png",
        "House🏠": "deblurred_images/house.jpg",
        "Street🌆":"deblurred_images/street.png",
        "Book📘":"deblurred_images/book.jpg",
        "Flag🚩":"deblurred_images/flag.png",
        "Flower🌷":"deblurred_images/flower.jpg",
    }
    
    kernel_images = {
        "Car🚘": "deblurred_images/ker_car.png",
        "City🌇": "deblurred_images/ker_city.png",
        "Face☺️": "deblurred_images/ker_face.png",
        "House🏠": "deblurred_images/ker_house.jpg",
        "Street🌆":"deblurred_images/ker_street.png",
        "Book📘":"deblurred_images/ker_book.jpg",
        "Flag🚩":"deblurred_images/ker_flag.png",   
        "Flower🌷":"deblurred_images/ker_flower.jpg",
        }
    
    interim_images = {
        "Car🚘": "deblurred_images/interim_car.png",
        "City🌇": "deblurred_images/interim_city.png",
        "Face☺️": "deblurred_images/interim_face.png",
        "House🏠": "deblurred_images/interim_house.jpg",
        "Street🌆":"deblurred_images/interim_street.png",
        "Book📘":"deblurred_images/interim_book.jpg",
        "Flag🚩":"deblurred_images/interim_flag.png",
        "Flower🌷":"deblurred_images/interim_flower.png",

                }

    selected = st.sidebar.selectbox("Choose a sample image", ["None"] + list(test_images.keys()))

    # Sidebar parameters
    st.sidebar.header("Parameters")
    kernel_size = st.sidebar.slider("Kernel Size", 5, 101, 29, step=2)
    lambda_ftr  = st.sidebar.number_input("Lambda Feature", value=3e-4, format="%.6f")
    lambda_dark = st.sidebar.number_input("Lambda Dark",    value=0.0,  format="%.6f")
    lambda_grad = st.sidebar.number_input("Lambda Grad",    value=4e-3, format="%.6f")
    lambda_tv   = st.sidebar.number_input("Lambda TV",      value=0.001,format="%.6f")
    lambda_l0   = st.sidebar.number_input("Lambda L0",      value=5e-4,format="%.6f")
    weight_ring = st.sidebar.number_input("Weight Ring",    value=1.0)

    # Create necessary directories
    os.makedirs("deblurred_images", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load uploaded file or sample
    if selected != "None":
        image_path = test_images[selected]
        image = Image.open(image_path)
        is_sample = True
        
        # Get file basename for saving
        base_name = os.path.basename(image_path)
        sample_key = selected
    else:
        uploaded_file = st.file_uploader("Upload a blurry image", type=["jpg","jpeg","png"])
        if not uploaded_file:
            return
            
        image = Image.open(uploaded_file)
        is_sample = False
        
        # Save to temp folder for custom uploads
        temp_path = os.path.join("temp", uploaded_file.name)
        image.save(temp_path)
        image_path = temp_path
        base_name = uploaded_file.name

    # Display the input image
    st.image(image, caption="Input (Blurry) Image", use_column_width=True)

    # Check if pre-computed results exist for sample image
    pre_computed_exists = False
    if is_sample and os.path.exists(deblurred_images[selected]) and os.path.exists(kernel_images[selected]):
        pre_computed_exists = True
        latent_img = Image.open(deblurred_images[selected])
        kernel_img = Image.open(kernel_images[selected])
        
        interim_fig = None
        if os.path.exists(interim_images[selected]):
            interim_fig = interim_images[selected]
        
        st.success(f"Using pre-computed results")
        
        # Display pre-computed results
        latent_resized = display_results(image, latent_img, kernel_img, interim_fig)
        
        # Download buttons
        st.subheader("Download Results")
        with open(deblurred_images[selected], "rb") as f:
            st.download_button("Download Deblurred Image", f, file_name=os.path.basename(deblurred_images[selected]))
        with open(kernel_images[selected], "rb") as f:
            st.download_button("Download Kernel", f, file_name=os.path.basename(kernel_images[selected]))
    
    # For samples without pre-computed results or custom uploads, show deblur button
    if not pre_computed_exists:
        # For sample images, automatically run deblur process
        if is_sample:
            run_deblur = True
            st.info("Pre-computed results not found. Processing image...")
        else:
            # For custom uploads, require button press
            run_deblur = st.button("⚡ INITIATE DEBLUR SEQUENCE")
        
        if run_deblur:
            with st.spinner("Deblurring in progress, please wait..."):
                progress = st.progress(0)
                start = time.time()
                try:
                    progress.progress(10)
                    latent_np, kernel_np, interim_fig_buf = perform_deblur(
                        image_path, kernel_size,
                        lambda_ftr, lambda_dark,
                        lambda_grad, lambda_tv,
                        lambda_l0, weight_ring
                    )
                    progress.progress(90)
                    
                    # Save results
                    if is_sample:
                        # Save to deblurred_images for future use
                        latent_path = deblurred_images[selected]
                        kernel_path = kernel_images[selected]
                        interim_path = interim_images[selected]
                    else:
                        # Save to results folder for custom uploads
                        latent_path = os.path.join("results", f"debl_{base_name}")
                        kernel_path = os.path.join("results", f"ker_{base_name}")
                        interim_path = os.path.join("results", f"interim_{base_name}")
                    
                    # Save latent image with format compatibility check
                    latent_img = Image.fromarray(latent_np)
                    latent_img, latent_path = ensure_compatible_format(latent_img, latent_path)
                    latent_img.save(latent_path)
                    
                    # Save kernel image with format compatibility check
                    kernel_img = Image.fromarray(kernel_np)
                    kernel_img, kernel_path = ensure_compatible_format(kernel_img, kernel_path)
                    kernel_img.save(kernel_path)
                    
                    # Save interim figure with format compatibility check
                    interim_img = Image.open(interim_fig_buf)
                    interim_img, interim_path = ensure_compatible_format(interim_img, interim_path)
                    interim_img.save(interim_path)
                    
                    progress.progress(100)
                    end = time.time()
                    processing_time = end - start

                    # Display results
                    latent_resized = display_results(
                        image, latent_img, kernel_img, 
                        interim_fig_buf, processing_time
                    )

                    # Download buttons
                    st.subheader("Download Results")
                    with open(latent_path, "rb") as f:
                        st.download_button("Download Deblurred Image", f, file_name=os.path.basename(latent_path))
                    with open(kernel_path, "rb") as f:
                        st.download_button("Download Kernel", f, file_name=os.path.basename(kernel_path))

                except Exception as e:
                    st.error(f"Error occurred: {e}")
                    st.exception(e)

if __name__ == "__main__":
    main()
