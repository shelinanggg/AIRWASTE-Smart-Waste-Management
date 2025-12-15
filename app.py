import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="Smart Waste Classifier", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <svg width="50" height="50" viewBox="0 0 24 24" fill="white" style="vertical-align: middle; margin-right: 10px;">
        <path d="M9,3V4H4V6H5V19A2,2 0 0,0 7,21H17A2,2 0 0,0 19,19V6H20V4H15V3H9M7,6H17V19H7V6M9,8V17H11V8H9M13,8V17H15V8H13Z"/>
    </svg>
    <h1 style="display: inline-block; vertical-align: middle;">Smart Waste Classification System</h1>
    <p>AI-Powered Waste Sorting for a Sustainable Future</p>
</div>
""", unsafe_allow_html=True)

# Load model
# NOTE: Ensure this path matches your folder structure exactly
MODEL_PATH = "waste-classification-cnn-model-tensorflow1-default-v1/Waste-Classification-CNN-Model.h5"

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

if error:
    st.error(f"Error loading model: {error}")
    st.warning("Please check if the model file path is correct.")
    st.stop()
else:
    st.success(f"Model loaded successfully!")

# Layout Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="#667eea" style="margin-right: 8px;">
            <path d="M12 9C10.3 9 9 10.3 9 12C9 13.7 10.3 15 12 15C13.7 15 15 13.7 15 12C15 10.3 13.7 9 12 9M12 17C9.2 17 7 14.8 7 12C7 9.2 9.2 7 12 7C14.8 7 17 9.2 17 12C17 14.8 14.8 17 12 17M12 4.5C7 4.5 2.7 7.6 1 12C2.7 16.4 7 19.5 12 19.5C17 19.5 21.3 16.4 23 12C21.3 7.6 17 4.5 12 4.5Z"/>
        </svg>
        <h3 style="margin: 0;">Webcam Feed</h3>
    </div>
    """, unsafe_allow_html=True)
    camera_image = st.camera_input("Point camera at waste item", key="camera")

with col2:
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="#667eea" style="margin-right: 8px;">
            <path d="M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2M12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20A8,8 0 0,0 20,12A8,8 0 0,0 12,4M12,10.5A1.5,1.5 0 0,1 13.5,12A1.5,1.5 0 0,1 12,13.5A1.5,1.5 0 0,1 10.5,12A1.5,1.5 0 0,1 12,10.5M7.5,10.5A1.5,1.5 0 0,1 9,12A1.5,1.5 0 0,1 7.5,13.5A1.5,1.5 0 0,1 6,12A1.5,1.5 0 0,1 7.5,10.5M16.5,10.5A1.5,1.5 0 0,1 18,12A1.5,1.5 0 0,1 16.5,13.5A1.5,1.5 0 0,1 15,12A1.5,1.5 0 0,1 16.5,10.5Z"/>
        </svg>
        <h3 style="margin: 0;">Classification Result</h3>
    </div>
    """, unsafe_allow_html=True)
    result_placeholder = st.empty()

# Smart Bins Placeholder
st.markdown("""
<div style="display: flex; align-items: center; margin: 2rem 0 1rem 0;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="#667eea" style="margin-right: 8px;">
        <path d="M9,3V4H4V6H5V19A2,2 0 0,0 7,21H17A2,2 0 0,0 19,19V6H20V4H15V3H9M7,6H17V19H7V6M9,8V17H11V8H9M13,8V17H15V8H13Z"/>
    </svg>
    <h3 style="margin: 0;">Smart Bins</h3>
</div>
""", unsafe_allow_html=True)
bins_placeholder = st.empty()

# Function to display bins with Updated SVGs
def display_bins(predicted_class, confidence):
    organic_active = "active" if predicted_class == "Organic" else ""
    recyclable_active = "active" if predicted_class == "Recyclable" else ""
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            padding: 20px;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .container {{
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 60px;
            max-width: 800px;
            margin: 0 auto;
        }}
        
        .bin {{
            width: 200px;
            position: relative;
            transition: all 0.3s ease;
        }}
        
        .bin-wrapper {{
            position: relative;
            height: 300px;
        }}
        
        .bin-body {{
            width: 100%;
            height: 240px;
            background: linear-gradient(145deg, #2a5c2a, #1a3d1a);
            border-radius: 15px 15px 25px 25px;
            position: absolute;
            bottom: 0;
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            transition: all 0.4s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        
        .bin-body.recyclable {{
            background: linear-gradient(145deg, #2563eb, #1e40af);
        }}
        
        .bin-lid {{
            width: 100%;
            height: 70px;
            background: linear-gradient(145deg, #357535, #234d23);
            border-radius: 15px 15px 8px 8px;
            position: absolute;
            top: 0;
            transform-origin: top;
            transition: transform 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            z-index: 10;
        }}
        
        .bin-lid.recyclable {{
            background: linear-gradient(145deg, #3b82f6, #2563eb);
        }}
        
        .bin-lid::before {{
            content: '';
            position: absolute;
            width: 70px;
            height: 18px;
            background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
            top: 25px;
            left: 50%;
            transform: translateX(-50%);
            border-radius: 8px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
        }}
        
        .bin.active .bin-lid {{
            transform: rotateX(-130deg);
        }}
        
        .bin-label {{
            color: white;
            font-size: 22px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            margin-top: 15px;
        }}
        
        .bin-icon {{
            width: 80px;
            height: 80px;
            filter: drop-shadow(3px 3px 6px rgba(0,0,0,0.3));
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
        
        .bin.active {{
            animation: pulse 0.6s ease-in-out;
        }}
        
        .bin.active .bin-body {{
            box-shadow: 0 15px 50px rgba(34, 197, 94, 0.5);
        }}
        
        .bin.active .bin-body.recyclable {{
            box-shadow: 0 15px 50px rgba(37, 99, 235, 0.5);
        }}
        
        .bin-status {{
            position: absolute;
            top: -40px;
            width: 100%;
            text-align: center;
            font-size: 16px;
            font-weight: 700;
            color: #22c55e;
            opacity: 0;
            transition: opacity 0.3s ease;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .bin.active .bin-status {{
            opacity: 1;
        }}
        
        .confidence-badge {{
            position: absolute;
            bottom: -35px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .bin.active .confidence-badge {{
            opacity: 1;
        }}
    </style>
    </head>
    <body>
        <div class="container">
            <div class="bin {organic_active}">
                <div class="bin-status">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="#22c55e" style="vertical-align: middle; margin-right: 4px;">
                        <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z"/>
                    </svg>
                    BIN OPENED
                </div>
                <div class="bin-wrapper">
                    <div class="bin-lid"></div>
                    <div class="bin-body">
                        <svg class="bin-icon" version="1.1" xmlns="http://www.w3.org/2000/svg" width="45" height="47" viewBox="0 0 45 47">
                            <path d="M0 0 C0.51219785 12.13624352 0.69734757 22.60197737 -7.375 32.5 C-12.78392802 38.15163633 -18.29254889 39.82345471 -25.9375 40.1875 C-29.59055213 40.15284106 -32.59148587 39.36340565 -36 38 C-36.66 40.97 -37.32 43.94 -38 47 C-39.65 47 -41.3 47 -43 47 C-43.66416927 35.43853499 -40.90836798 27.25587167 -34 18 C-38.39291786 21.31909349 -40.8227865 23.79887886 -43 29 C-43.66 28.67 -44.32 28.34 -45 28 C-45.75567555 14.04099327 -45.75567555 14.04099327 -41 8 C-34.41916064 1.59234063 -27.00635857 1.65986319 -18.3125 1.4375 C-16.24597747 1.37758232 -14.17955836 1.3139496 -12.11328125 1.24609375 C-11.20360596 1.22216553 -10.29393066 1.1982373 -9.35668945 1.17358398 C-6.00131026 0.92644065 -3.46910202 0 0 0 Z " fill="#FFFFFF" transform="translate(45,0)"/>
                        </svg>
                        <div class="bin-label">Organic</div>
                    </div>
                </div>
                {"<div class='confidence-badge'>Confidence: " + f"{confidence:.1f}%" + "</div>" if organic_active else ""}
            </div>
            
            <div class="bin {recyclable_active}">
                <div class="bin-status">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="#22c55e" style="vertical-align: middle; margin-right: 4px;">
                        <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z"/>
                    </svg>
                    BIN OPENED
                </div>
                <div class="bin-wrapper">
                    <div class="bin-lid recyclable"></div>
                    <div class="bin-body recyclable">
                        <svg class="bin-icon" version="1.1" xmlns="http://www.w3.org/2000/svg" width="60" height="58" viewBox="0 0 60 58">
                            <path d="M0 0 C0.87471834 1.36201818 1.72397503 2.74040563 2.5625 4.125 C3.03816406 4.89070313 3.51382813 5.65640625 4.00390625 6.4453125 C5.31222282 9.8007597 4.96911995 11.54696449 4 15 C2.44140625 18.17578125 2.44140625 18.17578125 0.5625 21.3125 C-0.05496094 22.36050781 -0.67242187 23.40851562 -1.30859375 24.48828125 C-3 27 -3 27 -5 28 C-7.6723621 28.13415472 -10.32250488 28.04318541 -13 28 C-13 29.65 -13 31.3 -13 33 C-16.30979968 30.18667027 -18.1048141 26.858057 -20 23 C-19.61842486 19.75984496 -18.19622523 17.45573113 -16.4375 14.75 C-15.98246094 14.04359375 -15.52742188 13.3371875 -15.05859375 12.609375 C-14.53458984 11.81273437 -14.53458984 11.81273437 -14 11 C-13.67 12.32 -13.34 13.64 -13 15 C-11.35 15 -9.7 15 -8 15 C-8.66 13.906875 -9.32 12.81375 -10 11.6875 C-11.58464929 9.06292462 -12 8.1743622 -12 5 C-10.56705502 4.15885829 -9.12845595 3.32734313 -7.6875 2.5 C-6.88699219 2.0359375 -6.08648437 1.571875 -5.26171875 1.09375 C-3 0 -3 0 0 0 Z " fill="#FFFFFF" transform="translate(51,23)"/>
                            <path d="M0 0 C0.80759766 -0.03287109 1.61519531 -0.06574219 2.44726562 -0.09960938 C6.88873322 -0.1396226 8.64489187 0.01347967 12.26171875 2.83984375 C14.1875 5.4375 14.1875 5.4375 14.1875 7.4375 C15.8375 6.7775 17.4875 6.1175 19.1875 5.4375 C17.62108249 9.74514815 15.50413844 13.49921464 13.1875 17.4375 C8.5675 17.4375 3.9475 17.4375 -0.8125 17.4375 C0.1775 16.1175 1.1675 14.7975 2.1875 13.4375 C1.1975 12.4475 0.2075 11.4575 -0.8125 10.4375 C-1.080625 11.200625 -1.34875 11.96375 -1.625 12.75 C-2.8125 15.4375 -2.8125 15.4375 -5.8125 18.4375 C-9.1125 16.4575 -12.4125 14.4775 -15.8125 12.4375 C-13.42779746 2.89868985 -9.31065316 0.06592909 0 0 Z " fill="#FFFFFF" transform="translate(28.8125,6.5625)"/>
                            <path d="M0 0 C4.44892584 -0.53466319 8.57357974 -0.79287006 13 0 C15.92858724 2.574868 17.57268544 5.38935038 19 9 C19 9.66 19 10.32 19 11 C16.525 10.505 16.525 10.505 14 10 C13.67 11.32 13.34 12.64 13 14 C16.3 14 19.6 14 23 14 C23 18.29 23 22.58 23 27 C10.63862416 27.49445503 10.63862416 27.49445503 7.78515625 25.28515625 C0 15.31947484 0 15.31947484 0 9.125 C0.81960435 6.5637364 1.61892259 4.30179569 3 2 C2.01 1.67 1.02 1.34 0 1 C0 0.67 0 0.34 0 0 Z " fill="#FFFFFF" transform="translate(5,24)"/>
                        </svg>
                        <div class="bin-label">Recyclable</div>
                    </div>
                </div>
                {"<div class='confidence-badge'>Confidence: " + f"{confidence:.1f}%" + "</div>" if recyclable_active else ""}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_code

# Real-time Prediction Process
if camera_image is not None:
    # Read Image
    img = Image.open(camera_image)
    
    # Get model input shape
    input_h, input_w, input_c = model.input_shape[1], model.input_shape[2], model.input_shape[3]
    
    # Convert based on channels
    if input_c == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    
    # Resize
    img_resized = img.resize((input_w, input_h))
    
    # Array conversion
    img_array = np.array(img_resized) / 255.0
    
    # Add channel if grayscale
    if input_c == 1:
        img_array = np.expand_dims(img_array, axis=-1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    with st.spinner('Analyzing waste...'):
        prediction = model.predict(img_array, verbose=0)[0]
    
    class_names = ["Organic", "Recyclable"]
    predicted_idx = np.argmax(prediction)
    predicted_label = class_names[predicted_idx]
    confidence = float(prediction[predicted_idx] * 100)
    
    # Display Results
    with result_placeholder.container():
        # Icon SVG for result card (Simple style maintained for UI clarity)
        if predicted_label == "Organic":
            icon_svg = """
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22L6.66 19.7C7.14 19.87 7.64 20 8 20C19 20 22 3 22 3S11 2 8 2C4 2 1 5 1 5L6 10C6 10 6.1 8 17 8Z" fill="#22c55e"/>
            </svg>
            """
            color = "#22c55e"
        else:
            icon_svg = """
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M21.82 15.42L19.32 19.75C18.92 20.44 18.17 20.88 17.37 20.88H6.63C5.83 20.88 5.08 20.44 4.68 19.75L2.18 15.42C1.97 15.06 1.86 14.65 1.86 14.23C1.86 13.26 2.65 12.47 3.62 12.47H4.63L6.78 7.97C7.09 7.31 7.76 6.88 8.5 6.88H9.94L9.4 5.77C9.14 5.24 9.3 4.59 9.8 4.29L11.23 3.42C11.73 3.12 12.38 3.28 12.68 3.78L14.57 7.13L16.46 3.78C16.76 3.28 17.41 3.12 17.91 3.42L19.34 4.29C19.84 4.59 20 5.24 19.74 5.77L19.2 6.88H15.5C14.76 6.88 14.09 7.31 13.78 7.97L11.63 12.47H20.38C21.35 12.47 22.14 13.26 22.14 14.23C22.14 14.65 22.03 15.06 21.82 15.42M6 14.47L8 10.47H16L18 14.47L20 18.88H4L6 14.47Z" fill="#2563eb"/>
            </svg>
            """
            color = "#2563eb"
        
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <div style="margin-bottom: 1rem;">{icon_svg}</div>
            <h2 style="color: {color}; margin: 0.5rem 0; font-size: 2rem;">{predicted_label}</h2>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.8rem; border-radius: 10px; font-size: 1.5rem; font-weight: 600; margin-top: 1rem;">
                Confidence: {confidence:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Probabilities
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 1.5rem 0 0.5rem 0;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="#667eea" style="margin-right: 8px;">
                <path d="M3,3H5V13H9V7H13V11H17V5H21V21H3V3M7,21H11V15H7V21M15,21H19V13H15V21Z"/>
            </svg>
            <h4 style="margin: 0;">Detailed Probabilities</h4>
        </div>
        """, unsafe_allow_html=True)
        for name, prob in zip(class_names, prediction):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.progress(float(prob))
            with col_b:
                st.markdown(f"**{prob*100:.1f}%**")
            st.caption(name)
    
    # Display Smart Bins Animation
    with bins_placeholder:
        components.html(display_bins(predicted_label, confidence), height=400)
    
else:
    # Initial Instructions
    with result_placeholder:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
            <svg width="80" height="80" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 1rem;">
                <path d="M12 9C10.3 9 9 10.3 9 12C9 13.7 10.3 15 12 15C13.7 15 15 13.7 15 12C15 10.3 13.7 9 12 9M12 17C9.2 17 7 14.8 7 12C7 9.2 9.2 7 12 7C14.8 7 17 9.2 17 12C17 14.8 14.8 17 12 17M12 4.5C7 4.5 2.7 7.6 1 12C2.7 16.4 7 19.5 12 19.5C17 19.5 21.3 16.4 23 12C21.3 7.6 17 4.5 12 4.5Z" fill="white"/>
            </svg>
            <h3 style="margin: 0.5rem 0;">Ready to Classify!</h3>
            <p style="margin: 0.5rem 0; opacity: 0.9;">Point your camera at the waste item to start real-time classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with bins_placeholder:
        components.html(display_bins("", 0), height=400)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin-top: 2rem;">
    <svg width="40" height="40" viewBox="0 0 24 24" fill="#667eea" style="margin-bottom: 1rem;">
        <path d="M17.9,17.39C17.64,16.59 16.89,16 16,16H15V13A1,1 0 0,0 14,12H8V10H10A1,1 0 0,0 11,9V7H13A2,2 0 0,0 15,5V4.59C17.93,5.77 20,8.64 20,12C20,14.08 19.2,15.97 17.9,17.39M11,19.93C7.05,19.44 4,16.08 4,12C4,11.38 4.08,10.78 4.21,10.21L9,15V16A2,2 0 0,0 11,18M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z"/>
    </svg>
    <h3 style="color: #333; margin-bottom: 1rem;">Smart Waste Management System</h3>
    <p style="color: #666; font-size: 1.1rem; margin: 0.5rem 0;">Helping you sort waste correctly for a sustainable future</p>
    <p style="color: #888; font-size: 0.9rem; margin-top: 1rem;">The smart bins will automatically open based on detected waste type</p>
</div>
""", unsafe_allow_html=True)