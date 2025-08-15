import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import re
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import google.generativeai as genai
from ultralytics import YOLO
import math
import io
import base64
import requests
from pathlib import Path
import time

# Set page config - optimized for web deployment
st.set_page_config(
    page_title="AI-Powered Object Detection & Scene Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/object-detection-app',
        'Report a bug': "https://github.com/yourusername/object-detection-app/issues",
        'About': "# AI Object Detection App\nPowered by YOLO, CLIP, and Gemini AI"
    }
)

# Custom CSS for better web appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'model_load_time' not in st.session_state:
        st.session_state.model_load_time = None

init_session_state()

@st.cache_resource(show_spinner=False)
def download_yolo_model():
    """Download YOLO model if not present"""
    model_path = "yolov8x-oiv7.pt"  # Using YOLOv8x model as requested
    try:
        # Try to use the model, download if needed
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_clip_models():
    """Load and cache CLIP models with better error handling"""
    try:
        with st.spinner("Loading CLIP models..."):
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return clip_model, text_processor
    except Exception as e:
        st.error(f"Error loading CLIP models: {e}")
        return None, None

def load_models_with_progress():
    """Load models with detailed progress tracking"""
    if st.session_state.models_loaded:
        return True
    
    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load YOLO model
        status_text.text("🤖 Loading YOLO object detection model...")
        progress_bar.progress(25)
        yolo_model = download_yolo_model()
        if yolo_model is None:
            return False
        
        # Load CLIP models
        status_text.text("🧠 Loading CLIP vision-language model...")
        progress_bar.progress(50)
        clip_model, text_processor = load_clip_models()
        if clip_model is None or text_processor is None:
            return False
        
        progress_bar.progress(75)
        status_text.text("✅ Finalizing model setup...")
        
        # Store in session state
        st.session_state.yolo_model = yolo_model
        st.session_state.clip_model = clip_model
        st.session_state.text_processor = text_processor
        st.session_state.models_loaded = True
        st.session_state.model_load_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text("🎉 All models loaded successfully!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return True
        
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        progress_bar.empty()
        status_text.empty()
        return False

def setup_gemini_api(api_key):
    """Setup Gemini API with validation"""
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")  # Updated to latest model
        
        # Test the API key with a simple request
        test_response = model.generate_content("Hello")
        if test_response:
            return model
        else:
            st.error("Failed to validate Gemini API key")
            return None
    except Exception as e:
        st.error(f"Error setting up Gemini API: {e}")
        st.error("Please check your API key and try again")
        return None

def validate_image(image):
    """Validate uploaded image"""
    if image is None:
        return False, "No image provided"
    
    try:
        # Check image size
        if hasattr(image, 'size'):
            width, height = image.size
            if width * height > 10000000:  # 10MP limit
                return False, "Image too large. Please upload an image smaller than 10MP"
            if width < 100 or height < 100:
                return False, "Image too small. Please upload an image at least 100x100 pixels"
        
        return True, "Valid image"
    except Exception as e:
        return False, f"Error validating image: {e}"

# Core processing functions (simplified versions of your original functions)
def get_yolo_detections(yolo_model, image):
    """YOLO detection with error handling"""
    try:
        results = yolo_model.predict(source=image, verbose=False, conf=0.3)  # Added confidence threshold
        df = results[0].to_df()
        
        if df.empty:
            return df
        
        # Handle column naming
        cols = list(df.columns)
        name_count = 0
        new_cols = []
        for col in cols:
            if col == 'name':
                name_count += 1
                new_cols.append('class_id' if name_count > 1 else 'name')
            else:
                new_cols.append(col)
        df.columns = new_cols
        
        # Extract coordinates
        try:
            coords = df['box'].apply(lambda b: pd.Series([b['x1'], b['y1'], b['x2'], b['y2']]))
            coords.columns = ['xmin', 'ymin', 'xmax', 'ymax']
            df = pd.concat([df.drop('box', axis=1), coords], axis=1)
        except Exception:
            return df.drop('box', axis=1, errors='ignore')
        
        return df
    except Exception as e:
        st.error(f"Error in YOLO detection: {e}")
        return pd.DataFrame()

def plot_detection_results(image, df, title="Detections", color=(0, 128, 255)):
    """Plot detection results with improved visualization"""
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_array)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        for _, row in df.iterrows():
            try:
                x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                width = x_max - x_min
                height = y_max - y_min
                
                # Draw rectangle
                rect = patches.Rectangle((x_min, y_min), width, height,
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                label = f"{row['name']} ({row['confidence']:.2f})"
                ax.text(x_min, y_min - 10, label, fontsize=10, color='red',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            except Exception as e:
                continue
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting results: {e}")
        return None

def process_query_with_gemini(query, gemini_model):
    """Process query with Gemini AI"""
    prompt = f"""
    Analyze this image search query and extract key information:
    Query: "{query}"
    
    Return a JSON object with:
    1. refined_prompt: A more detailed, descriptive version of the query
    2. objects: List of main objects/nouns mentioned (normalize similar terms)
    3. actions: List of main actions/verbs mentioned
    4. primary_object: The most important object that should be the focus
    
    Format as valid JSON only:
    {{
        "refined_prompt": "detailed description",
        "objects": ["object1", "object2"],
        "actions": ["action1", "action2"],
        "primary_object": "main_object"
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {
                "refined_prompt": query,
                "objects": [query.split()[0]] if query.split() else ["object"],
                "actions": ["interact"],
                "primary_object": "person"
            }
    except Exception as e:
        st.warning(f"Gemini processing failed: {e}")
        return {
            "refined_prompt": query,
            "objects": [query.split()[0]] if query.split() else ["object"],
            "actions": ["interact"],
            "primary_object": "person"
        }

def calculate_clip_similarity_simple(query, image_crops, clip_model, text_processor):
    """Simplified CLIP similarity calculation"""
    try:
        if not image_crops:
            return []
        
        with torch.no_grad():
            # Process text
            text_inputs = text_processor(text=query, return_tensors="pt", padding=True, truncation=True)
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            # Process images
            image_inputs = text_processor(images=image_crops, return_tensors="pt", padding=True)
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            # Calculate similarity
            similarity_scores = (text_features @ image_features.T).squeeze(0)
            
            if similarity_scores.dim() == 0:
                similarity_scores = similarity_scores.unsqueeze(0)
            
            return similarity_scores.cpu().numpy().tolist()
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return [0.0] * len(image_crops)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 AI Object Detection & Scene Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image and describe what you\'re looking for using natural language</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Gemini API Key", 
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if not api_key:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ API Key Required</strong><br>
                Please enter your Gemini API key to use this application.
                <br><br>
                <a href="https://makersuite.google.com/app/apikey" target="_blank">Get API Key →</a>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Model loading status
        st.subheader("🤖 Model Status")
        if st.session_state.models_loaded:
            st.success("✅ All models loaded")
            if st.session_state.model_load_time:
                st.info(f"⏱️ Load time: {st.session_state.model_load_time:.1f}s")
        else:
            st.warning("⏳ Models not loaded")
            if st.button("🚀 Load Models"):
                load_models_with_progress()
                st.rerun()
        
        # Settings
        st.subheader("🎛️ Settings")
        confidence_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.1)
        max_detections = st.slider("Max Detections", 5, 50, 20, 5)
    
    # Main interface
    if not st.session_state.models_loaded:
        st.info("👆 Please load the AI models using the sidebar before proceeding.")
        return
    
    # Setup Gemini
    gemini_model = setup_gemini_api(api_key)
    if not gemini_model:
        st.error("❌ Failed to setup Gemini AI. Please check your API key.")
        return
    
    # File upload and query input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG, WebP (max 10MB)"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            is_valid, message = validate_image(image)
            
            if is_valid:
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            else:
                st.error(f"❌ {message}")
                return
    
    with col2:
        st.subheader("💬 Describe What You're Looking For")
        query = st.text_area(
            "Enter your search query",
            placeholder="e.g., 'People talking behind a cart', 'Dog playing with a ball', 'Person holding a phone'",
            height=100,
            help="Describe the scene or objects you want to find in natural language"
        )
        
        # Example queries
        st.write("**💡 Example queries:**")
        example_queries = [
            "People talking behind a cart",
            "Person holding a phone",
            "Dog playing with a ball",
            "Car parked near a building"
        ]
        
        for i, example in enumerate(example_queries):
            if st.button(f"📝 {example}", key=f"example_{i}"):
                st.session_state.example_query = example
                st.rerun()
        
        if 'example_query' in st.session_state:
            query = st.session_state.example_query
    
    # Analysis button
    if uploaded_file and query and st.button("🔍 Analyze Scene", type="primary", use_container_width=True):
        analyze_scene(image, query, gemini_model, confidence_threshold, max_detections)

def analyze_scene(image, query, gemini_model, confidence_threshold, max_detections):
    """Simplified scene analysis function"""
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Step 1: Object Detection
        status_text.text("🎯 Detecting objects in the image...")
        progress_bar.progress(20)
        
        detections_df = get_yolo_detections(st.session_state.yolo_model, image)
        
        if detections_df.empty:
            st.warning("⚠️ No objects detected in the image. Try with a different image.")
            return
        
        # Limit detections
        detections_df = detections_df.head(max_detections)
        
        # Step 2: Process query
        status_text.text("🧠 Processing your query with AI...")
        progress_bar.progress(40)
        
        query_result = process_query_with_gemini(query, gemini_model)
        
        # Step 3: Create crops for relevant detections
        status_text.text("✂️ Extracting relevant image regions...")
        progress_bar.progress(60)
        
        # Simple filtering - just use all detections for now
        crops = []
        crop_info = []
        
        for _, detection in detections_df.iterrows():
            try:
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)
                crop_info.append({
                    'bbox': (x1, y1, x2, y2),
                    'object': detection['name'],
                    'confidence': detection['confidence']
                })
            except Exception as e:
                continue
        
        # Step 4: Calculate similarity
        status_text.text("🔍 Finding the best matches...")
        progress_bar.progress(80)
        
        if crops:
            similarity_scores = calculate_clip_similarity_simple(
                query_result['refined_prompt'], 
                crops, 
                st.session_state.clip_model, 
                st.session_state.text_processor
            )
            
            # Combine results
            results = []
            for i, (crop, info, score) in enumerate(zip(crops, crop_info, similarity_scores)):
                results.append({
                    'crop': crop,
                    'info': info,
                    'similarity': score,
                    'rank': i + 1
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
        else:
            results = []
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display results
        time.sleep(0.5)
        progress_container.empty()
        
        display_results(image, detections_df, query_result, results)
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        progress_container.empty()

def display_results(image, detections_df, query_result, results):
    """Display analysis results"""
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Query analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Object Detections")
        
        # Plot detections
        fig = plot_detection_results(image, detections_df, "All Detected Objects")
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        
        # Detection summary
        st.write(f"**Found {len(detections_df)} objects:**")
        object_counts = detections_df['name'].value_counts()
        for obj, count in object_counts.items():
            st.write(f"- {obj}: {count}")
    
    with col2:
        st.subheader("🧠 Query Analysis")
        
        st.write("**Refined Query:**")
        st.info(query_result['refined_prompt'])
        
        st.write("**Key Objects:**")
        st.write(", ".join(query_result['objects']))
        
        st.write("**Actions:**")
        st.write(", ".join(query_result['actions']))
        
        st.write("**Primary Focus:**")
        st.write(query_result['primary_object'])
    
    # Top matches
    if results:
        st.subheader("🏆 Best Matches")
        
        # Show top 3 results
        top_results = results[:3]
        
        cols = st.columns(len(top_results))
        
        for i, (col, result) in enumerate(zip(cols, top_results)):
            with col:
                st.image(
                    result['crop'], 
                    caption=f"#{i+1}: {result['info']['object']}\nSimilarity: {result['similarity']:.3f}",
                    use_column_width=True
                )
                
                # Details
                st.write(f"**Object:** {result['info']['object']}")
                st.write(f"**Confidence:** {result['info']['confidence']:.2f}")
                st.write(f"**Similarity:** {result['similarity']:.3f}")
        
        # Detailed results table
        with st.expander("📋 Detailed Results"):
            results_data = []
            for i, result in enumerate(results):
                results_data.append({
                    'Rank': i + 1,
                    'Object': result['info']['object'],
                    'Detection Confidence': f"{result['info']['confidence']:.2f}",
                    'Similarity Score': f"{result['similarity']:.3f}",
                    'Coordinates': f"({result['info']['bbox'][0]}, {result['info']['bbox'][1]}) to ({result['info']['bbox'][2]}, {result['info']['bbox'][3]})"
                })
            
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    else:
        st.warning("⚠️ No matching regions found. Try adjusting your query or using a different image.")
    
    # Save to history
    st.session_state.analysis_history.append({
        'timestamp': time.time(),
        'query': query_result['refined_prompt'],
        'num_detections': len(detections_df),
        'best_match_score': results[0]['similarity'] if results else 0
    })

if __name__ == "__main__":
    main()
