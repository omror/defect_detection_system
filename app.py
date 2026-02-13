import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Industrial Defect Detection",
    layout="centered",
)

# 2. Model Loading (Cached)
@st.cache_resource
def load_model():
    try:
        # Initialize architecture
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

        # Load the trained weights from the file
        # Ensure defect_model.pth is in the same directory
        model.load_state_dict(torch.load('defect_model.pth', map_location=torch.device('cpu')))
        model.eval() # Lock the model for testing
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Make sure 'defect_model.pth' is in the same directory.")
        return None

model = load_model()

# 3. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image, model):
    # Preprocess
    tensor = transform(image).unsqueeze(0) # Add batch dimension -> [1, 3, 224, 224]

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Class mapping (Alphabetical order of folders: defect, normal)
    class_names = ['defect', 'normal']
    result = class_names[predicted.item()]
    
    return result, confidence.item()

# 4. Main UI
st.title("Industrial Defect Detection System")
st.markdown("### AI powered quality control for manufacturing line")
st.write("This system utilizes a **ResNet18** model to detect manufacturing faults in casting products in real-time.")

# Sidebar
st.sidebar.header("Configuration")
if model:
    st.sidebar.success("Model loaded successfully")
else:
    st.sidebar.error("Model failed to load")

# Input Selection
st.sidebar.divider()
input_method = st.sidebar.radio("Select Input Method:", ["Upload Image", "Use Sample Image"])

image = None
image_source = ""

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a component image for inspection:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_source = "Uploaded Image"

elif input_method == "Use Sample Image":
    sample_type = st.sidebar.selectbox("Select Sample Type:", ["Defect", "Normal"])
    
    # Define sample paths (assuming data folder structure exists)
    # Define sample paths
    sample_dir = "sample_images"
    
    if sample_type == "Defect":
        target_dir = os.path.join(sample_dir, "defect")
    else:
        target_dir = os.path.join(sample_dir, "normal")

    if os.path.exists(target_dir):
        # Get all image files
        all_files = os.listdir(target_dir)
        sample_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if sample_files:
            selected_file = st.sidebar.selectbox(f"Choose a {sample_type.lower()} sample:", sample_files)
            image_path = os.path.join(target_dir, selected_file)
        else:
            st.sidebar.warning(f"No images found in {target_dir}")
            image_path = None
    else:
        st.sidebar.error(f"Directory not found: {target_dir}")
        image_path = None
    
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image_source = f"Sample: {selected_file}"
    else:
        st.error(f"Sample file not found: {image_path}")

# Display and Analyze
if image is not None and model is not None:
    st.divider()
    
    # Display image
    st.image(image, caption=image_source, use_container_width=True)
    
    # Analysis Button
    if st.button("Start Inspection", type="primary"):
        with st.spinner("AI is analyzing the component..."):
            try:
                prediction, confidence_score = predict(image, model)
                
                st.divider()
                st.subheader("Inspection Result")

                # Creating columns for metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="Model Prediction", value=prediction.upper())

                with col2:
                    st.metric(label="Confidence Score", value=f"{confidence_score * 100:.2f}%")

                # Visual Feedback
                if prediction == "defect":
                    st.error("Defect detected! Reject the component.")
                else:
                    st.success("Component is normal. Proceed to packaging.")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

elif model is None:
    st.warning("Please resolve the model loading error to proceed.")
