import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests

# ---------------------------
# Sidebar: Description About the App
# ---------------------------
st.sidebar.title("About This App")
st.sidebar.markdown(
    """
    **Lettuce Disease Classifier**

    This web application helps you classify diseases on lettuce leaves using a deep learning model.

    Simply upload your image(s) below, and the model will predict the disease condition of your lettuce.
    """
)

disease_descriptions = {
    "bacterial": "Bacterial diseases cause spots and lesions on lettuce leaves due to bacterial infection.",
    "downy": "Downy mildew appears as a fuzzy, discolored growth on the underside of leaves.",
    "fungal": "Fungal diseases are marked by moldy growth and discoloration on leaves.",
    "healthy": "Healthy lettuce shows vibrant, green leaves that are free from disease symptoms.",
    "powdery": "Powdery mildew is characterized by a white, powdery coating on the leaf surface.",
    "septoria": "Septoria leaf spot presents as small, circular spots with dark borders on the leaves.",
    "unhealthy": "Unhealthy leaves may show general discoloration, wilting, or other signs of stress.",
    "viral": "Viral infections often lead to mottled, deformed, or stunted leaves.",
    "wilt": "Wilt is indicated by drooping, water-soaked tissues and a loss of firmness in the leaves."
}

st.sidebar.markdown("### Disease Category Description")
#create a toggle (expander) for each disease category
for disease, description in disease_descriptions.items():
    with st.sidebar.expander(disease.capitalize()):
        st.write(description)


# ---------------------------
# 1. Load the Pretrained Model
# ---------------------------
@st.cache_resource
def load_model(checkpoint_path, num_classes=9):
    """
    Loads the pretrained EfficientNet-B0 model with a custom classifier.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model (must match the model used during training)
    model = models.efficientnet_b0(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    # Load model weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


# Adjust the checkpoint path if necessary.
checkpoint_path = "pretrained-efficientnet_model.pth"
model, device = load_model(checkpoint_path, num_classes=9)
# st.write("Model loaded successfully!")

# ---------------------------
# 2. Define the Transformation
# ---------------------------
# These transforms must match those used during training.
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# 3. Define Class Names Mapping
# ---------------------------
# Modify this list so that it matches the order of classes used in your training.
class_names = [
    "bacterial", "downy", "fungal",
    "healthy", "powdery", "septoria",
    "unhealthy", "viral", "wilt"
]

# ---------------------------
# Toolbar navigation
# ---------------------------
tabs = st.tabs(["📊 ThingSpeak Data Monitoring", "🌱 Lettuce Disease Classification"])



# ---------------------------
# Streamlit UI: Select Images and Predict
# ---------------------------
with tabs[1]:
    st.title("Smart Vertical Farming: Disease Detection")
    st.write("Select one or more image files using the file uploader below:")

    # Use the file uploader to allow manual selection of images.
    uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Predict"):
            st.write("### Predictions")
            # Lists to store images and predicted labels.
            images_list = []
            predicted_labels = []

            # Process each uploaded file.
            for uploaded_file in uploaded_files:
                # Open the image.
                image = Image.open(uploaded_file).convert("RGB")
                # Preprocess the image.
                input_tensor = test_transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, pred = torch.max(outputs, 1)

                # Map the numerical prediction to a class name.
                predicted_label = class_names[pred.item()]

                images_list.append(image)
                predicted_labels.append(predicted_label)

            # ---------------------------
            # 5. Display the Images in a Grid with 5 Columns per Row
            # ---------------------------
            num_cols = 5
            num_images = len(images_list)
            for i in range(0, num_images, num_cols):
                cols = st.columns(num_cols)
                for j, image in enumerate(images_list[i:i + num_cols]):
                    caption = f"Predicted: {predicted_labels[i + j]}"
                    cols[j].image(image, caption=caption, use_container_width=True)


# ---------------------------
# ThingSpeak data monitoring
# ---------------------------
with tabs[0]:
    st.title("Smart Vertical Farming: Real-time Data From Your System")
    #User inputs
    channel_id = 2776169
    read_api_key = "QUYUP24IJN8OHY5S"
    num_results = 20
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={read_api_key}&results={num_results}"
    try:
        response = requests.get(url)
        data = response.json()

        if "feeds" in data:
            feeds = data['feeds']
            df = pd.DataFrame(feeds)

            #display data
            with st.expander("### Retrieved Data From Your System"):
                st.dataframe(df)

            # Pump Status Display
            if 'field3' in df:
                latest_pump_state = int(df['field3'].dropna().iloc[-1])  # Get the latest pump status
                pump_status = "ON" if latest_pump_state == 1 else "OFF"
                color = "green" if latest_pump_state == 1 else "red"

                # Elegant Pump Status Indicator
                pump_status_html = f"""
                <div style='display: flex; align-items: center; gap: 8px; text-align: left;'>
                    <div style='
                        width: 18px; 
                        height: 18px; 
                        background-color: {color}; 
                        border-radius: 50%; 
                        border: 2px solid black;
                        box-shadow: 0 0 5px {color};
                    '></div>
                    <span style='font-size: 16px; font-weight: bold; color: #333;'>Pump Status:</span>
                    <span style='font-size: 16px; font-weight: bold; color: {color};'>{pump_status}</span>
                </div>
                """
                st.markdown(pump_status_html, unsafe_allow_html=True)

            # 1️⃣ Field 1 - Temperature (Line Chart)
            st.write("Smart farming systems use an intelligent on/off control to prevent overload.  Activating components (irrigation, sensors, lighting) only when needed, based on real-time data, saves energy, protects equipment, and optimizes resource use.  This dynamic on/off cycle ensures efficient and sustainable operation.")
            st.write("The system will sleep for an hour after a few hours of working!")
            if 'field1' in df:
                st.subheader("🌡️ Temperature")
                st.line_chart(df.set_index('created_at')['field1'].astype(float))

            # 2️⃣ Field 2 - Light Intensity (Line Chart)
            if 'field2' in df:
                st.subheader("💡 Light Intensity")
                st.line_chart(df.set_index('created_at')['field2'].astype(float))



        else:
            st.warning("No data found or error occur in the system.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

