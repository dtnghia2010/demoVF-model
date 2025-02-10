import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


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
st.write("Model loaded successfully!")

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
# 4. Streamlit UI: Select Images and Predict
# ---------------------------
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
