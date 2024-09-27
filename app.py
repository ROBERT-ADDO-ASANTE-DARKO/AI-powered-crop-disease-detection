import gradio as gr
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import os
import google.generativeai as genai

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key is None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key is None:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set it in your environment or pass it to the function."
        )
genai.configure(api_key=api_key)

# Generation config for Google Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

# Safety settings for Google Gemini
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Load the models
yolo_model_crop_disease = YOLO("models/crop_disease_model.pt")
yolo_model_tomato = YOLO("models/tomato_freshness_model.pt")

# Load the Gemini model for text generation
def load_gemini_model():
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model

gemini_model = load_gemini_model()

# Inference function for YOLOv8
# Inference function to choose between models
def inference(image, model_type):
    # Load the appropriate YOLO model based on the user's selection
    if model_type == "Crop Disease Detection":
        results = yolo_model_crop_disease(image, conf=0.4)
    else:
        results = yolo_model_tomato(image, conf=0.4)

    # Initialize output and class details
    infer = np.zeros(image.shape, dtype=np.uint8)
    classes = dict()
    names_infer = []

    # Process the detection results
    for r in results:
        infer = r.plot()  # Visualize detection results
        classes = r.names  # Retrieve class names
        names_infer = r.boxes.cls.tolist()  # Get detected class indices

    return infer, names_infer, classes

# Function to generate description using Gemini model based on predictions
def generate_description(detected_classes, class_names, user_text, model_type):
    # Map the detected class indices to their corresponding class names
    detected_objects = [class_names[cls] for cls in detected_classes]
    
    # Modify the prompt based on the selected model
    if model_type == "Crop Disease Detection":
        prompt = f"""
        You are crop disease pathologist with extensive knowledge in agriculture.
        Your task is interpret the diagnoses of the infected crops.
        The following crop diseases have been detected based on the analysis: {', '.join(detected_objects)}.
        
        Please provide a detailed explanation of each disease including:
        - The nature of the disease
        - Typical symptoms and effects on crops
        - Recommended treatment options
        - Preventative measures to avoid future occurrences.
        """
    else:
        prompt = f"""
        The following condition of the tomato has been detected: {', '.join(detected_objects)}.
        
        Please provide a detailed explanation on:
        - Whether the tomato is fresh or rotten
        - How this condition is identified (e.g., characteristics)
        - Any handling recommendations for the tomato (e.g., consumption, disposal).
        """

    # Generate content using the Gemini model
    response = gemini_model.generate_content(prompt)

    return response.text

# Gradio app
with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="numpy", label="Upload Image")
            conf_threshold = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
            iou_threshold = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
            model_type = gr.Dropdown(choices=["Crop Disease Detection", "Tomato Freshness Detection"], label="Select Model")

        with gr.Column():
            processed_image_output = gr.Image(type="pil", label="Processed Image")
            with gr.Column():
                chatbot = gr.Chatbot()
                #msg = gr.Textbox(label="Your Question")
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")

        def respond(img, conf_threshold, iou_threshold, chat_history, model_type):
            # Run YOLOv8 inference on the image based on the selected model
            processed_img, names_infer, classes = inference(img, model_type)
    
            # Get the last user message from the chat history, if any
            if chat_history:
                last_user_message = chat_history[-1][0] 
            else:
                last_user_message = ""  # Default to empty string if no history
    
            # Convert detected objects to text and generate a response using Gemini
            response = generate_description(names_infer, classes, last_user_message, model_type)
    
            # Append the user's question and AI's response to the chat history
            chat_history.append((last_user_message, response)) # Fixed: Add user message
    
            return processed_img, chat_history, response

    submit.click(respond, [img, conf_threshold, iou_threshold, chatbot, model_type], [processed_image_output, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    iface.launch()