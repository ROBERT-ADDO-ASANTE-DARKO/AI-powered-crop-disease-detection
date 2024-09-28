```
# Crop Disease and Tomato Freshness Detection

This project is a web-based application that utilizes **computer vision** and **natural language processing** to detect crop diseases and assess tomato freshness. The application leverages **YOLOv8** for object detection and **Google's Gemini model** for text generation.

## File Structure

The project consists of the following files and directories:

```plaintext
models/                          # Directory containing the pre-trained YOLOv8 models
  ├── crop_disease_model.pt       # YOLOv8 model for crop disease detection
  └── tomato_freshness_model.pt   # YOLOv8 model for tomato freshness detection

app.py                            # Main application file that defines the Gradio interface and inference functions
requirements.txt                  # List of required packages for installation
README.md                         # This file
```

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Required Packages:

- `gradio` - For building the web interface
- `numpy` - Numerical operations
- `opencv-python` - Image processing library
- `Pillow` - Image manipulation
- `ultralytics` - YOLOv8 model framework
- `google-cloud-generativeai` - Google's generative AI for text generation

## Environment Variables

The project requires a Google API key for the text generation component. Set the `GOOGLE_API_KEY` environment variable by running the following command:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

Replace `YOUR_API_KEY` with your actual Google API key from [Google Cloud](https://cloud.google.com/docs/authentication/api-keys).

## Running the Application

To run the application, execute the following command in your terminal:

```bash
python app.py
```

This will launch the Gradio interface. You can access the application in your web browser by navigating to:

```
http://localhost:7860
```

## Usage

1. **Upload an image** of a crop or tomato to the application.
2. **Select the model type**:
   - Crop Disease Detection
   - Tomato Freshness Detection
3. **Adjust the confidence threshold** and **IoU threshold** as needed.
4. Click the **"Submit"** button to run the inference.
5. The application will display the **processed image** and generate a **text description** of the detected objects.

## Notes

- This project is for **educational purposes only**.
- The accuracy of the models may vary based on the quality of the input images and the specific use case.

---

**Enjoy detecting crop diseases and tomato freshness with this application!**
