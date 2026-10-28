#  SmartMart AI — Intelligent Retail Product Detection System

**SmartMart AI** is an AI-powered retail application designed to help simplify product identification, barcode scanning, inventory management, and sales-related analysis. It combines computer vision, optical character recognition (OCR), and database management in an interactive Streamlit dashboard.

App Link : https://iadai201-1000407--namanjainstocksense-m4xrnj7dvvb9puz2of2eqi.streamlit.app/

Screenshot Link:https://drive.google.com/drive/folders/1uFNzwtugOc9HbYZY5nOzGEppGys8zCXU?usp=sharing

##  Features

- **AI Product Detection:** Detects objects in uploaded images using the YOLO-World model.
- **Image Tiling:** Analyzes image sections to help identify smaller or less prominent objects.
- **Product Text Recognition:** Uses EasyOCR to extract visible text from product packaging.
- **Barcode Scanning:** Uses Pyzbar to read supported barcodes when visible and clear.
- **Inventory Management:** Stores and manages product information in an SQLite database.
- **Detection Records:** Maintains records of processed detections.
- **Manual Review:** Supports checking and correcting uncertain product identifications.
- **Analytics Dashboard:** Presents product and detection information through interactive charts.
- **Reports:** Helps users review available inventory and detection records.

##  Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Ultralytics YOLO-World | Open-vocabulary object detection |
| PyTorch | Deep-learning framework |
| EasyOCR | Text recognition from images |
| OpenCV | Image processing |
| Pillow | Image handling |
| Pyzbar | Barcode decoding |
| SQLite | Local database storage |
| Pandas and NumPy | Data processing |
| Plotly | Interactive charts and analytics |

##  Project Structure

```text
SmartMart-AI/
├── app.py
├── requirements.txt
├── README.md
└── smartmart.db          # Created by the application, if configured
```

The database filename and additional files may differ depending on your application configuration.

##  Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SmartMart-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** Pyzbar may require the ZBar library to be installed on your operating system. Check the installation instructions for your platform if barcode scanning fails.

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL where you can open SmartMart AI in your browser.

## How to Use

1. Launch the application.
2. Upload an image containing retail products.
3. Run the AI detection process.
4. Review the detected objects and any text or barcode results.
5. Check uncertain results and correct product information where necessary.
6. Manage the product catalogue and inventory.
7. Explore the analytics dashboard and available reports.

##  How It Works

1. **Image input:** The user supplies a retail product image.
2. **Preprocessing:** The image is prepared for computer vision analysis and may be divided into smaller sections.
3. **Object detection:** YOLO-World identifies objects based on the categories or text prompts provided by the application.
4. **OCR and barcode processing:** EasyOCR and Pyzbar attempt to extract additional product information.
5. **Database operations:** The application stores and retrieves supported product and detection records.
6. **Visualisation:** Streamlit and Plotly present results in an accessible dashboard.

##  Limitations

- Detection accuracy depends on image quality, lighting, object size, packaging, and the model's capabilities.
- Similar-looking products and small snack packages may be missed or incorrectly classified.
- Detecting a product category does not guarantee identification of its exact brand or variant.
- OCR may fail when text is small, blurred, curved, or obstructed.
- Barcode recognition requires a readable barcode and compatible decoding support.
- AI-generated detections should be reviewed before being used to update inventory.
- Detection results are not guaranteed to be 100% accurate.

##  Data and Privacy

The application uses SQLite for local database storage when configured that way. Image handling, model downloads, and other processing may involve additional files or external resources depending on the deployment configuration. Avoid uploading confidential customer or business information unless the deployment has been reviewed for privacy and security.

##  Credits and Acknowledgements

SmartMart AI is built using the work and open-source contributions of the following projects:

- **Ultralytics** — YOLO and YOLO-World object detection framework.  
  https://github.com/ultralytics/ultralytics
- **YOLO-World research team** — Open-vocabulary object detection research.  
  https://github.com/AILab-CVC/YOLO-World
- **PyTorch** — Deep-learning framework.  
  https://pytorch.org/
- **Streamlit** — Framework for interactive Python applications.  
  https://streamlit.io/
- **JaidedAI** — EasyOCR text-recognition library.  
  https://github.com/JaidedAI/EasyOCR
- **OpenCV** — Computer vision and image-processing library.  
  https://opencv.org/
- **Pyzbar** — Python interface for barcode decoding.  
  https://github.com/NaturalHistoryMuseum/pyzbar
- **Pandas** — Data analysis and manipulation tools.  
  https://pandas.pydata.org/
- **NumPy** — Numerical computing library.  
  https://numpy.org/
- **Plotly** — Interactive data visualisation library.  
  https://plotly.com/python/
- **SQLite** — Lightweight database engine.  
  https://sqlite.org/

### Research Reference

Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., & Shan, Y. (2024). *YOLO-World: Real-Time Open-Vocabulary Object Detection*.  
https://arxiv.org/abs/2401.17270

Please review the licence terms and attribution requirements of each dependency and model before redistributing or deploying the project.

##  Project Author

**Developed by:** Naman Jain

**Project:** SmartMart AI — Intelligent Retail Product Detection System

**Mentor:** Ms. Arul  Jothi

This project explores the application of artificial intelligence and computer vision to retail automation, product recognition, and inventory analytics.

SmartMart AI is an educational and experimental project. It is not a substitute for verified inventory records or human review. Detection performance depends on the model, prompts, image quality, and deployment environment.
