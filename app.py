import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import easyocr
from pyzbar.pyzbar import decode
import plotly.express as px
import os
import datetime

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="SmartMart AI", page_icon="🛒", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0A192F; color: #FFFFFF; font-family: sans-serif; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 65, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .text-green { color: #00FF41; }
</style>
""", unsafe_allow_html=True)

DB_PATH = 'smartmart.db'

# ==========================================
# 2. DATABASE ENGINE (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS products 
                      (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, 
                       barcode TEXT UNIQUE, stock_quantity INTEGER DEFAULT 0, reorder_threshold INTEGER DEFAULT 5)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections 
                      (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                       detected_class TEXT, confidence REAL, source TEXT, status TEXT)''')
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        default_products = [
            ('Bottle', 'Beverages', '0000001', 20, 5), ('Apple', 'Produce', '0000002', 2, 10),
            ('Cup', 'Homeware', '0000003', 15, 5), ('Orange', 'Produce', '0000004', 0, 5)
        ]
        cursor.executemany("INSERT INTO products (name, category, barcode, stock_quantity, reorder_threshold) VALUES (?, ?, ?, ?, ?)", default_products)
    conn.commit()
    conn.close()

init_db()

def log_detection(detected_class, confidence, source, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO detections (detected_class, confidence, source, status) VALUES (?, ?, ?, ?)", 
                 (detected_class, confidence, source, status))
    conn.commit()
    conn.close()

def get_inventory():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    return df

# ==========================================
# 3. FALLBACK AI ENGINE (YOLO -> Barcode -> OCR)
# ==========================================
@st.cache_resource
def load_ai_models():
    try:
        yolo_model = YOLO('yolov8n.pt')
    except Exception:
        yolo_model = None
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return yolo_model, ocr_reader

yolo, ocr = load_ai_models()

def process_image(pil_image, conf_threshold=0.6):
    if yolo is None: return None, "Error: yolov8n.pt missing."
    
    cv_img = np.array(pil_image)
    cv_img = cv_img[:, :, ::-1].copy() # RGB to BGR for OpenCV
    
    results = yolo(cv_img)[0]
    detections = []
    
    for box in results.boxes:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = yolo.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        crop = cv_img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        source, status = "YOLO Primary", "Verified"
        
        # --- THE FALLBACK LOGIC ---
        if conf < conf_threshold:
            barcodes = decode(crop)
            if barcodes:
                class_name = f"Barcode: {barcodes[0].data.decode('utf-8')}"
                source, conf = "Barcode Fallback", 1.0
            else:
                ocr_res = ocr.readtext(crop, detail=0)
                if ocr_res:
                    class_name = f"OCR: {ocr_res[0]}"
                    source, conf = "OCR Fallback", 0.8
                else:
                    class_name, source, status = "Unknown Product", "Failed", "Manual Review Required"
                    
        detections.append({'class_name': class_name, 'confidence': round(conf, 2), 'source': source, 'status': status})
        
        # Draw bounding boxes
        color = (0, 255, 0) if status == "Verified" else (0, 255, 255) if "Fallback" in source else (0, 0, 255)
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(cv_img, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return Image.fromarray(cv_img[:, :, ::-1]), detections

# ==========================================
# 4. NAVIGATION & VIEWS
# ==========================================
st.sidebar.title("SmartMart AI Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📷 Smart Scanner", "📊 Analytics Dashboard"])

if page == "🏠 Home":
    st.title("🛒 SmartMart AI")
    st.markdown("### <span class='text-green'>Transforming Retail with Intelligent Computer Vision</span>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <h4>Automated Product Detection & Inventory Intelligence</h4>
        <p>SmartMart AI utilizes advanced object detection fused with a dynamic fallback pipeline (OCR & Barcode mapping) to ensure reliable shelf auditing.</p>
        <b>Workflow:</b> Upload → Detect → Fallback Verify → Analyse → Export
    </div>
    """, unsafe_allow_html=True)

elif page == "📷 Smart Scanner":
    st.title("📷 Smart Scanner & Review")
    img_file = st.file_uploader("Upload Shelf Image", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Raw Input", width=400)
        
        if st.button("Run Intelligent Detection"):
            with st.spinner("Running Primary YOLO & Fallback Pipeline..."):
                annotated_img, detections = process_image(image)
                st.image(annotated_img, caption="AI Vision Output", use_column_width=True)
                
                st.subheader("Detection Quality Centre")
                for d in detections:
                    log_detection(d['class_name'], d['confidence'], d['source'], d['status'])
                    if d['status'] == "Manual Review Required":
                        st.error(f"⚠️ **{d['class_name']}** | Source: {d['source']} | Needs Manual Input")
                    elif "Fallback" in d['source']:
                        st.warning(f"🔍 **{d['class_name']}** | Conf: {d['confidence']} | Source: {d['source']}")
                    else:
                        st.success(f"✅ **{d['class_name']}** | Conf: {d['confidence']} | Source: {d['source']}")

elif page == "📊 Analytics Dashboard":
    st.title("📊 Inventory Analytics Dashboard")
    df = get_inventory()
    
    def get_stock_status(row):
        if row['stock_quantity'] == 0: return 'Out of Stock'
        elif row['stock_quantity'] <= row['reorder_threshold']: return 'Low Stock'
        return 'In Stock'
        
    df['status'] = df.apply(get_stock_status, axis=1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Database Inventory")
        st.dataframe(df[['name', 'category', 'stock_quantity', 'status']])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Inventory CSV", data=csv, file_name='inventory.csv')
        
    with col2:
        st.subheader("Stock Status Distribution")
        fig = px.pie(df, names='status', color='status', 
                     color_discrete_map={'In Stock':'#00FF41', 'Low Stock':'#FFD700', 'Out of Stock':'#FF003C'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Restocking Recommendations")
    alerts = df[df['status'] != 'In Stock']
    if not alerts.empty:
        for _, row in alerts.iterrows():
            st.error(f"🚨 Priority Restock: **{row['name']}** (Current: {row['stock_quantity']}, Threshold: {row['reorder_threshold']})")
