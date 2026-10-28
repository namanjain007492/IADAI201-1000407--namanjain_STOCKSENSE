import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import datetime
import time

# ==========================================
# 1. FUTURISTIC UI & GLASSMORPHISM STYLING
# ==========================================
st.set_page_config(page_title="StockSense Pro AI", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #00ffcc; font-family: 'Courier New', Courier, monospace; }
    .glass-panel {
        background: rgba(16, 24, 43, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 204, 0.2);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 255, 204, 0.1);
    }
    .metric-text { font-size: 1.2rem; color: #ffffff; }
    .status-green { color: #00FF41; text-shadow: 0 0 5px #00FF41; font-weight: bold; }
    .status-yellow { color: #FFD700; text-shadow: 0 0 5px #FFD700; font-weight: bold; }
    .status-red { color: #FF003C; text-shadow: 0 0 5px #FF003C; font-weight: bold; }
    h1, h2, h3 { color: #00ffcc; text-transform: uppercase; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AI DESCRIPTION ENGINE & TARGETS
# ==========================================
# We track these specific retail items. If 0 are found, they are marked "Out of Stock"
TARGET_INVENTORY = ['bottle', 'cup', 'apple', 'orange', 'bowl', 'banana', 'broccoli', 'carrot']

AI_DESCRIPTIONS = {
    'bottle': "Liquid container unit. High demand in beverage aisles.",
    'cup': "Standard drinking vessel. Required for hot/cold beverage consumption.",
    'apple': "Fresh organic produce. High perishability, requires constant rotation.",
    'orange': "Citrus organic produce. Rich in Vitamin C, fast-moving inventory.",
    'bowl': "Ceramic/plastic serving unit. Essential homeware item.",
    'banana': "Tropical produce unit. Extremely short shelf-life, prioritize display.",
    'broccoli': "Cruciferous vegetable unit. Requires climate-controlled shelving.",
    'carrot': "Root vegetable unit. Stable shelf-life, high culinary demand."
}

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# ==========================================
# 3. SIDEBAR: SYSTEM TELEMETRY & HEALTH
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Jonker_5.png", width=50) # Placeholder futuristic icon
    st.markdown("### 🛰️ SYSTEM TELEMETRY")
    st.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}")
    st.write(f"**Time:** {datetime.datetime.now().strftime('%H:%M:%S')}")
    st.write("**Model:** YOLOv8 Nano (Neural Engine)")
    st.write("**Status:** 🟢 ONLINE & SECURE")
    st.markdown("---")
    st.markdown("### 📋 UPLOAD FEED")
    uploaded_file = st.file_uploader("Initialize Visual Scan (Upload Image)", type=['jpg', 'jpeg', 'png'])

st.title("🛒 STOCKSENSE PRO: COGNITIVE RETAIL VISION")
st.markdown("### Automated Shelf Intelligence & Space Optimization Platform")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("### 📷 RAW SENSOR FEED")
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.spinner("Processing Neural Network..."):
        start_time = time.time()
        results = model(image)
        inference_time = round(time.time() - start_time, 3)
        
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[..., ::-1])
        
        with col2:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("### 🧠 AI VISION OUTPUT")
            st.image(res_image, use_column_width=True)
            st.markdown(f"<p style='color:#00ffcc;'>⚡ Inference Speed: {inference_time} seconds</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # ==========================================
        # 4. INVENTORY LOGIC & ANALYTICS
        # ==========================================
        st.markdown("---")
        st.markdown("## 📊 SMART SHELF ANALYTICS & INSIGHTS")
        
        # Count detected objects
        detected_counts = {}
        names = model.names
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            if class_name in TARGET_INVENTORY:
                detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
        
        # Ensure all target categories are evaluated (even if 0 detected)
        inventory_data = []
        for item in TARGET_INVENTORY:
            count = detected_counts.get(item, 0)
            
            # Smart Assignment Logic
            if count == 0:
                status = "Out of stock"
                alert = "CRITICAL: Restock Immediately"
                priority = "High"
            elif 1 <= count <= 3:
                status = "Low stock"
                alert = "WARNING: Queue for Restock"
                priority = "Medium"
            else:
                status = "In stock"
                alert = "OPTIMAL: Stock sufficient"
                priority = "Low"
                
            desc = AI_DESCRIPTIONS.get(item, "Standard retail unit.")
            inventory_data.append({"Product": item.capitalize(), "Count": count, "Status": status, "Alert": alert, "Priority": priority, "AI Description": desc})
        
        df = pd.DataFrame(inventory_data)
        
        # ==========================================
        # 5. DATA VISUALIZATION & EXPORT
        # ==========================================
        col3, col4 = st.columns([2, 1])
        
        with col3:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("### 📝 DETAILED INVENTORY LEDGER")
            # Style the dataframe based on status
            def color_status(val):
                color = '#FF003C' if val == 'Out of stock' else '#FFD700' if val == 'Low stock' else '#00FF41'
                return f'color: {color}'
            
            st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)
            
            # Export Feature
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Inventory CSV Report", data=csv, file_name='stocksense_report.csv', mime='text/csv')
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("### 📈 STOCK DISTRIBUTION")
            st.bar_chart(df.set_index('Product')['Count'])
            st.markdown("</div>", unsafe_allow_html=True)
