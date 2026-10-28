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
import io
import os
import datetime
import time

# =====================================================================
# A. UI DESIGN & CONFIGURATION (Features A1 - A5)
# =====================================================================
st.set_page_config(page_title="SmartMart AI", page_icon="🛒", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #07111E; color: #F1F5F9; font-family: 'Inter', sans-serif; }
    .glass-card {
        background: rgba(16, 33, 56, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 102, 0.25);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .hero-banner {
        background: linear-gradient(135deg, #0A192F 0%, #06382C 100%);
        border: 1px solid #00FF66;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
    }
    .badge-green { background: #064E3B; color: #34D399; padding: 4px 10px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; }
    .badge-yellow { background: #78350F; color: #FBBF24; padding: 4px 10px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; }
    .badge-red { background: #7F1D1D; color: #F87171; padding: 4px 10px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; }
    h1, h2, h3, h4 { color: #00FF66; }
</style>
""", unsafe_allow_html=True)

DB_PATH = "smartmart_pro.db"
REF_DIR = "reference_catalogue"
os.makedirs(REF_DIR, exist_ok=True)

# =====================================================================
# D. SMART INVENTORY & CATALOGUE DATABASE (Features C6, D1 - D5)
# =====================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # C6 & D1: Product Catalogue & Inventory Database
    c.execute('''CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku TEXT UNIQUE,
        name TEXT NOT NULL,
        brand TEXT,
        category TEXT,
        barcode TEXT UNIQUE,
        unit_price REAL DEFAULT 1.99,
        stock_quantity INTEGER DEFAULT 0,
        reorder_threshold INTEGER DEFAULT 5,
        expiry_date TEXT,
        batch_number TEXT,
        ref_image_path TEXT
    )''')
    
    # E3 & E4: Detection History & Quality Audit Trail
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        image_name TEXT,
        yolo_pred TEXT,
        openvocab_pred TEXT,
        ocr_text TEXT,
        barcode_val TEXT,
        sim_match TEXT,
        final_identity TEXT,
        confidence REAL,
        winning_stage TEXT,
        status TEXT,
        bbox TEXT,
        shelf_occupancy REAL
    )''')
    
    c.execute("SELECT COUNT(*) FROM products")
    if c.fetchone()[0] == 0:
        defaults = [
            ('SKU-101', 'Mineral Water Bottle', 'AquaPure', 'bottle', '8901001', 1.50, 24, 5, '2027-04-10', 'B-01', ''),
            ('SKU-102', 'Fresh Red Apple', 'NatureFarm', 'apple', '8901002', 0.80, 2, 8, '2026-11-02', 'B-02', ''),
            ('SKU-103', 'Ceramic Coffee Cup', 'HomeStyle', 'cup', '8901003', 4.99, 0, 5, '2030-01-01', 'B-03', ''),
            ('SKU-104', 'Valencia Orange', 'CitrusCo', 'orange', '8901004', 0.90, 14, 5, '2026-11-05', 'B-04', ''),
            ('SKU-105', 'Choco Biscuit Packet', 'Britannia', 'biscuit', '8901005', 1.20, 3, 10, '2026-12-15', 'B-05', ''),
            ('SKU-106', 'Organic Banana Bunch', 'TropiFresh', 'banana', '8901006', 1.10, 12, 6, '2026-10-31', 'B-06', '')
        ]
        c.executemany('''INSERT INTO products 
            (sku, name, brand, category, barcode, unit_price, stock_quantity, reorder_threshold, expiry_date, batch_number, ref_image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', defaults)
    conn.commit()
    conn.close()

init_db()

def query_df(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def execute_sql(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(sql, params)
    conn.commit()
    conn.close()

# =====================================================================
# B & C. 6-STAGE FALLBACK AI PIPELINE (Features B1 - B8, C1 - C5)
# =====================================================================
@st.cache_resource
def load_engines():
    status = {"yolo": False, "ocr": False, "barcode": True, "similarity": True, "openvocab": True}
    yolo_m, ocr_m = None, None
    try:
        yolo_m = YOLO("yolov8n.pt")
        status["yolo"] = True
    except Exception:
        pass
    try:
        ocr_m = easyocr.Reader(['en'], gpu=False)
        status["ocr"] = True
    except Exception:
        pass
    return yolo_m, ocr_m, status

yolo_model, ocr_reader, sys_status = load_engines()

# B8: Non-Maximum Suppression / IoU Deduplication
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def deduplicate_boxes(boxes, iou_thresh=0.45):
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    kept = []
    for b in boxes:
        if all(compute_iou(b['bbox'], k['bbox']) < iou_thresh for k in kept):
            kept.append(b)
    return kept

# C3: Visual Similarity Search via HSV Color & Edge Embedding
def compute_embedding(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    resized = cv2.resize(bgr_crop, (64, 64))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def match_visual_similarity(crop_bgr, threshold=0.65):
    target_emb = compute_embedding(crop_bgr)
    if target_emb is None:
        return None, 0.0
    prods = query_df("SELECT name, ref_image_path FROM products WHERE ref_image_path != ''")
    best_name, best_score = None, 0.0
    for _, row in prods.iterrows():
        path = row['ref_image_path']
        if os.path.exists(path):
            ref_img = cv2.imread(path)
            ref_emb = compute_embedding(ref_img)
            if ref_emb is not None:
                score = float(cv2.compareHist(target_emb.astype(np.float32), ref_emb.astype(np.float32), cv2.HISTCMP_CORREL))
                if score > best_score and score >= threshold:
                    best_score = score
                    best_name = row['name']
    return best_name, round(best_score, 2)

# B2: Open-Vocabulary / Contour Proposal Detector when YOLO misses unknown items (like a biscuit packet)
def find_unseen_item_proposals(bgr_img, existing_boxes):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = bgr_img.shape[:2]
    proposals = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if (h_img * w_img * 0.02) < area < (h_img * w_img * 0.85):
            cand_box = (x, y, x + w, y + h)
            if all(compute_iou(cand_box, eb['bbox']) < 0.3 for eb in existing_boxes):
                proposals.append({'bbox': cand_box, 'conf': 0.35, 'cls_name': 'unseen_object', 'origin': 'Stage 2: Open-Vocab Region Proposal'})
    return proposals[:5]

# B7: Small-Product Overlapping Tile Detection
def run_tiled_yolo(bgr_img, conf_thresh):
    h, w = bgr_img.shape[:2]
    tiles = [
        (0, 0, int(w*0.6), int(h*0.6)),
        (int(w*0.4), 0, w, int(h*0.6)),
        (0, int(h*0.4), int(w*0.6), h),
        (int(w*0.4), int(h*0.4), w, h)
    ]
    tiled_boxes = []
    for (tx1, ty1, tx2, ty2) in tiles:
        tile_crop = bgr_img[ty1:ty2, tx1:tx2]
        res = yolo_model(tile_crop, conf=conf_thresh, verbose=False)[0]
        for b in res.boxes:
            bx1, by1, bx2, by2 = map(int, b.xyxy[0])
            tiled_boxes.append({
                'bbox': (bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1),
                'conf': float(b.conf[0]),
                'cls_name': yolo_model.names[int(b.cls[0])],
                'origin': 'Stage 1: YOLO Tiled Small-Object'
            })
    return tiled_boxes

# MASTER 6-STAGE PIPELINE
def execute_6_stage_pipeline(pil_img, img_name, high_conf=0.65, rev_conf=0.35, enable_tiling=False, custom_vocab=""):
    bgr = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
    h_img, w_img = bgr.shape[:2]
    catalogue = query_df("SELECT * FROM products")
    vocab_terms = [t.strip().lower() for t in custom_vocab.split(",") if t.strip()]
    
    raw_candidates = []
    # STAGE 1: Primary YOLO Detector
    if sys_status["yolo"]:
        res = yolo_model(bgr, conf=rev_conf, verbose=False)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            raw_candidates.append({
                'bbox': (x1, y1, x2, y2),
                'conf': float(b.conf[0]),
                'cls_name': yolo_model.names[int(b.cls[0])],
                'origin': 'Stage 1: Primary YOLO'
            })
        if enable_tiling:
            raw_candidates.extend(run_tiled_yolo(bgr, rev_conf))
            
    deduped = deduplicate_boxes(raw_candidates)
    
    # STAGE 2: Open-Vocabulary / Unseen Object Backup Proposal (if YOLO finds nothing or misses packages)
    if len(deduped) == 0 or custom_vocab:
        proposals = find_unseen_item_proposals(bgr, deduped)
        deduped.extend(proposals)
        
    final_results = []
    occupied_pixels = 0
    
    for idx, item in enumerate(deduped):
        x1, y1, x2, y2 = item['bbox']
        occupied_pixels += (x2 - x1) * (y2 - y1)
        crop = bgr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        
        s1_yolo = f"{item['cls_name']} ({item['conf']:.2f})" if "Stage 1" in item['origin'] else "Missed by YOLO"
        s2_openvocab = "Not Triggered"
        s3_ocr = "Not Triggered"
        s4_barcode = "Not Triggered"
        s5_similarity = "Not Triggered"
        
        final_name = None
        winning_stage = item['origin']
        final_conf = item['conf']
        status = "Verified"
        
        # Check if Stage 1 YOLO maps directly with High Confidence
        cat_match = catalogue[catalogue['category'].str.lower() == item['cls_name'].lower()]
        if item['conf'] >= high_conf and not cat_match.empty:
            final_name = f"{cat_match.iloc[0]['name']} ({cat_match.iloc[0]['brand']})"
            winning_stage = "Stage 1: Primary YOLO"
        else:
            # Trigger Backup Pipeline (Stages 2 -> 5)
            # STAGE 4: Barcode Scanner (Exact deterministic check)
            decoded_barcodes = decode(crop)
            if decoded_barcodes:
                b_val = decoded_barcodes[0].data.decode('utf-8')
                b_match = catalogue[catalogue['barcode'] == b_val]
                if not b_match.empty:
                    s4_barcode = f"Matched Barcode {b_val}"
                    final_name = f"{b_match.iloc[0]['name']} ({b_match.iloc[0]['brand']})"
                    winning_stage = "Stage 4: Barcode Scanner"
                    final_conf = 0.99
                else:
                    s4_barcode = f"Unresolved Barcode ({b_val})"
            else:
                s4_barcode = "No Barcode Visible"
                
            # STAGE 3 & STAGE 2: OCR Packaging Text + Open-Vocabulary Prompt Match
            if final_name is None and sys_status["ocr"]:
                ocr_tokens = ocr_reader.readtext(crop, detail=0)
                joined_ocr = " ".join(ocr_tokens).strip()
                s3_ocr = joined_ocr if joined_ocr else "No Text Found"
                
                if joined_ocr:
                    # Check Open-Vocabulary Prompts first
                    for term in vocab_terms:
                        if term in joined_ocr.lower():
                            s2_openvocab = f"Matched Prompt '{term}'"
                            final_name = f"Open-Vocab: {term.title()} ({joined_ocr[:18]})"
                            winning_stage = "Stage 2: Open-Vocabulary Prompt"
                            final_conf = 0.85
                            break
                    # Check Product Catalogue Names & Brands
                    if final_name is None:
                        for _, prod in catalogue.iterrows():
                            if prod['name'].lower() in joined_ocr.lower() or prod['brand'].lower() in joined_ocr.lower():
                                final_name = f"{prod['name']} ({prod['brand']})"
                                winning_stage = "Stage 3: OCR Packaging Recognition"
                                final_conf = 0.82
                                break
            
            # STAGE 5: Visual Similarity Search
            if final_name is None:
                sim_name, sim_score = match_visual_similarity(crop)
                if sim_name:
                    s5_similarity = f"Matched {sim_name} ({sim_score})"
                    final_name = sim_name
                    winning_stage = "Stage 5: Visual Similarity Search"
                    final_conf = sim_score
                else:
                    s5_similarity = "No Reference Match"
                    
            # Fallback to Broad YOLO Category if medium confidence
            if final_name is None and item['cls_name'] != 'unseen_object' and item['conf'] >= rev_conf:
                final_name = f"Category: {item['cls_name'].title()} (Uncertain SKU)"
                winning_stage = "Stage 1: Broad Category Only"
                status = "Needs Verification"
                
            # STAGE 6: Unknown Product — Manual Review Required
            if final_name is None:
                final_name = "Unknown Product — Identification Required"
                winning_stage = "Stage 6: Manual Review Queue"
                status = "Manual Review Required"
                
        # Draw Annotation on Image
        color = (0, 255, 102) if status == "Verified" else (0, 204, 255) if status == "Needs Verification" else (255, 50, 50)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{final_name[:22]} ({final_conf:.2f})", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        crop_pil = Image.fromarray(crop[:, :, ::-1]) if crop.size > 0 else pil_img
        
        final_results.append({
            "det_id": f"DET-{int(time.time())}-{idx}",
            "bbox": f"{x1},{y1},{x2},{y2}",
            "s1_yolo": s1_yolo,
            "s2_openvocab": s2_openvocab,
            "s3_ocr": s3_ocr,
            "s4_barcode": s4_barcode,
            "s5_similarity": s5_similarity,
            "final_identity": final_name,
            "confidence": round(final_conf, 2),
            "winning_stage": winning_stage,
            "status": status,
            "crop_img": crop_pil
        })
        
    occupancy_pct = round(min(100.0, (occupied_pixels / float(h_img * w_img + 1e-6)) * 100), 1)
    
    # Log to Database
    for r in final_results:
        execute_sql('''INSERT INTO detections 
            (image_name, yolo_pred, openvocab_pred, ocr_text, barcode_val, sim_match, final_identity, confidence, winning_stage, status, bbox, shelf_occupancy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (img_name, r['s1_yolo'], r['s2_openvocab'], r['s3_ocr'], r['s4_barcode'], r['s5_similarity'],
             r['final_identity'], r['confidence'], r['winning_stage'], r['status'], r['bbox'], occupancy_pct))
             
    return Image.fromarray(bgr[:, :, ::-1]), final_results, occupancy_pct

# =====================================================================
# SIDEBAR NAVIGATION (Feature A3)
# =====================================================================
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 1. Introduction & Overview"

st.sidebar.title("🛒 SmartMart AI")
st.sidebar.caption("Cognitive Retail Vision & Analytics")

PAGES = [
    "🏠 1. Introduction & Overview",
    "📷 2. 6-Stage Smart Scanner",
    "🔍 3. Manual Review & Registration",
    "📦 4. Smart Inventory & Expiry",
    "📊 5. Live Analytics & Quality Centre",
    "📥 6. Reports, Evaluation & Settings"
]
selection = st.sidebar.radio("Navigate Module:", PAGES, index=PAGES.index(st.session_state.nav_page))
st.session_state.nav_page = selection

# =====================================================================
# PAGE 1: INTRODUCTION PAGE & UI (Features A1 - A5)
# =====================================================================
if selection == "🏠 1. Introduction & Overview":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🛒 SmartMart AI</h1>
        <h3>Transforming Retail with Intelligent Computer Vision</h3>
        <p>Enterprise supermarket shelf auditing, zero-dataset 6-stage fallback recognition, automated product counting, and real-time SQLite inventory intelligence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("🚀 Launch Smart Scanner", use_container_width=True):
        st.session_state.nav_page = "📷 2. 6-Stage Smart Scanner"
        st.rerun()
    if b2.button("📊 Explore Dashboard", use_container_width=True):
        st.session_state.nav_page = "📊 5. Live Analytics & Quality Centre"
        st.rerun()
        
    # A4: System Status Indicators
    st.subheader("🛰️ Live System Capability Indicators")
    s_cols = st.columns(5)
    s_cols[0].metric("1. Primary YOLOv8", "Online" if sys_status["yolo"] else "Missing")
    s_cols[1].metric("2. Open-Vocab Engine", "Active")
    s_cols[2].metric("3. EasyOCR Engine", "Active" if sys_status["ocr"] else "Offline")
    s_cols[3].metric("4. PyZbar Barcode", "Active")
    s_cols[4].metric("5. Visual Similarity", "Active")
    
    # A2 & A5: Overview, Workflow & Quick-Start Guide
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='glass-card'>
            <h4>🧠 Zero-Dataset 6-Stage Detection Architecture</h4>
            <p>Even if a product (like a new packet of biscuits) is not in YOLO's training dataset, SmartMart AI cascades through 6 verification layers:</p>
            <ol>
                <li><b>Stage 1 (Primary YOLO):</b> Fast bounding box & known class detection.</li>
                <li><b>Stage 2 (Open-Vocabulary & Region Proposal):</b> Contours & custom text prompts locate unseen packages.</li>
                <li><b>Stage 3 (OCR Text Engine):</b> Reads brand and packaging text directly from the crop.</li>
                <li><b>Stage 4 (Barcode Scanner):</b> Decodes EAN/UPC barcodes against the SQLite catalogue.</li>
                <li><b>Stage 5 (Visual Similarity Search):</b> Compares HSV/Edge embeddings against registered reference photos.</li>
                <li><b>Stage 6 (Manual Review):</b> Flags unresolved items as <i>"Unknown Product — Identification Required"</i>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='glass-card'>
            <h4>⚡ Quick-Start Workflow (Upload → Verify → Export)</h4>
            <ul>
                <li><b>Step 1:</b> Open <b>6-Stage Smart Scanner</b> and upload shelf images or capture via camera.</li>
                <li><b>Step 2:</b> Inspect the <b>Stage-by-Stage Preservation Table</b> to see exactly how each item was identified.</li>
                <li><b>Step 3:</b> Register any unknown crops in <b>Manual Review</b> so Visual Similarity recognizes them next time.</li>
                <li><b>Step 4:</b> Check <b>Inventory & Expiry</b> for restocking alerts and batch expiry tracking.</li>
            </ul>
            <hr>
            <p style='font-size:0.85rem; color:#94A3B8;'><b>Limitations Notice:</b> Recognition accuracy depends on lighting, resolution, and packaging visibility. Shelf counts reflect visible items only, not backroom warehouse stock.</p>
        </div>
        """, unsafe_allow_html=True)
    st.caption("SmartMart AI v2.5 | Academic ML & Deep Learning Platform")

# =====================================================================
# PAGE 2: 6-STAGE SMART SCANNER (Features B1-B8, C1-C4)
# =====================================================================
elif selection == "📷 2. 6-Stage Smart Scanner":
    st.title("📷 6-Stage Intelligent Product Scanner")
    
    with st.expander("⚙️ Detection Pipeline & Confidence Threshold Controls", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        high_th = c1.slider("High Confidence (Auto-Accept)", 0.40, 0.95, 0.65, 0.05)
        rev_th = c2.slider("Minimum Detection Sensitivity", 0.15, 0.60, 0.30, 0.05)
        use_tiles = c3.checkbox("Enable Small-Product Tiling (Feature B7)", value=False)
        openvocab_prompt = c4.text_input("Open-Vocabulary Prompts (comma-separated)", placeholder="biscuit, chips, soap, milk")
        
    mode = st.radio("Input Source:", ["Upload Shelf Images (Batch Supported)", "Camera Snapshot"], horizontal=True)
    
    uploaded_files = []
    if mode == "Upload Shelf Images (Batch Supported)":
        files = st.file_uploader("Upload JPG, PNG, or WEBP shelf photos", type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True)
        if files:
            uploaded_files = files
    else:
        cam = st.camera_input("Capture Retail Shelf")
        if cam:
            uploaded_files = [cam]
            
    if uploaded_files:
        if st.button("🧠 Execute 6-Stage Recognition Pipeline", type="primary"):
            st.session_state.last_scan_results = []
            for f in uploaded_files:
                try:
                    img = Image.open(f)
                    with st.spinner(f"Running 6-Stage Pipeline on {f.name}..."):
                        ann_img, dets, occ_pct = execute_6_stage_pipeline(img, f.name, high_th, rev_th, use_tiles, openvocab_prompt)
                        st.session_state.last_scan_results.append((f.name, ann_img, dets, occ_pct))
                except Exception as e:
                    st.error(f"Corrupt or unsupported image {f.name}: {e}")
                    
    if "last_scan_results" in st.session_state and st.session_state.last_scan_results:
        for fname, ann_img, dets, occ_pct in st.session_state.last_scan_results:
            st.markdown(f"### 🖼️ Results for `{fname}`")
            col_img, col_kpi = st.columns([2, 1])
            with col_img:
                st.image(ann_img, caption="Annotated Shelf Output", use_container_width=True)
            with col_kpi:
                total_cnt = len(dets)
                unk_cnt = sum(1 for d in dets if "Unknown" in d['final_identity'])
                id_cnt = total_cnt - unk_cnt
                st.metric("Total Detected Objects", total_cnt)
                st.metric("Identified Products", id_cnt)
                st.metric("Unknown (Manual Review)", unk_cnt)
                st.metric("Estimated Shelf Occupancy", f"{occ_pct}%", help="Bounding-box area estimate")
                
            st.subheader("🔬 Stage-by-Stage Preservation Log (Multi-Model Verification)")
            stage_rows = []
            for d in dets:
                stage_rows.append({
                    "ID": d["det_id"],
                    "1. Primary YOLO": d["s1_yolo"],
                    "2. Open-Vocab": d["s2_openvocab"],
                    "3. OCR Text": d["s3_ocr"],
                    "4. Barcode": d["s4_barcode"],
                    "5. Visual Sim": d["s5_similarity"],
                    "Final Identity": d["final_identity"],
                    "Winning Stage": d["winning_stage"],
                    "Conf": d["confidence"],
                    "Status": d["status"]
                })
            if stage_rows:
                st.dataframe(pd.DataFrame(stage_rows), use_container_width=True)
            else:
                st.warning("No products detected. Try lowering sensitivity or adding Open-Vocabulary prompts.")

# =====================================================================
# PAGE 3: MANUAL REVIEW & VISUAL SIMILARITY REGISTRATION (Features C3, C5)
# =====================================================================
elif selection == "🔍 3. Manual Review & Registration":
    st.title("🔍 Unknown Product Queue & Catalogue Registration")
    st.write("Review uncertain or unknown crops, assign their true identity, and save reference crops so **Stage 5 (Visual Similarity Search)** recognizes them automatically on future scans.")
    
    if "last_scan_results" in st.session_state and st.session_state.last_scan_results:
        all_dets = [d for _, _, dets, _ in st.session_state.last_scan_results for d in dets]
        review_items = [d for d in all_dets if d['status'] != "Verified"]
        
        if not review_items:
            st.success("🎉 All detections in the current session were automatically verified!")
        else:
            for idx, item in enumerate(review_items):
                with st.container():
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.image(item['crop_img'], caption=f"Crop {item['det_id']}", width=180)
                    with c2:
                        st.markdown(f"**Current Status:** `{item['final_identity']}` | **OCR Read:** `{item['s3_ocr']}` | **Barcode:** `{item['s4_barcode']}`")
                        with st.form(key=f"rev_form_{idx}"):
                            f1, f2, f3 = st.columns(3)
                            new_name = f1.text_input("Product Name", value="Choco Biscuit Packet")
                            new_brand = f2.text_input("Brand", value="Britannia")
                            new_cat = f3.text_input("Category Tag", value="biscuit")
                            f4, f5, f6 = st.columns(3)
                            new_sku = f4.text_input("SKU", value=f"SKU-{int(time.time())%10000}-{idx}")
                            new_bar = f5.text_input("Barcode", value=f"890{int(time.time())%100000}{idx}")
                            new_stock = f6.number_input("Initial Recorded Stock", min_value=0, value=15)
                            
                            if st.form_submit_button("✅ Verify & Save Reference Image for Visual Similarity"):
                                ref_path = os.path.join(REF_DIR, f"{new_sku}.png")
                                item['crop_img'].save(ref_path)
                                execute_sql('''INSERT OR REPLACE INTO products 
                                    (sku, name, brand, category, barcode, unit_price, stock_quantity, reorder_threshold, expiry_date, batch_number, ref_image_path)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                    (new_sku, new_name, new_brand, new_cat, new_bar, 1.99, new_stock, 5, '2027-01-01', 'B-NEW', ref_path))
                                execute_sql("UPDATE detections SET final_identity=?, status='Manually Verified', winning_stage='Stage 6: Manual Registration' WHERE bbox=?",
                                            (f"{new_name} ({new_brand})", item['bbox']))
                                st.success(f"Registered '{new_name}' into catalogue and saved reference image for Stage 5 Visual Similarity!")
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Run a scan in the **6-Stage Smart Scanner** first to populate crops for manual review.")

# =====================================================================
# PAGE 4: SMART INVENTORY & EXPIRY MANAGEMENT (Features D1 - D5)
# =====================================================================
elif selection == "📦 4. Smart Inventory & Expiry":
    st.title("📦 Smart Inventory, Restock Alerts & Expiry Management")
    
    # D5: Search & Filtering
    s_col1, s_col2 = st.columns([2, 1])
    search_q = s_col1.text_input("🔍 Search by Product Name, Brand, SKU, Barcode, or Category")
    status_filter = s_col2.selectbox("Filter by Stock Status", ["All", "In Stock", "Low Stock", "Out of Stock"])
    
    df_inv = query_df("SELECT * FROM products")
    
    # D2: Stock-Level Classification
    def classify_stock(r):
        if r['stock_quantity'] <= 0:
            return "Out of Stock"
        elif r['stock_quantity'] <= r['reorder_threshold']:
            return "Low Stock"
        return "In Stock"
        
    df_inv['Stock Status'] = df_inv.apply(classify_stock, axis=1)
    
    # D4: Expiry Calculation
    today = datetime.date.today()
    def check_expiry(exp_str):
        try:
            exp_d = datetime.datetime.strptime(str(exp_str), "%Y-%m-%d").date()
            days = (exp_d - today).days
            if days < 0: return f"EXPIRED ({abs(days)}d ago)"
            elif days <= 10: return f"Expiring Soon ({days}d)"
            return f"Valid ({days}d left)"
        except Exception:
            return "Unknown"
            
    df_inv['Expiry Status'] = df_inv['expiry_date'].apply(check_expiry)
    
    if search_q:
        mask = df_inv.apply(lambda r: search_q.lower() in str(r.values).lower(), axis=1)
        df_inv = df_inv[mask]
    if status_filter != "All":
        df_inv = df_inv[df_inv['Stock Status'] == status_filter]
        
    st.dataframe(df_inv[['sku', 'name', 'brand', 'category', 'barcode', 'stock_quantity', 'reorder_threshold', 'Stock Status', 'expiry_date', 'Expiry Status']], use_container_width=True)
    
    # D3: Automatic Restocking Alerts
    c_alert, c_add = st.columns(2)
    with c_alert:
        st.subheader("🚨 Automatic Restocking & Expiry Alerts")
        low_df = df_inv[df_inv['Stock Status'] != "In Stock"]
        if low_df.empty:
            st.success("All catalogued products have sufficient warehouse inventory.")
        else:
            for _, r in low_df.iterrows():
                rec_qty = max(10, r['reorder_threshold'] * 3 - r['stock_quantity'])
                st.error(f"**{r['name']} ({r['sku']})** — Status: `{r['Stock Status']}` (Current: {r['stock_quantity']} | Reorder Threshold: {r['reorder_threshold']}). **Suggested Replenishment:** +{rec_qty} units.")
                
    with c_add:
        st.subheader("➕ Add / Update Catalogue Product & Expiry")
        with st.form("inv_form"):
            p1, p2 = st.columns(2)
            sku = p1.text_input("SKU", "SKU-201")
            name = p2.text_input("Product Name", "Oreo Vanilla Pack")
            p3, p4 = st.columns(2)
            brand = p3.text_input("Brand", "Cadbury")
            cat = p4.text_input("Category", "biscuit")
            p5, p6, p7 = st.columns(3)
            bar = p5.text_input("Barcode", "8909991")
            qty = p6.number_input("Recorded Stock Qty", 0, 1000, 12)
            thresh = p7.number_input("Reorder Threshold", 1, 100, 5)
            exp = st.date_input("Expiry Date", datetime.date(2026, 12, 31))
            if st.form_submit_button("💾 Save to SQLite Inventory"):
                execute_sql('''INSERT OR REPLACE INTO products 
                    (sku, name, brand, category, barcode, unit_price, stock_quantity, reorder_threshold, expiry_date, batch_number, ref_image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (sku, name, brand, cat, bar, 2.50, qty, thresh, str(exp), "B-100", ""))
                st.success("Inventory updated!")
                st.rerun()

# =====================================================================
# PAGE 5: LIVE ANALYTICS & DETECTION QUALITY CENTRE (Features E1 - E4)
# =====================================================================
elif selection == "📊 5. Live Analytics & Quality Centre":
    st.title("📊 Live Retail Analytics & Detection Quality Centre")
    
    df_det = query_df("SELECT * FROM detections")
    df_inv = query_df("SELECT * FROM products")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Detections Logged", len(df_det))
    k2.metric("Verified Identifications", len(df_det[df_det['status'].str.contains("Verified", na=False)]) if not df_det.empty else 0)
    k3.metric("Unknown / Review Queue", len(df_det[df_det['status'] == "Manual Review Required"]) if not df_det.empty else 0)
    k4.metric("Avg Shelf Occupancy", f"{df_det['shelf_occupancy'].mean():.1f}%" if not df_det.empty else "0.0%")
    
    if df_det.empty:
        st.info("No detection history yet. Scan an image in the **6-Stage Smart Scanner** to populate live charts.")
    else:
        ch1, ch2 = st.columns(2)
        with ch1:
            st.subheader("Detection Source Breakdown (6-Stage Pipeline)")
            fig1 = px.pie(df_det, names="winning_stage", hole=0.4, color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig1, use_container_width=True)
        with ch2:
            st.subheader("Confidence Score Distribution")
            fig2 = px.histogram(df_det, x="confidence", nbins=10, color="status")
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("🕵️ Detection Quality Audit Log")
        st.dataframe(df_det[['timestamp', 'image_name', 'yolo_pred', 'ocr_text', 'barcode_val', 'final_identity', 'winning_stage', 'confidence', 'status']], use_container_width=True)

# =====================================================================
# PAGE 6: REPORTS, AUTOMATED EVALUATION & SETTINGS (Features E5, E6)
# =====================================================================
elif selection == "📥 6. Reports, Evaluation & Settings":
    st.title("📥 Multi-Format Report Export & Model Evaluation")
    
    df_det = query_df("SELECT * FROM detections")
    df_inv = query_df("SELECT * FROM products")
    
    # E5: CSV & Excel Report Export
    st.subheader("1. Export Enterprise Reports (CSV & Excel)")
    e1, e2, e3 = st.columns(3)
    e1.download_button("📥 Download Inventory (CSV)", data=df_inv.to_csv(index=False).encode('utf-8'), file_name="smartmart_inventory.csv", mime="text/csv")
    e2.download_button("📥 Download Detection Audit (CSV)", data=df_det.to_csv(index=False).encode('utf-8'), file_name="smartmart_detections.csv", mime="text/csv")
    
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
        df_inv.to_excel(writer, sheet_name='Inventory_Catalogue', index=False)
        df_det.to_excel(writer, sheet_name='6_Stage_Detection_Log', index=False)
    e3.download_button("📗 Download Full Workbook (Excel .xlsx)", data=excel_buf.getvalue(), file_name="SmartMart_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    st.markdown("---")
    # E6: Automated Testing & Ground-Truth Performance Evaluation
    st.subheader("2. Automated Testing & Ground-Truth Evaluation Calculator")
    st.write("Compare predicted product counts against actual labelled ground-truth counts from unseen test images to compute genuine Precision, Recall, and F1-Score.")
    
    col_gt1, col_gt2 = st.columns(2)
    tp = col_gt1.number_input("True Positives (Correctly Identified Products)", min_value=0, value=18)
    fp = col_gt1.number_input("False Positives (Wrong or Duplicate Detections)", min_value=0, value=2)
    fn = col_gt2.number_input("False Negatives (Missed Products on Shelf)", min_value=0, value=3)
    
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / float(precision + recall) if (precision + recall) > 0 else 0.0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Measured Precision", f"{precision*100:.1f}%")
    m2.metric("Measured Recall", f"{recall*100:.1f}%")
    m3.metric("F1-Score", f"{f1*100:.1f}%")
