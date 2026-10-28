import os
import io
import re
import sqlite3
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

# Optional components: the app still opens if these are not installed.
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import easyocr
except Exception:
    easyocr = None

try:
    from pyzbar.pyzbar import decode as decode_barcodes
except Exception:
    decode_barcodes = None


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(page_title="SmartMart AI", page_icon="🛒", layout="wide")

DB_PATH = os.getenv("SMARTMART_DB", "smartmart_ai.db")
REFERENCE_DIR = Path("reference_catalogue")
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROMPTS = (
    "bottle, can, carton, box, packet, bag, jar, cup, fruit, apple, banana, "
    "orange, snack packet, chocolate bar, biscuit packet, cereal box, "
    "milk carton, shampoo bottle, soap bar, toothpaste box, chips packet, "
    "detergent bottle, medicine box, food container"
)

st.markdown("""
<style>
.stApp { background: #07111e; color: #f1f5f9; }
[data-testid="stSidebar"] { background: #0b1727; }
.hero {
  padding: 1.35rem 1.5rem; border-radius: 16px; margin-bottom: 1rem;
  background: linear-gradient(125deg,#0a192f,#064e3b);
  border: 1px solid #22c55e;
}
.card {
  padding: 1rem; border-radius: 12px; background: #102138;
  border: 1px solid #245044; margin-bottom: .75rem;
}
.small-note { color: #cbd5e1; font-size: .92rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------
def connect_db():
    return sqlite3.connect(DB_PATH, timeout=20)

def init_db():
    with connect_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            brand TEXT DEFAULT '',
            category TEXT DEFAULT '',
            barcode TEXT UNIQUE,
            unit_price REAL DEFAULT 0,
            stock_quantity INTEGER DEFAULT 0,
            reorder_threshold INTEGER DEFAULT 5,
            expiry_date TEXT DEFAULT '',
            reference_image TEXT DEFAULT ''
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT,
            detected_label TEXT,
            identified_product TEXT,
            identity_method TEXT,
            confidence REAL,
            review_status TEXT,
            barcode TEXT,
            ocr_text TEXT,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
            box_area_pct REAL
        )
        """)
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if count == 0:
            sample = [
                ("SKU-101", "Snickers Chocolate Bar", "Snickers", "Chocolate", "890100100001", 1.50, 24, 5, "2027-04-10", ""),
                ("SKU-102", "Coca-Cola Soda Can", "Coca-Cola", "Beverage", "890100100002", 1.20, 4, 8, "2027-08-02", ""),
                ("SKU-103", "Chocolate Wafer Packet", "Nestlé", "Snack", "890100100003", 2.50, 3, 5, "2027-03-01", ""),
                ("SKU-104", "Sanitizer Bottle", "Example Brand", "Personal Care", "890100100004", 1.99, 14, 5, "2028-01-05", ""),
                ("SKU-105", "Red Apple", "Fresh Produce", "Fruit", "890100100005", 0.80, 2, 10, "2026-11-15", ""),
                ("SKU-106", "Banana", "Fresh Produce", "Fruit", "890100100006", 1.10, 0, 6, "2026-10-31", ""),
            ]
            conn.executemany("""
                INSERT INTO products
                (sku,name,brand,category,barcode,unit_price,stock_quantity,reorder_threshold,expiry_date,reference_image)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, sample)

def read_df(sql, params=()):
    with connect_db() as conn:
        return pd.read_sql_query(sql, conn, params=params)

def run_sql(sql, params=()):
    with connect_db() as conn:
        conn.execute(sql, params)

init_db()


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_detector():
    """YOLO-World is an open-vocabulary detector; weights download on first use."""
    if YOLO is None:
        return None, "Ultralytics is not installed."
    try:
        model = YOLO("yolov8s-worldv2.pt")
        return model, "YOLO-World loaded."
    except Exception as exc:
        return None, f"Could not load YOLO-World: {exc}"

@st.cache_resource(show_spinner=False)
def load_ocr():
    if easyocr is None:
        return None, "EasyOCR is not installed."
    try:
        return easyocr.Reader(["en"], gpu=False), "EasyOCR loaded."
    except Exception as exc:
        return None, f"Could not load EasyOCR: {exc}"

detector, detector_message = load_detector()
ocr_reader, ocr_message = load_ocr()


# ---------------------------------------------------------------------
# IMAGE / IDENTIFICATION HELPERS
# ---------------------------------------------------------------------
def pil_to_bgr(pil_image):
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

def safe_crop(bgr, box):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return bgr[0:0, 0:0]
    return bgr[y1:y2, x1:x2]

def ocr_text(crop):
    if ocr_reader is None or crop is None or crop.size == 0:
        return ""
    try:
        # EasyOCR expects RGB images.
        return " ".join(ocr_reader.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), detail=0)).strip()
    except Exception:
        return ""

def barcode_values(image_bgr):
    if decode_barcodes is None:
        return []
    try:
        return [item.data.decode("utf-8", errors="ignore").strip()
                for item in decode_barcodes(image_bgr)]
    except Exception:
        return []

def product_by_barcode(value, products):
    if not value:
        return None
    matched = products[products["barcode"].fillna("").astype(str) == str(value)]
    return None if matched.empty else matched.iloc[0].to_dict()

def identify_from_text(text, products):
    """Only returns an exact catalogue match when the OCR contains catalogue terms."""
    cleaned = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    if not cleaned.strip():
        return None
    candidates = []
    for _, row in products.iterrows():
        terms = [str(row.get("name", "")), str(row.get("brand", "")), str(row.get("sku", ""))]
        for term in terms:
            term = re.sub(r"[^a-z0-9 ]", " ", term.lower()).strip()
            if len(term) >= 4 and term in cleaned:
                candidates.append((len(term), row.to_dict()))
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1] if candidates else None

def set_world_prompts(model, prompts):
    if model is None:
        return
    try:
        labels = [x.strip() for x in prompts.split(",") if x.strip()]
        if labels:
            model.set_classes(labels)
    except Exception:
        pass

def detect_objects(pil_image, prompts, conf_threshold=0.25):
    """Return open-vocabulary object boxes and their prompted labels."""
    if detector is None:
        return [], "Detector unavailable. Install dependencies and check model download."
    try:
        set_world_prompts(detector, prompts)
        result = detector(pil_image.convert("RGB"), conf=conf_threshold, verbose=False)[0]
        detections = []
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                coords = [int(v) for v in box.xyxy[0].tolist()]
                class_id = int(box.cls[0])
                detections.append({
                    "bbox": coords,
                    "label": str(names.get(class_id, f"class_{class_id}")),
                    "confidence": float(box.conf[0]),
                })
        return detections, None
    except Exception as exc:
        return [], f"Detection failed: {exc}"

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

def remove_duplicate_boxes(detections, threshold=0.55):
    kept = []
    for item in sorted(detections, key=lambda x: x["confidence"], reverse=True):
        if all(iou(item["bbox"], other["bbox"]) < threshold for other in kept):
            kept.append(item)
    return kept

def run_scan(pil_image, image_name, prompts, conf_threshold):
    products = read_df("SELECT * FROM products")
    bgr = pil_to_bgr(pil_image)
    raw_dets, error = detect_objects(pil_image, prompts, conf_threshold)
    detections = remove_duplicate_boxes(raw_dets)
    full_image_barcodes = barcode_values(bgr)
    annotated = bgr.copy()
    records = []
    h_img, w_img = bgr.shape[:2]
    total_box_area = 0

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        crop = safe_crop(bgr, det["bbox"])
        total_box_area += max(0, x2-x1) * max(0, y2-y1)
        text = ocr_text(crop)

        # Barcode decoding on the crop first, then whole image if there is one barcode.
        barcodes = barcode_values(crop)
        if not barcodes and len(full_image_barcodes) == 1:
            barcodes = full_image_barcodes
        barcode = barcodes[0] if barcodes else ""

        matched = product_by_barcode(barcode, products)
        method = "Barcode → catalogue" if matched else ""
        if matched is None:
            matched = identify_from_text(text, products)
            if matched:
                method = "OCR → catalogue text match"

        if matched:
            identity = str(matched["name"])
            review_status = "Catalogue match"
            identity_confidence = 0.99 if method.startswith("Barcode") else min(0.90, max(0.50, det["confidence"]))
        else:
            # Do not invent an exact brand/SKU from a generic object label.
            identity = f"Unverified: {det['label']}"
            review_status = "Needs manual identification"
            method = "Open-vocabulary object detection only"
            identity_confidence = det["confidence"]

        color = (25, 220, 120) if matched else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{identity[:38]} | {det['confidence']:.2f}"
        cv2.putText(annotated, label, (x1, max(20, y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        record = {
            "Detection #": idx,
            "Detected category": det["label"],
            "Product name": identity,
            "Identification method": method,
            "Detector confidence": round(det["confidence"], 3),
            "Identity confidence (estimate)": round(identity_confidence, 3),
            "Review status": review_status,
            "Barcode": barcode,
            "OCR text": text,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Box area %": round(100 * max(0, x2-x1) * max(0, y2-y1) / (w_img*h_img), 2),
            "_crop": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) if crop.size else pil_image
        }
        records.append(record)

        run_sql("""
            INSERT INTO detections
            (timestamp,image_name,detected_label,identified_product,identity_method,confidence,
             review_status,barcode,ocr_text,x1,y1,x2,y2,box_area_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            dt.datetime.now().isoformat(timespec="seconds"), image_name, det["label"], identity,
            method, float(det["confidence"]), review_status, barcode, text,
            x1, y1, x2, y2, record["Box area %"]
        ))

    occupancy = round(min(100.0, total_box_area / (w_img * h_img + 1e-9) * 100), 2)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    return annotated_pil, records, occupancy, error


# ---------------------------------------------------------------------
# SIDEBAR / NAVIGATION
# ---------------------------------------------------------------------
PAGES = [
    "🏠 Overview",
    "📷 Smart Scanner",
    "🔎 Manual Review",
    "📦 Inventory Catalogue",
    "📊 Analytics & Graph Descriptions",
    "📥 Reports & Evaluation",
]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

st.sidebar.title("🛒 SmartMart AI")
st.sidebar.caption("Open-vocabulary retail vision + inventory tools")
st.session_state.page = st.sidebar.radio("Navigate", PAGES, index=PAGES.index(st.session_state.page))

# ---------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------------------
if st.session_state.page == PAGES[0]:
    st.markdown("""
    <div class="hero">
      <h1>🛒 SmartMart AI</h1>
      <h3>Product Detection, Catalogue Matching & Retail Analytics</h3>
      <p>Detects prompted object categories, reads visible packaging text and barcodes,
      matches known products to your catalogue, and sends uncertain identities for review.</p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("Open-vocabulary detector", "Ready" if detector else "Unavailable")
    b.metric("OCR", "Ready" if ocr_reader else "Unavailable")
    c.metric("Barcode reader", "Ready" if decode_barcodes else "Unavailable")
    products = read_df("SELECT * FROM products")
    d.metric("Catalogue products", len(products))

    st.info(
        "Important limitation: no single model can reliably detect and name every retail SKU worldwide "
        "from every image. YOLO-World detects categories supplied as prompts. Exact brand/product matching "
        "requires readable packaging text, a readable barcode connected to the catalogue, or a manually "
        "registered reference. Uncertain items are deliberately marked for review."
    )
    st.markdown("### How the pipeline works")
    st.markdown("""
    1. **Open-vocabulary detection:** finds objects matching the category prompts.
    2. **Duplicate suppression:** removes heavily overlapping boxes.
    3. **Barcode lookup:** matches a decoded barcode to the local catalogue.
    4. **OCR matching:** checks readable packaging text against catalogue names and brands.
    5. **Manual review:** leaves unmatched items unverified instead of guessing a brand.
    6. **Logging and analytics:** saves detections, methods, confidence and box coordinates.
    """)
    st.markdown("### Quick start")
    st.write("Open **Smart Scanner**, upload a clear shelf/tabletop image, adjust the prompts to include likely object categories, and click **Run scan**.")


# ---------------------------------------------------------------------
# PAGE 2: SCANNER
# ---------------------------------------------------------------------
elif st.session_state.page == PAGES[1]:
    st.title("📷 Smart Scanner")
    st.caption("For best results, use sharp, well-lit images with products not excessively occluded.")

    with st.expander("Detection settings", expanded=True):
        prompts = st.text_area("Object/category prompts (comma-separated)", DEFAULT_PROMPTS, height=100,
                               help="YOLO-World searches for these labels. Add category names relevant to your image.")
        conf = st.slider("Detection confidence threshold", 0.05, 0.80, 0.20, 0.05)
        st.caption("Lower thresholds may find more objects but also produce more false positives.")

    input_mode = st.radio("Image source", ["Upload image(s)", "Camera"], horizontal=True)
    files = []
    if input_mode == "Upload image(s)":
        uploads = st.file_uploader("Choose JPG, JPEG, PNG or WEBP files", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)
        if uploads:
            files = uploads
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            files = [cam]

    if files and st.button("🚀 Run scan", type="primary"):
        st.session_state.scan_results = []
        for file in files:
            try:
                image = Image.open(file).convert("RGB")
                with st.spinner(f"Scanning {getattr(file, 'name', 'camera image')}..."):
                    annotated, records, occupancy, error = run_scan(
                        image, getattr(file, "name", "camera_image.jpg"), prompts, conf
                    )
                st.session_state.scan_results.append({
                    "filename": getattr(file, "name", "camera_image.jpg"),
                    "image": annotated,
                    "records": records,
                    "occupancy": occupancy,
                    "error": error,
                })
            except Exception as exc:
                st.error(f"Could not process image: {exc}")

    for result in st.session_state.get("scan_results", []):
        st.markdown(f"### Results: `{result['filename']}`")
        if result["error"]:
            st.warning(result["error"])
        left, right = st.columns([2,1])
        with left:
            st.image(result["image"], use_container_width=True, caption="Green = catalogue match; orange = unverified identity")
        with right:
            st.metric("Detected object boxes", len(result["records"]))
            st.metric("Estimated box coverage", f"{result['occupancy']}%")
            st.caption("Box coverage is the sum of bounding-box areas, not true shelf stock or empty-space measurement.")
        if result["records"]:
            public_rows = [{k:v for k,v in row.items() if not k.startswith("_")} for row in result["records"]]
            st.dataframe(pd.DataFrame(public_rows), use_container_width=True)
            csv = pd.DataFrame(public_rows).to_csv(index=False).encode("utf-8")
            st.download_button("Download scan results CSV", csv, file_name="smartmart_scan_results.csv", mime="text/csv", key=f"csv_{result['filename']}")
        else:
            st.warning("No objects were detected. Try adding more relevant prompts or lowering the confidence threshold.")


# ---------------------------------------------------------------------
# PAGE 3: MANUAL REVIEW
# ---------------------------------------------------------------------
elif st.session_state.page == PAGES[2]:
    st.title("🔎 Manual Review & Product Registration")
    st.write("Register an unverified object using its actual name and SKU. This adds it to the catalogue; it does not train the detector.")
    scan_results = st.session_state.get("scan_results", [])
    review_items = []
    for result in scan_results:
        for record in result["records"]:
            if record["Review status"] != "Catalogue match":
                review_items.append((result["filename"], record))

    if not review_items:
        st.info("No unverified detections from the current session. Run a scan first, or add a product directly from Inventory Catalogue.")
    else:
        for idx, (filename, record) in enumerate(review_items):
            with st.container(border=True):
                col1, col2 = st.columns([1,3])
                with col1:
                    st.image(record["_crop"], use_container_width=True)
                with col2:
                    st.write(f"**Detected:** {record['Detected category']} · **Source:** {filename}")
                    with st.form(f"review_form_{idx}"):
                        p_name = st.text_input("Verified product name", key=f"review_name_{idx}")
                        p_brand = st.text_input("Brand (if known)", key=f"review_brand_{idx}")
                        p_sku = st.text_input("SKU (unique)", value=f"SKU-{dt.datetime.now().strftime('%y%m%d%H%M%S')}-{idx}", key=f"review_sku_{idx}")
                        p_category = st.text_input("Category", value=record["Detected category"], key=f"review_cat_{idx}")
                        p_barcode = st.text_input("Barcode (optional)", key=f"review_bar_{idx}")
                        if st.form_submit_button("Save verified product"):
                            if not p_name.strip() or not p_sku.strip():
                                st.error("Product name and SKU are required.")
                            else:
                                ref_path = str(REFERENCE_DIR / f"{re.sub(r'[^A-Za-z0-9_-]','_',p_sku)}.png")
                                record["_crop"].save(ref_path)
                                try:
                                    run_sql("""
                                    INSERT INTO products
                                    (sku,name,brand,category,barcode,unit_price,stock_quantity,reorder_threshold,expiry_date,reference_image)
                                    VALUES (?,?,?,?,?,?,?,?,?,?)
                                    """, (p_sku.strip(), p_name.strip(), p_brand.strip(), p_category.strip(),
                                          p_barcode.strip() or None, 0.0, 0, 5, "", ref_path))
                                    st.success(f"Added {p_name} to the catalogue.")
                                    st.rerun()
                                except sqlite3.IntegrityError:
                                    st.error("That SKU or barcode already exists. Use a unique value.")


# ---------------------------------------------------------------------
# PAGE 4: INVENTORY CATALOGUE
# ---------------------------------------------------------------------
elif st.session_state.page == PAGES[3]:
    st.title("📦 Inventory Catalogue")
    df = read_df("SELECT * FROM products ORDER BY name")
    if not df.empty:
        low = df[df["stock_quantity"] <= df["reorder_threshold"]]
        expiring = df[(df["expiry_date"].fillna("") != "") & (df["expiry_date"] <= (dt.date.today()+dt.timedelta(days=30)).isoformat())]
        c1, c2, c3 = st.columns(3)
        c1.metric("Catalogue items", len(df))
        c2.metric("At/below reorder threshold", len(low))
        c3.metric("Expired/expiring within 30 days", len(expiring))
        st.dataframe(df.drop(columns=["reference_image"]), use_container_width=True)
        st.subheader("Add a product")
        with st.form("add_product_form"):
            a,b,c = st.columns(3)
            sku = a.text_input("SKU *")
            name = b.text_input("Product name *")
            brand = c.text_input("Brand")
            d,e,f = st.columns(3)
            category = d.text_input("Category")
            barcode = e.text_input("Barcode (optional)")
            price = f.number_input("Unit price", min_value=0.0, value=0.0, step=0.1)
            g,h,i = st.columns(3)
            stock = g.number_input("Stock quantity", min_value=0, value=0, step=1)
            reorder = h.number_input("Reorder threshold", min_value=0, value=5, step=1)
            expiry = i.date_input("Expiry date (optional)", value=None)
            submitted = st.form_submit_button("Add product")
            if submitted:
                if not sku.strip() or not name.strip():
                    st.error("SKU and product name are required.")
                else:
                    expiry_text = expiry.isoformat() if expiry else ""
                    try:
                        run_sql("""INSERT INTO products
                        (sku,name,brand,category,barcode,unit_price,stock_quantity,reorder_threshold,expiry_date,reference_image)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (sku.strip(),name.strip(),brand.strip(),category.strip(),barcode.strip() or None,
                         price,int(stock),int(reorder),expiry_text,""))
                        st.success("Product added.")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("SKU or barcode already exists.")
        st.subheader("Update stock")
        options = {f"{r['sku']} — {r['name']}": int(r["product_id"]) for _,r in df.iterrows()}
        with st.form("stock_form"):
            chosen = st.selectbox("Product", list(options.keys()))
            new_stock = st.number_input("New stock quantity", min_value=0, value=0, step=1)
            update_stock = st.form_submit_button("Update quantity")
            if update_stock:
                run_sql("UPDATE products SET stock_quantity=? WHERE product_id=?", (int(new_stock), options[chosen]))
                st.success("Stock updated.")
                st.rerun()
        if not low.empty:
            st.warning("Restock review: " + ", ".join(low["name"].astype(str).tolist()))
        if not expiring.empty:
            st.warning("Expiry review: " + ", ".join(expiring["name"].astype(str).tolist()))
    else:
        st.info("Catalogue is empty.")


# ---------------------------------------------------------------------
# PAGE 5: ANALYTICS + EXPLANATIONS FOR EVERY GRAPH
# ---------------------------------------------------------------------
elif st.session_state.page == PAGES[4]:
    st.title("📊 Analytics & Graph Descriptions")
    df = read_df("SELECT * FROM detections ORDER BY timestamp DESC")
    products = read_df("SELECT * FROM products")
    if df.empty:
        st.info("No scan data yet. Run a scan to populate the analytics.")
    else:
        total = len(df)
        matched = int((df["review_status"] == "Catalogue match").sum())
        needs_review = int((df["review_status"] != "Catalogue match").sum())
        avg_conf = float(df["confidence"].mean()) if total else 0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Detection records", total)
        c2.metric("Catalogue matches", matched)
        c3.metric("Needs review", needs_review)
        c4.metric("Mean detector confidence", f"{avg_conf:.2f}")

        st.subheader("1. Detections by object label")
        label_counts = df["detected_label"].fillna("Unknown").value_counts().rename_axis("Object label").reset_index(name="Count")
        fig1 = px.bar(label_counts, x="Object label", y="Count", title="How many detected boxes were assigned to each prompted label?")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Description: each bar represents the number of logged bounding boxes for a detected category across all scans. It counts detections, not unique physical products or confirmed inventory.")

        st.subheader("2. Identification outcomes")
        outcome = df["review_status"].fillna("Unknown").value_counts().rename_axis("Outcome").reset_index(name="Count")
        fig2 = px.pie(outcome, names="Outcome", values="Count", hole=0.45, title="Catalogue matches vs. items needing manual review")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Description: the chart shows the share of detections matched to the local product catalogue versus detections that remain unverified. A high review share means OCR/barcodes/catalogue coverage or image quality may need improvement.")

        st.subheader("3. Identification method usage")
        method_counts = df["identity_method"].fillna("Not recorded").value_counts().rename_axis("Method").reset_index(name="Count")
        fig3 = px.bar(method_counts, x="Method", y="Count", title="Which recognition route produced each identity?")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Description: compares how often the system relied on barcode matching, OCR-to-catalogue matching, or object detection alone. Object detection alone is not an exact brand/SKU identification.")

        st.subheader("4. Confidence distribution")
        fig4 = px.histogram(df, x="confidence", nbins=10, title="Distribution of detector confidence scores")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Description: groups logged detector confidence values into ranges. Confidence is the model's detection score, not a calibrated probability that the product's exact name is correct.")

        st.subheader("5. Inventory stock vs. reorder threshold")
        if not products.empty:
            inv = products[["name","stock_quantity","reorder_threshold"]].melt(
                id_vars="name", var_name="Measure", value_name="Units"
            )
            fig5 = px.bar(inv, x="name", y="Units", color="Measure", barmode="group",
                          title="Current catalogue stock compared with reorder thresholds")
            st.plotly_chart(fig5, use_container_width=True)
            st.caption("Description: compares manually maintained stock quantity with the reorder threshold for each catalogue product. Shelf detections do not automatically update inventory quantities.")
        st.subheader("Recent detection log")
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 6: REPORTS / EVALUATION
# ---------------------------------------------------------------------
elif st.session_state.page == PAGES[5]:
    st.title("📥 Reports & Evaluation")
    inv = read_df("SELECT * FROM products ORDER BY name")
    det = read_df("SELECT * FROM detections ORDER BY timestamp DESC")
    c1,c2 = st.columns(2)
    c1.download_button("Download inventory CSV", inv.to_csv(index=False).encode("utf-8"),
                       "smartmart_inventory.csv", "text/csv")
    c2.download_button("Download detection log CSV", det.to_csv(index=False).encode("utf-8"),
                       "smartmart_detection_log.csv", "text/csv")

    st.subheader("Evaluation checklist")
    st.markdown("""
    - Test on images not used while tuning prompts.
    - Manually label a small test set with object boxes and exact product names.
    - Calculate detection precision/recall separately from exact product-name accuracy.
    - Record false positives, missed objects, duplicate boxes, barcode failures and OCR failures.
    - Test crowded shelves, reflective packaging, small products, low light and occlusion.
    - Do not describe this as universal or 100% accurate without measured evidence.
    """)
    st.warning("A detection log is not a ground-truth evaluation. To report precision/recall or accuracy, compare results with manually verified labels.")
    st.caption(f"Database: {DB_PATH} · Detector status: {detector_message} · OCR status: {ocr_message}")
