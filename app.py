
import os
import re
import sqlite3
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image, ImageDraw

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


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartMart AI",
    page_icon="🛒",
    layout="wide"
)

DB_PATH = os.getenv("SMARTMART_DB", "smartmart_ai.db")
REFERENCE_DIR = Path("reference_catalogue")
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROMPTS = (
    "snack package, snack bag, chips bag, potato chip packet, "
    "candy bar, chocolate bar, chocolate wrapper, candy package, "
    "biscuit packet, cookie package, food package, packaged food, "
    "cereal box, cardboard box, plastic bottle, soda can, "
    "aluminum beverage can, beverage container, toothpaste tube, "
    "toothbrush, soap bar, shampoo bottle, carton, jar, bag, box"
)

st.markdown("""
<style>
.stApp {
    background: #07111e;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #0b1727;
}
.hero {
    padding: 1.35rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    background: linear-gradient(125deg,#0a192f,#064e3b);
    border: 1px solid #22c55e;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================

def connect_db():
    return sqlite3.connect(DB_PATH, timeout=30)


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
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            box_area_pct REAL
        )
        """)

        count = conn.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0]

        if count == 0:
            sample = [
                ("SKU-101", "Snickers Chocolate Bar",
                 "Snickers", "Chocolate", None, 1.50, 24, 5,
                 "2027-04-10", ""),
                ("SKU-102", "Coca-Cola Soda Can",
                 "Coca-Cola", "Beverage", None, 1.20, 4, 8,
                 "2027-08-02", ""),
                ("SKU-103", "Chocolate Wafer Packet",
                 "Nestle", "Snack", None, 2.50, 3, 5,
                 "2027-03-01", ""),
                ("SKU-104", "Sanitizer Bottle",
                 "Example Brand", "Personal Care", None,
                 1.99, 14, 5, "2028-01-05", ""),
                ("SKU-105", "Red Apple",
                 "Fresh Produce", "Fruit", None,
                 0.80, 2, 10, "2026-11-15", ""),
                ("SKU-106", "Banana",
                 "Fresh Produce", "Fruit", None,
                 1.10, 0, 6, "2026-10-31", "")
            ]

            conn.executemany("""
            INSERT INTO products
            (sku,name,brand,category,barcode,unit_price,
             stock_quantity,reorder_threshold,expiry_date,
             reference_image)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """, sample)


def read_df(sql, params=()):
    with connect_db() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def run_sql(sql, params=()):
    with connect_db() as conn:
        conn.execute(sql, params)


init_db()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_detector():
    if YOLO is None:
        return None, "Ultralytics is not installed."

    try:
        model = YOLO("yolov8s-worldv2.pt")
        return model, "YOLO-World loaded successfully."
    except Exception as exc:
        return None, f"Detector loading failed: {exc}"


@st.cache_resource(show_spinner=False)
def load_ocr():
    if easyocr is None:
        return None, "EasyOCR is not installed."

    try:
        reader = easyocr.Reader(["en"], gpu=False)
        return reader, "EasyOCR loaded successfully."
    except Exception as exc:
        return None, f"OCR loading failed: {exc}"


detector, detector_message = load_detector()
ocr_reader, ocr_message = load_ocr()


# ============================================================
# IMAGE HELPERS
# ============================================================

def pil_to_bgr(image):
    return cv2.cvtColor(
        np.array(image.convert("RGB")),
        cv2.COLOR_RGB2BGR
    )


def safe_crop(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return image[y1:y2, x1:x2]


def ocr_text(crop):
    if ocr_reader is None or crop is None or crop.size == 0:
        return ""

    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = ocr_reader.readtext(rgb, detail=0)
        return " ".join(results).strip()
    except Exception:
        return ""


def barcode_values(image):
    if decode_barcodes is None:
        return []

    try:
        values = []

        for item in decode_barcodes(image):
            value = item.data.decode(
                "utf-8", errors="ignore"
            ).strip()

            if value and value not in values:
                values.append(value)

        return values
    except Exception:
        return []


# ============================================================
# PRODUCT IDENTIFICATION
# ============================================================

def product_by_barcode(value, products):
    if not value or products.empty:
        return None

    matched = products[
        products["barcode"].fillna("").astype(str)
        == str(value)
    ]

    return None if matched.empty else matched.iloc[0].to_dict()


def identify_from_text(text, products):
    if not text or products.empty:
        return None

    cleaned = re.sub(
        r"[^a-z0-9 ]", " ", str(text).lower()
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return None

    candidates = []

    for _, row in products.iterrows():
        terms = [
            str(row.get("name", "")),
            str(row.get("brand", "")),
            str(row.get("sku", ""))
        ]

        for term in terms:
            term = re.sub(
                r"[^a-z0-9 ]", " ", term.lower()
            )
            term = re.sub(r"\s+", " ", term).strip()

            if len(term) >= 4 and term in cleaned:
                candidates.append(
                    (len(term), row.to_dict())
                )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


# ============================================================
# OPEN-VOCABULARY DETECTION
# ============================================================

def set_world_prompts(model, prompts):
    labels = [
        item.strip()
        for item in prompts.split(",")
        if item.strip()
    ]

    if not labels:
        raise ValueError("Enter at least one detection prompt.")

    model.set_classes(labels)


def generate_tiles(image, tile_size=768, overlap=0.25):
    """
    Return overlapping tiles and their original-image offsets.
    Small objects may become easier to detect in each tile.
    """
    width, height = image.size

    if width <= tile_size and height <= tile_size:
        return [(image, 0, 0)]

    step = max(1, int(tile_size * (1 - overlap)))
    tiles = []

    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))

    final_x = max(0, width - tile_size)
    final_y = max(0, height - tile_size)

    if not xs or xs[-1] != final_x:
        xs.append(final_x)

    if not ys or ys[-1] != final_y:
        ys.append(final_y)

    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)

            crop = image.crop((x, y, x2, y2))
            tiles.append((crop, x, y))

    return tiles


def detect_objects(
    pil_image,
    prompts,
    conf_threshold=0.15,
    use_tiles=True,
    tile_size=768,
    imgsz=1024
):
    if detector is None:
        return [], "Detector unavailable. Check model installation."

    try:
        image = pil_image.convert("RGB")
        set_world_prompts(detector, prompts)

        if use_tiles:
            tiles = generate_tiles(image, tile_size, 0.25)
        else:
            tiles = [(image, 0, 0)]

        detections = []

        for tile, offset_x, offset_y in tiles:
            result = detector.predict(
                source=tile,
                conf=conf_threshold,
                imgsz=imgsz,
                iou=0.50,
                max_det=100,
                verbose=False
            )[0]

            if result.boxes is None:
                continue

            names = result.names

            for box in result.boxes:
                coords = [
                    int(v) for v in box.xyxy[0].tolist()
                ]

                # Convert tile coordinates to original coordinates.
                x1, y1, x2, y2 = coords
                x1 += offset_x
                x2 += offset_x
                y1 += offset_y
                y2 += offset_y

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": str(
                        names.get(
                            int(box.cls[0]),
                            "unknown object"
                        )
                    ),
                    "confidence": float(box.conf[0])
                })

        return detections, None

    except Exception as exc:
        return [], f"Detection failed: {exc}"


# ============================================================
# DUPLICATE SUPPRESSION
# ============================================================

def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0, x2 - x1) * max(0, y2 - y1)
    )

    area_a = (
        max(0, box_a[2] - box_a[0])
        * max(0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0, box_b[2] - box_b[0])
        * max(0, box_b[3] - box_b[1])
    )

    return intersection / (
        area_a + area_b - intersection + 1e-9
    )


def remove_duplicate_boxes(detections, threshold=0.65):
    """
    Suppress duplicate detections from overlapping tiles.
    Keep the highest-confidence overlapping candidate.
    """
    kept = []

    for item in sorted(
        detections,
        key=lambda obj: obj["confidence"],
        reverse=True
    ):
        duplicate = False

        for other in kept:
            overlap = iou(item["bbox"], other["bbox"])

            # Strong overlap usually indicates the same object.
            if overlap >= threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(item)

    return kept


# ============================================================
# COMPLETE SCAN PIPELINE
# ============================================================

def run_scan(
    pil_image,
    image_name,
    prompts,
    conf_threshold,
    use_tiles=True,
    tile_size=768,
    imgsz=1024
):
    products = read_df("SELECT * FROM products")
    bgr = pil_to_bgr(pil_image)

    raw_dets, error = detect_objects(
        pil_image,
        prompts,
        conf_threshold,
        use_tiles,
        tile_size,
        imgsz
    )

    detections = remove_duplicate_boxes(raw_dets)

    full_barcodes = barcode_values(bgr)
    annotated = bgr.copy()

    records = []
    image_height, image_width = bgr.shape[:2]
    total_box_area = 0

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]

        # Protect against invalid coordinates.
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))

        box = [x1, y1, x2, y2]
        crop = safe_crop(bgr, box)

        area = (x2 - x1) * (y2 - y1)
        total_box_area += area

        text = ocr_text(crop)

        crop_barcodes = barcode_values(crop)
        barcodes = crop_barcodes or full_barcodes

        # Avoid assigning a single image barcode to every object
        # when several different barcodes are present.
        barcode = (
            barcodes[0]
            if len(barcodes) == 1
            else ""
        )

        matched = product_by_barcode(barcode, products)
        method = "Barcode → catalogue" if matched else ""

        if matched is None:
            matched = identify_from_text(text, products)

            if matched:
                method = "OCR → catalogue"

        if matched:
            identity = str(matched["name"])
            review_status = "Catalogue match"

            identity_confidence = (
                0.99
                if method.startswith("Barcode")
                else min(
                    0.90,
                    max(0.50, det["confidence"])
                )
            )
        else:
            identity = f"Unverified: {det['label']}"
            review_status = "Needs manual identification"
            method = "Object detection only"
            identity_confidence = det["confidence"]

        color = (
            (25, 220, 120)
            if matched
            else (0, 165, 255)
        )

        cv2.rectangle(
            annotated, (x1, y1), (x2, y2), color, 2
        )

        label = (
            f"{identity[:34]} "
            f"| {det['confidence']:.2f}"
        )

        text_y = max(20, y1 - 7)

        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2
        )

        record = {
            "Detection #": idx,
            "Detected category": det["label"],
            "Product name": identity,
            "Identification method": method,
            "Detector confidence": round(
                det["confidence"], 3
            ),
            "Identity confidence (estimate)": round(
                identity_confidence, 3
            ),
            "Review status": review_status,
            "Barcode": barcode,
            "OCR text": text,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "Box area %": round(
                100 * area / (image_width * image_height),
                2
            ),
            "_crop": Image.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )
        }

        records.append(record)

        run_sql("""
        INSERT INTO detections
        (timestamp,image_name,detected_label,identified_product,
         identity_method,confidence,review_status,barcode,ocr_text,
         x1,y1,x2,y2,box_area_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            dt.datetime.now().isoformat(timespec="seconds"),
            image_name,
            det["label"],
            identity,
            method,
            float(det["confidence"]),
            review_status,
            barcode,
            text,
            x1, y1, x2, y2,
            record["Box area %"]
        ))

    occupancy = round(
        min(
            100.0,
            100 * total_box_area
            / (image_width * image_height)
        ),
        2
    )

    annotated_image = Image.fromarray(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    )

    return annotated_image, records, occupancy, error


# ============================================================
# NAVIGATION
# ============================================================

PAGES = [
    "🏠 Overview",
    "📷 Smart Scanner",
    "🔎 Manual Review",
    "📦 Inventory Catalogue",
    "📊 Analytics",
    "📥 Reports"
]

if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

st.sidebar.title("🛒 SmartMart AI")
st.sidebar.caption("Retail detection and inventory management")

st.session_state.page = st.sidebar.radio(
    "Navigate",
    PAGES,
    index=PAGES.index(st.session_state.page)
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if st.session_state.page == PAGES[0]:

    st.markdown("""
    <div class="hero">
      <h1>🛒 SmartMart AI</h1>
      <h3>Product Detection & Inventory Intelligence</h3>
      <p>
      Detect retail objects, read visible text, scan barcodes,
      match catalogue products and review uncertain identities.
      </p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)

    a.metric(
        "Object detector",
        "Ready" if detector else "Unavailable"
    )

    b.metric(
        "OCR",
        "Ready" if ocr_reader else "Unavailable"
    )

    c.metric(
        "Barcode reader",
        "Ready" if decode_barcodes else "Unavailable"
    )

    products = read_df("SELECT * FROM products")
    d.metric("Catalogue products", len(products))

    st.info(
        "Detection is not guaranteed for every object. "
        "Exact product identification requires reliable OCR, "
        "a matching barcode, or manual verification."
    )

    st.write("### How the system works")

    st.markdown("""
    1. Detect objects using prompted retail descriptions.
    2. Scan overlapping image tiles to help find small products.
    3. Remove duplicate bounding boxes.
    4. Read packaging text using OCR.
    5. Match decoded barcodes against the catalogue.
    6. Send unverified products for manual review.
    7. Save detection logs for analytics and evaluation.
    """)

    st.caption(detector_message)
    st.caption(ocr_message)


# ============================================================
# PAGE 2 — SMART SCANNER
# ============================================================

elif st.session_state.page == PAGES[1]:

    st.title("📷 Smart Scanner")

    st.caption(
        "Use clear, well-lit images. Keep products visible "
        "and avoid excessive overlap."
    )

    with st.expander("Detection settings", expanded=True):

        prompts = st.text_area(
            "Object prompts (comma-separated)",
            DEFAULT_PROMPTS,
            height=110
        )

        conf = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.80,
            value=0.15,
            step=0.05
        )

        use_tiles = st.checkbox(
            "Use overlapping image tiles for small products",
            value=True
        )

        tile_size = st.select_slider(
            "Tile size",
            options=[512, 640, 768, 896],
            value=768
        )

        imgsz = st.select_slider(
            "Inference image size",
            options=[640, 768, 896, 1024],
            value=1024
        )

        st.caption(
            "Smaller tiles can enlarge small objects relative "
            "to the input. Larger inference sizes may use more "
            "memory and take longer."
        )

    input_mode = st.radio(
        "Image source",
        ["Upload image(s)", "Camera"],
        horizontal=True
    )

    files = []

    if input_mode == "Upload image(s)":

        uploads = st.file_uploader(
            "Choose product images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True
        )

        if uploads:
            files = uploads

    else:

        camera_image = st.camera_input("Take a photo")

        if camera_image:
            files = [camera_image]

    if files and st.button(
        "🚀 Run scan",
        type="primary"
    ):

        st.session_state.scan_results = []

        for file in files:

            filename = getattr(
                file, "name", "camera_image.jpg"
            )

            try:
                image = Image.open(file).convert("RGB")

                with st.spinner(f"Scanning {filename}..."):

                    annotated, records, occupancy, error = (
                        run_scan(
                            image,
                            filename,
                            prompts,
                            conf,
                            use_tiles,
                            tile_size,
                            imgsz
                        )
                    )

                st.session_state.scan_results.append({
                    "filename": filename,
                    "image": annotated,
                    "records": records,
                    "occupancy": occupancy,
                    "error": error
                })

            except Exception as exc:
                st.error(
                    f"Could not process {filename}: {exc}"
                )

    for result in st.session_state.get(
        "scan_results", []
    ):

        st.markdown(
            f"### Results: `{result['filename']}`"
        )

        if result["error"]:
            st.warning(result["error"])

        left, right = st.columns([2, 1])

        with left:
            st.image(
                result["image"],
                use_container_width=True,
                caption=(
                    "Green: catalogue match. "
                    "Orange: unverified identity."
                )
            )

        with right:
            st.metric(
                "Detected boxes",
                len(result["records"])
            )

            st.metric(
                "Estimated box coverage",
                f"{result['occupancy']}%"
            )

            st.caption(
                "Box coverage is not a measurement of "
                "stock quantity or empty shelf space."
            )

        if result["records"]:

            public_rows = [
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }
                for row in result["records"]
            ]

            df = pd.DataFrame(public_rows)

            st.dataframe(
                df,
                use_container_width=True
            )

            st.download_button(
                "Download scan CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="smartmart_scan_results.csv",
                mime="text/csv",
                key=f"csv_{result['filename']}"
            )

        else:
            st.warning(
                "No objects detected. Try clearer images, "
                "different prompts or a lower confidence "
                "threshold. Check the detector status above."
            )


# ============================================================
# PAGE 3 — MANUAL REVIEW
# ============================================================

elif st.session_state.page == PAGES[2]:

    st.title("🔎 Manual Review & Product Registration")

    st.write(
        "Verify an unrecognised object before registering "
        "its product details."
    )

    scan_results = st.session_state.get(
        "scan_results", []
    )

    review_items = []

    for result in scan_results:
        for record in result["records"]:
            if record["Review status"] != "Catalogue match":
                review_items.append(
                    (result["filename"], record)
                )

    if not review_items:

        st.info(
            "No unverified detections in the current session. "
            "Run a scan first."
        )

    for idx, (filename, record) in enumerate(review_items):

        with st.container(border=True):

            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(
                    record["_crop"],
                    use_container_width=True
                )

            with col2:

                st.write(
                    f"**Detected:** {record['Detected category']}"
                )

                st.caption(f"Image: {filename}")

                with st.form(f"review_form_{idx}"):

                    name = st.text_input(
                        "Verified product name",
                        key=f"name_{idx}"
                    )

                    brand = st.text_input(
                        "Brand",
                        key=f"brand_{idx}"
                    )

                    sku = st.text_input(
                        "Unique SKU",
                        value=(
                            "SKU-"
                            + dt.datetime.now().strftime(
                                "%y%m%d%H%M%S"
                            )
                            + f"-{idx}"
                        ),
                        key=f"sku_{idx}"
                    )

                    category = st.text_input(
                        "Category",
                        value=record["Detected category"],
                        key=f"category_{idx}"
                    )

                    barcode = st.text_input(
                        "Actual barcode (optional)",
                        key=f"barcode_{idx}"
                    )

                    submitted = st.form_submit_button(
                        "Save verified product"
                    )

                    if submitted:

                        if not name.strip() or not sku.strip():
                            st.error(
                                "Product name and SKU are required."
                            )

                        else:

                            safe_sku = re.sub(
                                r"[^A-Za-z0-9_-]",
                                "_",
                                sku.strip()
                            )

                            ref_path = str(
                                REFERENCE_DIR
                                / f"{safe_sku}.png"
                            )

                            record["_crop"].save(ref_path)

                            try:

                                run_sql("""
                                INSERT INTO products
                                (sku,name,brand,category,barcode,
                                 unit_price,stock_quantity,
                                 reorder_threshold,expiry_date,
                                 reference_image)
                                VALUES (?,?,?,?,?,?,?,?,?,?)
                                """, (
                                    sku.strip(),
                                    name.strip(),
                                    brand.strip(),
                                    category.strip(),
                                    barcode.strip() or None,
                                    0.0, 0, 5, "", ref_path
                                ))

                                st.success(
                                    "Product registered. "
                                    "The detector itself has not "
                                    "been retrained."
                                )

                                st.rerun()

                            except sqlite3.IntegrityError:
                                st.error(
                                    "SKU or barcode already exists."
                                )


# ============================================================
# PAGE 4 — INVENTORY
# ============================================================

elif st.session_state.page == PAGES[3]:

    st.title("📦 Inventory Catalogue")

    df = read_df(
        "SELECT * FROM products ORDER BY name"
    )

    if not df.empty:

        low = df[
            df["stock_quantity"]
            <= df["reorder_threshold"]
        ]

        expiry_cutoff = (
            dt.date.today() + dt.timedelta(days=30)
        ).isoformat()

        expiring = df[
            (df["expiry_date"].fillna("") != "")
            & (df["expiry_date"] <= expiry_cutoff)
        ]

        a, b, c = st.columns(3)

        a.metric("Catalogue items", len(df))
        b.metric("Low-stock items", len(low))
        c.metric("Expired/expiring items", len(expiring))

        st.dataframe(
            df.drop(columns=["reference_image"]),
            use_container_width=True
        )

        st.subheader("Add a product")

        with st.form("add_product_form"):

            a, b, c = st.columns(3)

            sku = a.text_input("SKU *")
            name = b.text_input("Product name *")
            brand = c.text_input("Brand")

            d, e, f = st.columns(3)

            category = d.text_input("Category")
            barcode = e.text_input("Actual barcode")
            price = f.number_input(
                "Unit price",
                min_value=0.0,
                value=0.0,
                step=0.1
            )

            g, h, i = st.columns(3)

            stock = g.number_input(
                "Stock quantity",
                min_value=0,
                value=0,
                step=1
            )

            reorder = h.number_input(
                "Reorder threshold",
                min_value=0,
                value=5,
                step=1
            )

            expiry = i.date_input(
                "Expiry date",
                value=None
            )

            submitted = st.form_submit_button(
                "Add product"
            )

            if submitted:

                if not sku.strip() or not name.strip():
                    st.error(
                        "SKU and product name are required."
                    )

                else:

                    expiry_text = (
                        expiry.isoformat()
                        if expiry else ""
                    )

                    try:

                        run_sql("""
                        INSERT INTO products
                        (sku,name,brand,category,barcode,
                         unit_price,stock_quantity,
                         reorder_threshold,expiry_date,
                         reference_image)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                        """, (
                            sku.strip(),
                            name.strip(),
                            brand.strip(),
                            category.strip(),
                            barcode.strip() or None,
                            price,
                            int(stock),
                            int(reorder),
                            expiry_text,
                            ""
                        ))

                        st.success("Product added.")
                        st.rerun()

                    except sqlite3.IntegrityError:
                        st.error(
                            "SKU or barcode already exists."
                        )

        st.subheader("Update stock")

        options = {
            f"{row['sku']} — {row['name']}":
            int(row["product_id"])
            for _, row in df.iterrows()
        }

        with st.form("stock_form"):

            chosen = st.selectbox(
                "Product",
                list(options.keys())
            )

            new_stock = st.number_input(
                "New stock quantity",
                min_value=0,
                value=0,
                step=1
            )

            submitted = st.form_submit_button(
                "Update stock"
            )

            if submitted:

                run_sql(
                    "UPDATE products SET stock_quantity=? "
                    "WHERE product_id=?",
                    (
                        int(new_stock),
                        options[chosen]
                    )
                )

                st.success("Stock updated.")
                st.rerun()

        if not low.empty:
            st.warning(
                "Low stock: "
                + ", ".join(low["name"].astype(str))
            )

        if not expiring.empty:
            st.warning(
                "Expiry review: "
                + ", ".join(expiring["name"].astype(str))
            )

    else:
        st.info("Catalogue is empty.")


# ============================================================
# PAGE 5 — ANALYTICS
# ============================================================

elif st.session_state.page == PAGES[4]:

    st.title("📊 Analytics")

    df = read_df(
        "SELECT * FROM detections ORDER BY timestamp DESC"
    )

    products = read_df("SELECT * FROM products")

    if df.empty:

        st.info(
            "No scan records yet. Run a scan to populate analytics."
        )

    else:

        total = len(df)

        matched = int(
            (df["review_status"] == "Catalogue match").sum()
        )

        needs_review = total - matched

        avg_conf = float(df["confidence"].mean())

        a, b, c, d = st.columns(4)

        a.metric("Detection records", total)
        b.metric("Catalogue matches", matched)
        c.metric("Needs review", needs_review)
        d.metric("Mean confidence", f"{avg_conf:.2f}")

        st.subheader("Detections by object label")

        label_counts = (
            df["detected_label"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Object label")
            .reset_index(name="Count")
        )

        fig1 = px.bar(
            label_counts,
            x="Object label",
            y="Count",
            title="Detections by prompted category"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        st.caption(
            "Counts logged bounding boxes, not unique products."
        )

        st.subheader("Identification outcomes")

        outcome = (
            df["review_status"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Count")
        )

        fig2 = px.pie(
            outcome,
            names="Outcome",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.subheader("Identification methods")

        methods = (
            df["identity_method"]
            .fillna("Not recorded")
            .value_counts()
            .rename_axis("Method")
            .reset_index(name="Count")
        )

        fig3 = px.bar(
            methods,
            x="Method",
            y="Count"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.subheader("Confidence distribution")

        fig4 = px.histogram(
            df,
            x="confidence",
            nbins=10
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        st.caption(
            "Detector confidence is not the probability that "
            "the exact product name is correct."
        )

        st.subheader("Inventory vs. reorder threshold")

        if not products.empty:

            inv = products[
                ["name", "stock_quantity", "reorder_threshold"]
            ].melt(
                id_vars="name",
                var_name="Measure",
                value_name="Units"
            )

            fig5 = px.bar(
                inv,
                x="name",
                y="Units",
                color="Measure",
                barmode="group"
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )

        st.subheader("Recent detection records")

        st.dataframe(
            df,
            use_container_width=True
        )


# ============================================================
# PAGE 6 — REPORTS
# ============================================================

elif st.session_state.page == PAGES[5]:

    st.title("📥 Reports & Evaluation")

    inventory = read_df(
        "SELECT * FROM products ORDER BY name"
    )

    detections = read_df(
        "SELECT * FROM detections ORDER BY timestamp DESC"
    )

    a, b = st.columns(2)

    a.download_button(
        "Download inventory CSV",
        inventory.to_csv(index=False).encode("utf-8"),
        file_name="smartmart_inventory.csv",
        mime="text/csv"
    )

    b.download_button(
        "Download detection log CSV",
        detections.to_csv(index=False).encode("utf-8"),
        file_name="smartmart_detection_log.csv",
        mime="text/csv"
    )

    st.subheader("Evaluation checklist")

    st.markdown("""
    - Test on images not used to tune the prompts.
    - Manually label the actual products in each image.
    - Measure missed objects and false detections.
    - Measure exact brand identification separately.
    - Test small products, overlapping packaging and poor lighting.
    - Use actual barcodes in the product catalogue.
    - Do not claim 100% accuracy without measured evidence.
    """)

    st.caption(
        f"Database: {DB_PATH}"
    )

    st.caption(detector_message)
    st.caption(ocr_message)
