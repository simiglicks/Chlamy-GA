import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import re
from datetime import datetime
from collections import defaultdict
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Colony Analyzer", page_icon="🔬", layout="wide")
st.title("🔬 Colony Plate Analyzer")
st.markdown("Upload all plate images at once. Images are grouped by batch letter and sorted by date automatically.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Filename format:** `A20260611.jpg` → Batch A, June 11 2026")
    st.markdown("---")
    st.markdown("**How to use:**\n1. Upload all images\n2. Click Analyze\n3. Download results per batch")

# ── Filename parser ───────────────────────────────────────────────────────────
def parse_filename(filename):
    """Extract batch letter and date from e.g. A20260611.jpg"""
    name = filename.rsplit(".", 1)[0]
    match = re.match(r"^([A-Za-z]+)(\d{8})$", name)
    if not match:
        return None, None
    batch = match.group(1).upper()
    try:
        date = datetime.strptime(match.group(2), "%Y%m%d")
    except ValueError:
        return None, None
    return batch, date

# ── Colony detection ──────────────────────────────────────────────────────────
def detect_colonies(image_np):
    """
    Find exactly 16 colonies:
    1. Convert to grayscale, even out lighting
    2. Threshold to find dark blobs
    3. Filter out blobs smaller than min_area (splashes)
    4. Take the 16 largest remaining blobs
    5. Sort into 4x4 grid by position (row then col)
    Returns list of 16 dicts with position, centroid, bbox, area, mean_intensity, darkness_score
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Even out uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Threshold — colonies are dark on light background
    _, thresh = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to fill holes in colonies
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    colonies = []
    for label_idx in range(1, num_labels):  # skip background (0)
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_idx]

        # Mean intensity from original (non-equalized) image within the blob mask
        blob_mask = (labels == label_idx).astype(np.uint8)
        mean_intensity = float(cv2.mean(gray, mask=blob_mask)[0])
        darkness_score = round(255 - mean_intensity, 2)

        colonies.append({
            "area_px": area,
            "cx": float(cx),
            "cy": float(cy),
            "bbox": (x, y, w, h),
            "mean_intensity": round(mean_intensity, 2),
            "darkness_score": darkness_score,
        })

    # Take 16 largest
    colonies = sorted(colonies, key=lambda c: c["area_px"], reverse=True)[:16]

    if len(colonies) < 16:
        # Pad missing colonies with zeros
        while len(colonies) < 16:
            colonies.append({
                "area_px": 0, "cx": 0.0, "cy": 0.0, "bbox": (0,0,0,0),
                "mean_intensity": 255.0, "darkness_score": 0.0
            })

    # Sort into 4x4 grid: sort by Y (row) then X (col)
    colonies = sorted(colonies, key=lambda c: (c["cy"], c["cx"]))
    rows = []
    for i in range(4):
        row = sorted(colonies[i*4:(i+1)*4], key=lambda c: c["cx"])
        rows.append(row)

    # Assign grid labels
    result = []
    for r_idx, row in enumerate(rows):
        for c_idx, colony in enumerate(row):
            colony["colony_id"] = f"R{r_idx+1}C{c_idx+1}"
            result.append(colony)

    return result

# ── QC checks ─────────────────────────────────────────────────────────────────
def run_qc(colony):
    flags = []
    if colony["area_px"] == 0:
        flags.append("colony not detected")
    if colony["darkness_score"] < 5:
        flags.append("very low darkness — possible empty spot")
    return ("FLAG" if flags else "OK"), "; ".join(flags)

# ── Chart builder ─────────────────────────────────────────────────────────────
def make_chart(batch_data, metric, metric_label, batch_name):
    """
    batch_data: list of {date, colonies: [{colony_id, area_px, darkness_score, ...}]}
    metric: 'darkness_score' or 'area_px'
    Returns matplotlib figure as BytesIO PNG
    """
    dates = [d["date"] for d in batch_data]
    date_labels = [d.strftime("%b %d") for d in dates]

    colony_ids = [c["colony_id"] for c in batch_data[0]["colonies"]]
    colors = cm.tab20(np.linspace(0, 1, 16))

    fig, ax = plt.subplots(figsize=(10, 5))

    all_values = []
    for i, cid in enumerate(colony_ids):
        values = []
        for day in batch_data:
            colony = next((c for c in day["colonies"] if c["colony_id"] == cid), None)
            values.append(colony[metric] if colony else 0)
        ax.plot(date_labels, values, color=colors[i], alpha=0.4, linewidth=1.2, label=cid)
        all_values.append(values)

    # Average line
    avg_values = np.mean(all_values, axis=0)
    ax.plot(date_labels, avg_values, color="black", linewidth=2.5, label="Average", zorder=5)

    ax.set_title(f"Batch {batch_name} — {metric_label} Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric_label)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Word doc ──────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def build_word_doc(batch_name, batch_data, dark_chart_buf, area_chart_buf):
    doc = Document()
    title = doc.add_heading(f"Colony Analysis Report — Batch {batch_name}", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dates_str = ", ".join(d["date"].strftime("%b %d %Y") for d in batch_data)
    doc.add_paragraph(f"Dates analyzed: {dates_str}")
    doc.add_paragraph(f"Colonies per plate: 16 (4×4 grid, technical replicates)")
    doc.add_paragraph("")

    # Charts
    doc.add_heading("Darkness Score Over Time", level=1)
    dark_chart_buf.seek(0)
    doc.add_picture(dark_chart_buf, width=Inches(6))
    doc.add_paragraph("")

    doc.add_heading("Colony Area Over Time", level=1)
    area_chart_buf.seek(0)
    doc.add_picture(area_chart_buf, width=Inches(6))
    doc.add_paragraph("")

    # Per-day tables
    doc.add_heading("Raw Data Per Day", level=1)
    for day in batch_data:
        doc.add_heading(day["date"].strftime("%B %d, %Y"), level=2)

        # Average row
        valid = [c for c in day["colonies"] if c["area_px"] > 0]
        avg_darkness = round(np.mean([c["darkness_score"] for c in valid]), 2) if valid else 0
        avg_area = round(np.mean([c["area_px"] for c in valid]), 1) if valid else 0
        flags = sum(1 for c in day["colonies"] if c["qc_flag"] == "FLAG")
        doc.add_paragraph(f"Average darkness: {avg_darkness}   |   Average area: {avg_area} px   |   Flagged: {flags}/16")

        table = doc.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        for i, h in enumerate(["Colony", "Area (px)", "Mean Intensity", "Darkness Score", "QC", "Notes"]):
            hdr[i].text = h
            hdr[i].paragraphs[0].runs[0].bold = True
            set_cell_bg(hdr[i], "D9E1F2")

        for c in day["colonies"]:
            rc = table.add_row().cells
            rc[0].text = c["colony_id"]
            rc[1].text = str(c["area_px"])
            rc[2].text = str(c["mean_intensity"])
            rc[3].text = str(c["darkness_score"])
            rc[4].text = c["qc_flag"]
            rc[5].text = c["qc_note"]
            set_cell_bg(rc[4], "FFD7D7" if c["qc_flag"] == "FLAG" else "D7FFD7")

        doc.add_paragraph("")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


# ── CSV builder ───────────────────────────────────────────────────────────────
def build_csv(batch_name, batch_data):
    rows = []
    for day in batch_data:
        valid = [c for c in day["colonies"] if c["area_px"] > 0]
        avg_darkness = round(np.mean([c["darkness_score"] for c in valid]), 2) if valid else 0
        avg_area = round(np.mean([c["area_px"] for c in valid]), 1) if valid else 0
        for c in day["colonies"]:
            rows.append({
                "batch": batch_name,
                "date": day["date"].strftime("%Y-%m-%d"),
                "colony_id": c["colony_id"],
                "area_px": c["area_px"],
                "mean_intensity": c["mean_intensity"],
                "darkness_score": c["darkness_score"],
                "qc_flag": c["qc_flag"],
                "qc_note": c["qc_note"],
                "day_avg_darkness": avg_darkness,
                "day_avg_area": avg_area,
            })
        all_darkness = [c["darkness_score"] for c in day["colonies"]]
        all_area = [c["area_px"] for c in day["colonies"]]
        rows.append({
            "batch": batch_name,
            "date": day["date"].strftime("%Y-%m-%d"),
            "colony_id": "AVERAGE",
            "area_px": round(float(np.mean(all_area)), 1),
            "darkness_score": round(float(np.mean(all_darkness)), 2),
            "std_darkness": round(float(np.std(all_darkness)), 2),
            "std_area": round(float(np.std(all_area)), 1),
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("### Upload Images")
st.info("Name files as `A20260611.jpg` (letter + YYYYMMDD). Upload all batches together.")

uploaded_files = st.file_uploader(
    "Upload all plate images",
    type=["jpg", "jpeg", "png", "tiff", "tif"],
    accept_multiple_files=True
)

run_btn = st.button("🔬 Analyze All Batches", type="primary", use_container_width=True,
                    disabled=not uploaded_files)

if run_btn and uploaded_files:
    # ── Parse and group files ─────────────────────────────────────────────────
    batches = defaultdict(list)
    bad_files = []
    for f in uploaded_files:
        batch, date = parse_filename(f.name)
        if batch is None:
            bad_files.append(f.name)
        else:
            batches[batch].append((date, f))

    if bad_files:
        st.warning(f"Skipped files with unrecognized names: {', '.join(bad_files)}\nExpected format: A20260611.jpg")

    if not batches:
        st.error("No valid files found. Check filenames match format: A20260611.jpg")
        st.stop()

    # Sort each batch by date
    for batch in batches:
        batches[batch] = sorted(batches[batch], key=lambda x: x[0])

    st.markdown(f"**Found {len(batches)} batch(es):** {', '.join(sorted(batches.keys()))}")
    st.markdown("---")

    total_images = sum(len(v) for v in batches.values())
    progress = st.progress(0)
    processed = 0

    batch_results = []

    # ── Process each batch ────────────────────────────────────────────────────
    for batch_name in sorted(batches.keys()):
        batch_days = batches[batch_name]
        batch_data = []

        for date, file in batch_days:
            status = st.empty()
            status.info(f"Batch {batch_name} — {date.strftime('%b %d %Y')}...")

            pil_image = Image.open(file).convert("RGB")
            image_np = np.array(pil_image)

            colonies = detect_colonies(image_np)

            # QC
            for c in colonies:
                c["qc_flag"], c["qc_note"] = run_qc(c)

            batch_data.append({"date": date, "colonies": colonies})
            processed += 1
            progress.progress(processed / total_images)
            status.empty()

        # ── Generate chart bytes (stored so reruns don't recompute) ──────────
        dark_chart_bytes = make_chart(batch_data, "darkness_score", "Darkness Score (255 − mean intensity)", batch_name).getvalue()
        area_chart_bytes = make_chart(batch_data, "area_px", "Colony Area (px)", batch_name).getvalue()

        # ── Summary table (colony averages across all days) ───────────────────
        colony_ids = [c["colony_id"] for c in batch_data[0]["colonies"]]
        summary_rows = []
        for cid in colony_ids:
            d_scores = []
            a_scores = []
            for day in batch_data:
                match = next((c for c in day["colonies"] if c["colony_id"] == cid), None)
                if match:
                    d_scores.append(match["darkness_score"])
                    a_scores.append(match["area_px"])
            summary_rows.append({
                "Colony": cid,
                "Avg Darkness": round(np.mean(d_scores), 2) if d_scores else 0,
                "Avg Area (px)": round(np.mean(a_scores), 1) if a_scores else 0,
            })

        df_sum = pd.DataFrame(summary_rows)
        avg_row = pd.DataFrame([{
            "Colony": "AVERAGE",
            "Avg Darkness": round(df_sum["Avg Darkness"].mean(), 2),
            "Avg Area (px)": round(df_sum["Avg Area (px)"].mean(), 1),
        }])
        df_sum = pd.concat([df_sum, avg_row], ignore_index=True)

        # ── Build download bytes ──────────────────────────────────────────────
        csv_bytes = build_csv(batch_name, batch_data)
        word_bytes = build_word_doc(
            batch_name, batch_data,
            make_chart(batch_data, "darkness_score", "Darkness Score", batch_name),
            make_chart(batch_data, "area_px", "Colony Area (px)", batch_name),
        ).getvalue()

        batch_results.append({
            "batch_name": batch_name,
            "batch_data": batch_data,
            "dark_chart_bytes": dark_chart_bytes,
            "area_chart_bytes": area_chart_bytes,
            "df_summary": df_sum,
            "csv_bytes": csv_bytes,
            "word_bytes": word_bytes,
        })

    st.session_state.batch_results = batch_results
    progress.progress(1.0)
    st.success("✅ All batches complete!")

# ── Display results (persists across reruns so downloads don't clear UI) ──────
if "batch_results" in st.session_state:
    for result in st.session_state.batch_results:
        batch_name = result["batch_name"]

        st.markdown(f"## Batch {batch_name}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(result["dark_chart_bytes"], caption=f"Batch {batch_name} — Darkness Score", use_column_width=True)
        with col2:
            st.image(result["area_chart_bytes"], caption=f"Batch {batch_name} — Colony Area", use_column_width=True)

        st.markdown(f"#### Colony Averages — Batch {batch_name}")
        st.dataframe(result["df_summary"], use_container_width=True, hide_index=True)

        st.markdown(f"#### Download — Batch {batch_name}")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                f"📥 CSV — Batch {batch_name}",
                result["csv_bytes"],
                file_name=f"batch_{batch_name}_analysis.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"csv_{batch_name}",
            )
        with dl2:
            st.download_button(
                f"📄 Word Report — Batch {batch_name}",
                result["word_bytes"],
                file_name=f"batch_{batch_name}_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key=f"word_{batch_name}",
            )

        st.markdown("---")
