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
from docx.shared import Inches
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
    st.markdown("---")
    st.markdown("**This version compares several growth metrics:**")
    st.markdown("- Original mean darkness")
    st.markdown("- Dark colony area")
    st.markdown("- Growth index = dark area × darkness")
    st.markdown("- Background-corrected integrated darkness")

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

# ── Local background helper ───────────────────────────────────────────────────
def estimate_local_background(gray, bbox, image_shape, pad_factor=2.0):
    """
    Estimate local agar/background intensity from a rectangular ring around the colony bbox.
    Uses the median, which is robust to dust/small artifacts.
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape[:2]

    pad_x = max(8, int(w * pad_factor))
    pad_y = max(8, int(h * pad_factor))

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img_w, x + w + pad_x)
    y1 = min(img_h, y + h + pad_y)

    outer = gray[y0:y1, x0:x1]
    if outer.size == 0:
        return 255.0

    ring_mask = np.ones(outer.shape, dtype=bool)
    inner_x0 = x - x0
    inner_y0 = y - y0
    inner_x1 = inner_x0 + w
    inner_y1 = inner_y0 + h
    ring_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False

    ring_pixels = outer[ring_mask]
    if ring_pixels.size < 20:
        return float(np.median(outer))
    return float(np.median(ring_pixels))

# ── Colony detection and measurement ──────────────────────────────────────────
def detect_colonies(image_np):
    """
    Find exactly 16 colonies and measure several possible growth metrics.

    Important metrics:
    - darkness_score: original metric, 255 - mean intensity inside detected colony mask.
      Kept because it produced the expected biological curve in the original app.
    - dark_area_px: number of pixels inside the colony mask that are meaningfully darker
      than the local agar/background.
    - growth_index: dark_area_px × darkness_score. This preserves the original darkness
      behavior but allows expansion/growth in area to increase the signal.
    - bg_corrected_integrated_darkness: sum of local_background - colony_pixel, clipped at 0.
      Included for comparison/debugging, but not used as the main chart.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Even out uneven lighting for segmentation only.
    # Measurements are done on the original grayscale image.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Threshold: colonies are dark on a lighter background.
    _, thresh = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up the segmentation mask.
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Find blobs.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    colonies = []
    for label_idx in range(1, num_labels):  # skip background (0)
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_idx]

        if area <= 0 or w <= 0 or h <= 0:
            continue

        blob_pixels = gray[labels == label_idx].astype(np.float32)
        mean_intensity = float(np.mean(blob_pixels)) if blob_pixels.size else 255.0

        # Original metric. This is intentionally retained because the user reported
        # that the original output looked biologically plausible.
        darkness_score = float(255.0 - mean_intensity)

        local_bg = estimate_local_background(gray, (x, y, w, h), gray.shape)
        bg_diff = np.clip(local_bg - blob_pixels, 0, None)
        bg_corrected_integrated_darkness = float(np.sum(bg_diff))

        # Area of meaningfully dark colony pixels. The threshold is adaptive:
        # a pixel counts as colony signal only if it is at least 10 intensity units
        # darker than the nearby local background.
        dark_threshold = 10.0
        dark_area_px = int(np.sum((local_bg - blob_pixels) > dark_threshold))

        # Main combined growth metric:
        # Keeps the original darkness behavior but allows increased dark colony area
        # to raise the growth signal.
        growth_index = float(darkness_score * dark_area_px)

        colonies.append({
            "area_px": area,
            "dark_area_px": dark_area_px,
            "cx": float(cx),
            "cy": float(cy),
            "bbox": (x, y, w, h),
            "local_background": round(local_bg, 2),
            "mean_intensity": round(mean_intensity, 2),
            "darkness_score": round(darkness_score, 2),
            "growth_index": round(growth_index, 2),
            "bg_corrected_integrated_darkness": round(bg_corrected_integrated_darkness, 2),
        })

    # Take 16 largest connected components by segmented area.
    colonies = sorted(colonies, key=lambda c: c["area_px"], reverse=True)[:16]

    if len(colonies) < 16:
        # Pad missing colonies with zeros.
        while len(colonies) < 16:
            colonies.append({
                "area_px": 0,
                "dark_area_px": 0,
                "cx": 0.0,
                "cy": 0.0,
                "bbox": (0, 0, 0, 0),
                "local_background": 255.0,
                "mean_intensity": 255.0,
                "darkness_score": 0.0,
                "growth_index": 0.0,
                "bg_corrected_integrated_darkness": 0.0,
            })

    # Sort into 4x4 grid: sort by Y (row), then X (col).
    colonies = sorted(colonies, key=lambda c: (c["cy"], c["cx"]))
    rows = []
    for i in range(4):
        row = sorted(colonies[i * 4:(i + 1) * 4], key=lambda c: c["cx"])
        rows.append(row)

    # Assign grid labels.
    result = []
    for r_idx, row in enumerate(rows):
        for c_idx, colony in enumerate(row):
            colony["colony_id"] = f"R{r_idx + 1}C{c_idx + 1}"
            result.append(colony)

    return result

# ── Relative metric helper ───────────────────────────────────────────────────
def add_relative_growth(batch_data, metric="growth_index"):
    """Add relative_growth = metric / first nonzero value for each colony."""
    if not batch_data:
        return batch_data

    colony_ids = [c["colony_id"] for c in batch_data[0]["colonies"]]
    baseline_by_colony = {}

    for cid in colony_ids:
        baseline = None
        for day in batch_data:
            colony = next((c for c in day["colonies"] if c["colony_id"] == cid), None)
            if colony and colony.get(metric, 0) > 0:
                baseline = float(colony[metric])
                break
        baseline_by_colony[cid] = baseline

    for day in batch_data:
        for colony in day["colonies"]:
            baseline = baseline_by_colony.get(colony["colony_id"])
            value = float(colony.get(metric, 0))
            if baseline and baseline > 0:
                colony["relative_growth"] = round(value / baseline, 4)
            else:
                colony["relative_growth"] = np.nan

    return batch_data

# ── QC checks ─────────────────────────────────────────────────────────────────
def run_qc(colony):
    flags = []
    if colony["area_px"] == 0:
        flags.append("colony not detected")
    if colony["darkness_score"] < 5:
        flags.append("very low darkness — possible empty spot")
    if colony.get("dark_area_px", 0) == 0:
        flags.append("no dark-area signal above local background")
    return ("FLAG" if flags else "OK"), "; ".join(flags)

# ── Chart builder ─────────────────────────────────────────────────────────────
def make_chart(batch_data, metric, metric_label, batch_name, include_average=True):
    """
    batch_data: list of {date, colonies: [{colony_id, area_px, darkness_score, ...}]}
    metric: e.g. 'darkness_score', 'dark_area_px', 'growth_index', 'relative_growth'
    Returns matplotlib figure as BytesIO PNG.
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
            val = colony.get(metric, np.nan) if colony else np.nan
            values.append(val)
        ax.plot(date_labels, values, color=colors[i], alpha=0.4, linewidth=1.2, marker="o", markersize=3, label=cid)
        all_values.append(values)

    if include_average and all_values:
        avg_values = np.nanmean(np.array(all_values, dtype=float), axis=0)
        ax.plot(date_labels, avg_values, color="black", linewidth=2.8, marker="o", markersize=4, label="Average", zorder=5)

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

# ── Word doc helpers ──────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def build_word_doc(batch_name, batch_data, growth_chart_buf, darkness_chart_buf, area_chart_buf):
    doc = Document()
    title = doc.add_heading(f"Colony Analysis Report — Batch {batch_name}", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dates_str = ", ".join(d["date"].strftime("%b %d %Y") for d in batch_data)
    doc.add_paragraph(f"Dates analyzed: {dates_str}")
    doc.add_paragraph("Colonies per plate: 16 (4×4 grid, technical replicates)")
    doc.add_paragraph("Main metric in this version: Growth Index = dark area × original mean darkness.")
    doc.add_paragraph("This version also keeps original mean darkness and dark-area measurements for comparison.")
    doc.add_paragraph("")

    # Charts
    doc.add_heading("Growth Index Over Time", level=1)
    growth_chart_buf.seek(0)
    doc.add_picture(growth_chart_buf, width=Inches(6))
    doc.add_paragraph("")

    doc.add_heading("Original Mean Darkness Over Time", level=1)
    darkness_chart_buf.seek(0)
    doc.add_picture(darkness_chart_buf, width=Inches(6))
    doc.add_paragraph("")

    doc.add_heading("Dark Colony Area Over Time", level=1)
    area_chart_buf.seek(0)
    doc.add_picture(area_chart_buf, width=Inches(6))
    doc.add_paragraph("")

    # Per-day tables
    doc.add_heading("Raw Data Per Day", level=1)
    for day in batch_data:
        doc.add_heading(day["date"].strftime("%B %d, %Y"), level=2)

        valid = [c for c in day["colonies"] if c["area_px"] > 0]
        avg_growth = round(np.mean([c["growth_index"] for c in valid]), 2) if valid else 0
        avg_relative = round(np.nanmean([c["relative_growth"] for c in valid]), 3) if valid else 0
        avg_darkness = round(np.mean([c["darkness_score"] for c in valid]), 2) if valid else 0
        avg_dark_area = round(np.mean([c["dark_area_px"] for c in valid]), 1) if valid else 0
        flags = sum(1 for c in day["colonies"] if c["qc_flag"] == "FLAG")
        doc.add_paragraph(
            f"Average growth index: {avg_growth}   |   Average relative growth: {avg_relative}   |   "
            f"Average original darkness: {avg_darkness}   |   Average dark area: {avg_dark_area} px   |   Flagged: {flags}/16"
        )

        table = doc.add_table(rows=1, cols=9)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        headers = [
            "Colony", "Growth Index", "Relative Growth", "Dark Area (px)", "Original Darkness",
            "Mean Intensity", "Local BG", "QC", "Notes"
        ]
        for i, h in enumerate(headers):
            hdr[i].text = h
            hdr[i].paragraphs[0].runs[0].bold = True
            set_cell_bg(hdr[i], "D9E1F2")

        for c in day["colonies"]:
            rc = table.add_row().cells
            rc[0].text = c["colony_id"]
            rc[1].text = str(c["growth_index"])
            rc[2].text = str(c["relative_growth"])
            rc[3].text = str(c["dark_area_px"])
            rc[4].text = str(c["darkness_score"])
            rc[5].text = str(c["mean_intensity"])
            rc[6].text = str(c["local_background"])
            rc[7].text = c["qc_flag"]
            rc[8].text = c["qc_note"]
            set_cell_bg(rc[7], "FFD7D7" if c["qc_flag"] == "FLAG" else "D7FFD7")

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
        avg_growth = round(np.mean([c["growth_index"] for c in valid]), 2) if valid else 0
        avg_relative = round(np.nanmean([c["relative_growth"] for c in valid]), 4) if valid else 0
        avg_darkness = round(np.mean([c["darkness_score"] for c in valid]), 2) if valid else 0
        avg_dark_area = round(np.mean([c["dark_area_px"] for c in valid]), 1) if valid else 0
        avg_area = round(np.mean([c["area_px"] for c in valid]), 1) if valid else 0

        for c in day["colonies"]:
            rows.append({
                "batch": batch_name,
                "date": day["date"].strftime("%Y-%m-%d"),
                "colony_id": c["colony_id"],
                "growth_index": c["growth_index"],
                "relative_growth": c["relative_growth"],
                "dark_area_px": c["dark_area_px"],
                "original_darkness_score": c["darkness_score"],
                "area_px": c["area_px"],
                "mean_intensity": c["mean_intensity"],
                "local_background": c["local_background"],
                "bg_corrected_integrated_darkness": c["bg_corrected_integrated_darkness"],
                "qc_flag": c["qc_flag"],
                "qc_note": c["qc_note"],
                "day_avg_growth_index": avg_growth,
                "day_avg_relative_growth": avg_relative,
                "day_avg_original_darkness": avg_darkness,
                "day_avg_dark_area_px": avg_dark_area,
                "day_avg_area_px": avg_area,
            })

        rows.append({
            "batch": batch_name,
            "date": day["date"].strftime("%Y-%m-%d"),
            "colony_id": "AVERAGE",
            "growth_index": avg_growth,
            "relative_growth": avg_relative,
            "dark_area_px": avg_dark_area,
            "original_darkness_score": avg_darkness,
            "area_px": avg_area,
            "mean_intensity": "",
            "local_background": "",
            "bg_corrected_integrated_darkness": round(float(np.mean([c["bg_corrected_integrated_darkness"] for c in valid])), 2) if valid else 0,
            "qc_flag": "",
            "qc_note": "",
            "std_growth_index": round(float(np.std([c["growth_index"] for c in valid])), 2) if valid else 0,
            "std_relative_growth": round(float(np.nanstd([c["relative_growth"] for c in valid])), 4) if valid else 0,
            "std_original_darkness": round(float(np.std([c["darkness_score"] for c in valid])), 2) if valid else 0,
            "std_dark_area_px": round(float(np.std([c["dark_area_px"] for c in valid])), 1) if valid else 0,
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

    # Sort each batch by date.
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

        # Add relative growth based on the main metric.
        batch_data = add_relative_growth(batch_data, metric="growth_index")

        # ── Generate chart bytes ──────────────────────────────────────────────
        growth_chart_bytes = make_chart(batch_data, "growth_index", "Growth Index (Dark Area × Original Darkness)", batch_name).getvalue()
        relative_chart_bytes = make_chart(batch_data, "relative_growth", "Relative Growth (Fold Change from First Nonzero Day)", batch_name).getvalue()
        darkness_chart_bytes = make_chart(batch_data, "darkness_score", "Original Mean Darkness Score", batch_name).getvalue()
        dark_area_chart_bytes = make_chart(batch_data, "dark_area_px", "Dark Colony Area (px)", batch_name).getvalue()
        bg_integrated_chart_bytes = make_chart(batch_data, "bg_corrected_integrated_darkness", "Background-Corrected Integrated Darkness", batch_name).getvalue()

        # ── Summary table (colony averages across all days) ───────────────────
        colony_ids = [c["colony_id"] for c in batch_data[0]["colonies"]]
        summary_rows = []
        for cid in colony_ids:
            growth_values = []
            relative_values = []
            darkness_values = []
            dark_area_values = []
            area_values = []
            for day in batch_data:
                match = next((c for c in day["colonies"] if c["colony_id"] == cid), None)
                if match:
                    growth_values.append(match["growth_index"])
                    relative_values.append(match["relative_growth"])
                    darkness_values.append(match["darkness_score"])
                    dark_area_values.append(match["dark_area_px"])
                    area_values.append(match["area_px"])
            summary_rows.append({
                "Colony": cid,
                "Avg Growth Index": round(np.mean(growth_values), 2) if growth_values else 0,
                "Avg Relative Growth": round(np.nanmean(relative_values), 3) if relative_values else 0,
                "Avg Original Darkness": round(np.mean(darkness_values), 2) if darkness_values else 0,
                "Avg Dark Area (px)": round(np.mean(dark_area_values), 1) if dark_area_values else 0,
                "Avg Segmented Area (px)": round(np.mean(area_values), 1) if area_values else 0,
            })

        df_sum = pd.DataFrame(summary_rows)
        avg_row = pd.DataFrame([{
            "Colony": "AVERAGE",
            "Avg Growth Index": round(df_sum["Avg Growth Index"].mean(), 2),
            "Avg Relative Growth": round(df_sum["Avg Relative Growth"].mean(), 3),
            "Avg Original Darkness": round(df_sum["Avg Original Darkness"].mean(), 2),
            "Avg Dark Area (px)": round(df_sum["Avg Dark Area (px)"].mean(), 1),
            "Avg Segmented Area (px)": round(df_sum["Avg Segmented Area (px)"].mean(), 1),
        }])
        df_sum = pd.concat([df_sum, avg_row], ignore_index=True)

        # ── Build download bytes ──────────────────────────────────────────────
        csv_bytes = build_csv(batch_name, batch_data)
        word_bytes = build_word_doc(
            batch_name,
            batch_data,
            make_chart(batch_data, "growth_index", "Growth Index (Dark Area × Original Darkness)", batch_name),
            make_chart(batch_data, "darkness_score", "Original Mean Darkness Score", batch_name),
            make_chart(batch_data, "dark_area_px", "Dark Colony Area (px)", batch_name),
        ).getvalue()

        batch_results.append({
            "batch_name": batch_name,
            "batch_data": batch_data,
            "growth_chart_bytes": growth_chart_bytes,
            "relative_chart_bytes": relative_chart_bytes,
            "darkness_chart_bytes": darkness_chart_bytes,
            "dark_area_chart_bytes": dark_area_chart_bytes,
            "bg_integrated_chart_bytes": bg_integrated_chart_bytes,
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
        st.markdown(
            "**Recommended first graph to inspect:** Growth Index. "
            "It combines the original darkness signal with dark colony area, so it should preserve the original curve shape while still rewarding colony expansion."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(result["growth_chart_bytes"], caption=f"Batch {batch_name} — Growth Index", use_column_width=True)
        with col2:
            st.image(result["relative_chart_bytes"], caption=f"Batch {batch_name} — Relative Growth", use_column_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.image(result["darkness_chart_bytes"], caption=f"Batch {batch_name} — Original Mean Darkness", use_column_width=True)
        with col4:
            st.image(result["dark_area_chart_bytes"], caption=f"Batch {batch_name} — Dark Colony Area", use_column_width=True)

        with st.expander("Show background-corrected integrated darkness chart"):
            st.image(result["bg_integrated_chart_bytes"], caption=f"Batch {batch_name} — Background-Corrected Integrated Darkness", use_column_width=True)

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
