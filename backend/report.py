"""
report.py
Generates summary_report.pdf using ReportLab.
Audience: coaches and physiotherapists — no clinical language.
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, HRFlowable, PageBreak,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


DISCLAIMER = (
    "IMPORTANT: This report is a movement quality screening tool only. "
    "It uses 2D camera estimates to assess movement patterns and is NOT "
    "a medical diagnosis. Do not make clinical or return-to-sport decisions "
    "based solely on this output. Consult a qualified sports medicine professional "
    "for any injury-related concerns."
)

KNOWN_LIMITATIONS = [
    "2D pose estimates carry up to 18° absolute error vs. 3D clinical systems; "
    "relative temporal trends are reliable, absolute angle values are not.",
    "Literature thresholds used as directional references, not exact clinical cutoffs.",
    "Proxy labels used in training; no verified injury outcome data.",
    "The annotation identifies the most prominent biomechanical signal in a flagged "
    "window, not the causal interaction the model detected (BiLSTM temporal "
    "co-occurrence patterns exceed per-frame sub-score attribution).",
]


def generate_report(
    output_pdf_path: str,
    timeline_img_path: str,
    clip_meta: dict,
    risk_summary: dict,
    annotated_events: List[dict],
) -> None:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed. Run: pip install reportlab")

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1a237e"), spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "H2Style", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#283593"), spaceBefore=12, spaceAfter=4,
    )
    body_style = styles["BodyText"]
    body_style.fontSize = 10
    warn_style = ParagraphStyle(
        "WarnStyle", parent=body_style,
        textColor=colors.HexColor("#b71c1c"),
        backColor=colors.HexColor("#fff8f8"),
        borderColor=colors.HexColor("#e53935"),
        borderWidth=1, borderPadding=6, spaceAfter=8,
    )

    story = []

    # ── Title ──────────────────────────────────────────────────────────────────
    story.append(Paragraph("ACL Risk Movement Screening Report", title_style))
    story.append(Paragraph("SCREENING ONLY — NOT A CLINICAL DIAGNOSIS", warn_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c5cae9")))
    story.append(Spacer(1, 0.3 * cm))

    # ── 1. Clip Metadata ──────────────────────────────────────────────────────
    story.append(Paragraph("1. Clip Metadata", h2_style))
    meta_data = [
        ["Field", "Value"],
        ["Filename", clip_meta.get("filename", "—")],
        ["Duration", f"{clip_meta.get('duration_sec', 0):.2f} s"],
        ["FPS", f"{clip_meta.get('fps', 0):.1f}"],
        ["Total frames analyzed", str(clip_meta.get("total_frames", 0))],
        ["Camera angle", clip_meta.get("camera_angle", "Not verified")],
    ]
    meta_table = Table(meta_data, colWidths=[5 * cm, 11 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5cae9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── 2. Risk Distribution ──────────────────────────────────────────────────
    story.append(Paragraph("2. Risk Distribution", h2_style))
    rs = risk_summary
    total = rs.get("total_windows", 1)
    risk_data = [
        ["Risk Level", "Windows", "Percentage"],
        ["Low",    str(rs.get("low_count", 0)),    f"{rs.get('low_pct', 0):.1f}%"],
        ["Medium", str(rs.get("medium_count", 0)), f"{rs.get('medium_pct', 0):.1f}%"],
        ["High",   str(rs.get("high_count", 0)),   f"{rs.get('high_pct', 0):.1f}%"],
        ["Total",  str(total), "100%"],
    ]
    risk_colors = [None, colors.HexColor("#e8f5e9"), colors.HexColor("#fff9c4"), colors.HexColor("#ffebee"), colors.HexColor("#e8eaf6")]
    risk_table = Table(risk_data, colWidths=[5 * cm, 5 * cm, 6 * cm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaf6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5cae9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        *[("BACKGROUND", (0, i), (-1, i), risk_colors[i]) for i in range(1, 5) if risk_colors[i]],
    ]))
    story.append(risk_table)

    # Peak risk event
    peak = rs.get("peak_high_window")
    if peak:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            f"<b>Peak Risk Event:</b> Frames {peak['start_frame']}–{peak['end_frame']} "
            f"| P(High) = {peak['P_high']:.3f}",
            body_style,
        ))
    story.append(Spacer(1, 0.4 * cm))

    # ── 3. Annotated Events ───────────────────────────────────────────────────
    story.append(Paragraph("3. Annotated High Risk Events", h2_style))
    if not annotated_events:
        story.append(Paragraph("No High Risk events detected in this clip.", body_style))
    else:
        ev_data = [["#", "Frames", "Signal", "Annotation"]]
        for i, ev in enumerate(annotated_events, 1):
            ev_data.append([
                str(i),
                f"{ev['start_frame']}–{ev['end_frame']}",
                ev.get("dominant_subscore", "—").upper(),
                ev.get("annotation", "—"),
            ])
        ev_table = Table(ev_data, colWidths=[1 * cm, 3 * cm, 2.5 * cm, 9.5 * cm])
        ev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaf6")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff3e0")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5cae9")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(ev_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── 4. Risk Timeline ──────────────────────────────────────────────────────
    if os.path.exists(timeline_img_path):
        story.append(Paragraph("4. Risk Score Timeline", h2_style))
        img = Image(timeline_img_path, width=16 * cm, height=8 * cm)
        story.append(img)
        story.append(Spacer(1, 0.4 * cm))

    # ── 5. Known Limitations ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Known Limitations (Mandatory Disclosure)", h2_style))
    story.append(Paragraph(DISCLAIMER, warn_style))
    for i, lim in enumerate(KNOWN_LIMITATIONS, 1):
        story.append(Paragraph(f"{i}. {lim}", body_style))
        story.append(Spacer(1, 0.15 * cm))

    doc.build(story)