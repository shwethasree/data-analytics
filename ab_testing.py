
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)
from datetime import datetime


# =========================================================
# 1. LOAD AND VALIDATE DATASET
# =========================================================

DATA_FILE = "checkout_ab_test_v2.csv"

df = pd.read_csv(DATA_FILE)

required_columns = ["VisitorID", "Variant", "Converted"]

if not all(col in df.columns for col in required_columns):
    raise ValueError(
        f"Dataset must contain these columns: {required_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in required columns.")

if not set(df["Variant"].unique()).issubset({"A", "B"}):
    raise ValueError("Unexpected variant labels. Expected A and B.")

if not set(df["Converted"].unique()).issubset({"Yes", "No"}):
    raise ValueError("Unexpected conversion values. Expected Yes and No.")

if df["VisitorID"].duplicated().any():
    raise ValueError("Duplicate visitor IDs found.")

if set(df["Variant"].unique()) != {"A", "B"}:
    raise ValueError("Both Control A and Variant B must be present.")


# =========================================================
# 2. CALCULATE CONVERSION METRICS
# =========================================================

summary = df.groupby("Variant").agg(
    Visitors=("VisitorID", "count"),
    Conversions=(
        "Converted",
        lambda x: (x == "Yes").sum()
    )
)

nA = int(summary.loc["A", "Visitors"])
nB = int(summary.loc["B", "Visitors"])

xA = int(summary.loc["A", "Conversions"])
xB = int(summary.loc["B", "Conversions"])

if nA == 0 or nB == 0:
    raise ValueError("Both groups must contain visitors.")

pA = xA / nA
pB = xB / nB

difference = pB - pA

relative_lift = (
    difference / pA if pA > 0 else np.nan
)

summary["Non_Conversions"] = (
    summary["Visitors"] - summary["Conversions"]
)

summary["Conversion_Rate_%"] = (
    summary["Conversions"] / summary["Visitors"] * 100
)


# =========================================================
# 3. TWO-PROPORTION Z-TEST
# =========================================================

# H0: Conversion rates are equal.
# H1: Conversion rates are different.

alpha = 0.05

pooled = (xA + xB) / (nA + nB)

se_null = np.sqrt(
    pooled * (1 - pooled) * (1 / nA + 1 / nB)
)

z_stat = difference / se_null

# Two-sided p-value
p_value = 2 * stats.norm.sf(abs(z_stat))


# =========================================================
# 4. CONFIDENCE INTERVAL AND EFFECT SIZE
# =========================================================

se_diff = np.sqrt(
    pA * (1 - pA) / nA
    + pB * (1 - pB) / nB
)

z_critical = stats.norm.ppf(1 - alpha / 2)

ci_low = difference - z_critical * se_diff
ci_high = difference + z_critical * se_diff

# Cohen's h effect size
cohens_h = (
    2 * np.arcsin(np.sqrt(pB))
    - 2 * np.arcsin(np.sqrt(pA))
)

significant = p_value < alpha

decision = (
    "Statistically significant difference"
    if significant
    else "No statistically significant difference"
)

recommendation = (
    "Consider launching Variant B, subject to business "
    "checks, because the result is statistically significant."
    if significant
    else
    "Do not declare Variant B the winner yet. Continue "
    "testing with an adequately planned sample size."
)


# =========================================================
# 5. DISPLAY RESULTS IN TERMINAL
# =========================================================

print("\n" + "=" * 55)
print("       A/B TESTING AND HYPOTHESIS TESTING")
print("=" * 55)

print("\n--- DATA QUALITY ---")
print("Dataset shape:", df.shape)
print("Total missing values:", int(df.isnull().sum().sum()))
print("Unique visitors:", df["VisitorID"].nunique())

print("\n--- CONVERSION SUMMARY ---")
print(summary.round(4))

print("\n--- HYPOTHESES ---")
print("H0: Conversion rates of A and B are equal.")
print("H1: Conversion rates of A and B are different.")
print("Significance level:", alpha)

print("\n--- STATISTICAL RESULTS ---")
print(f"Control A: {pA:.2%} ({xA}/{nA})")
print(f"Variant B: {pB:.2%} ({xB}/{nB})")
print(f"Absolute lift: {difference:.2%}")
print(f"Relative lift: {relative_lift:.2%}")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"95% confidence interval: [{ci_low:.2%}, {ci_high:.2%}]")
print(f"Cohen's h: {cohens_h:.4f}")
print("Decision:", decision)
print("Recommendation:", recommendation)


# =========================================================
# 6. SAVE SUMMARY CSV
# =========================================================

summary.to_csv("ab_test_summary.csv")

print("\nSummary saved: ab_test_summary.csv")


# =========================================================
# 7. GENERATE CONVERSION COMPARISON GRAPH
# =========================================================

labels = ["Control A", "Variant B"]
rates = [pA * 100, pB * 100]

fig, ax = plt.subplots(figsize=(7, 4.5))

bars = ax.bar(labels, rates)

ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Checkout A/B Test: Conversion Rates")
ax.set_ylim(0, max(rates) * 1.3)

for bar, rate in zip(bars, rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{rate:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    "conversion_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Graph saved: conversion_comparison.png")


# =========================================================
# 8. CREATE PDF REPORT
# =========================================================

pdf = SimpleDocTemplate(
    "ab_test_report.pdf",
    pagesize=A4,
    rightMargin=45,
    leftMargin=45,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=25,
        spaceAfter=12
    )
)

styles.add(
    ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        fontSize=13,
        leading=17,
        spaceAfter=10
    )
)

story = []


# ---------------- PAGE 1 ----------------

story.append(
    Paragraph(
        "A/B TESTING AND HYPOTHESIS TESTING REPORT",
        styles["ReportTitle"]
    )
)

story.append(
    Paragraph(
        "Checkout Conversion Experiment",
        styles["ReportSubtitle"]
    )
)

story.append(
    Paragraph(
        "<b>Dataset:</b> checkout_ab_test_v2.csv",
        styles["Normal"]
    )
)

story.append(
    Paragraph(
        "<b>Report generated:</b> "
        + datetime.now().strftime("%d %B %Y"),
        styles["Normal"]
    )
)

story.append(Spacer(1, 12))


# Objective
story.append(
    Paragraph("1. Objective", styles["Heading1"])
)

story.append(
    Paragraph(
        "Evaluate whether the modified checkout experience "
        "(Variant B) produces a different conversion rate from "
        "the existing checkout experience (Control A).",
        styles["BodyText"]
    )
)

story.append(Spacer(1, 8))


# Data quality
story.append(
    Paragraph("2. Data Quality", styles["Heading1"])
)

story.append(
    Paragraph(
        f"The dataset contains {len(df)} rows and "
        f"{len(df.columns)} columns, with "
        f"{df['VisitorID'].nunique()} unique visitors. "
        f"Total missing values across all columns: "
        f"{int(df.isnull().sum().sum())}.",
        styles["BodyText"]
    )
)

story.append(Spacer(1, 8))


# Conversion results
story.append(
    Paragraph("3. Conversion Results", styles["Heading1"])
)

conversion_data = [
    ["Metric", "Control A", "Variant B"],
    ["Visitors", str(nA), str(nB)],
    ["Conversions", str(xA), str(xB)],
    ["Non-conversions", str(nA - xA), str(nB - xB)],
    ["Conversion rate", f"{pA:.2%}", f"{pB:.2%}"]
]

conversion_table = Table(
    conversion_data,
    colWidths=[2.5 * inch, 1.35 * inch, 1.35 * inch],
    repeatRows=1
)

conversion_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#49358F")
        ),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        (
            "BACKGROUND",
            (0, 1),
            (-1, -1),
            colors.whitesmoke
        )
    ])
)

story.append(conversion_table)
story.append(Spacer(1, 8))

story.append(
    Image(
        "conversion_comparison.png",
        width=6 * inch,
        height=3.85 * inch
    )
)


# =========================================================
# PAGE BREAK: KEEP STATISTICAL RESULTS TOGETHER ON PAGE 2
# =========================================================

story.append(PageBreak())


# ---------------- PAGE 2 ----------------

story.append(
    Paragraph(
        "4. Hypotheses and Statistical Test",
        styles["Heading1"]
    )
)

story.append(
    Paragraph(
        "<b>Null hypothesis (H0):</b> The population conversion "
        "rates are equal.<br/>"
        "<b>Alternative hypothesis (H1):</b> The population "
        "conversion rates differ.<br/>"
        "A two-sided two-proportion Z-test was conducted at "
        "a significance level of 0.05.",
        styles["BodyText"]
    )
)

story.append(Spacer(1, 10))


# Statistical results table
test_data = [
    ["Statistical measure", "Result"],
    ["Absolute lift (B - A)", f"{difference:.2%}"],
    ["Relative lift", f"{relative_lift:.2%}"],
    ["Z-statistic", f"{z_stat:.4f}"],
    ["P-value", f"{p_value:.4f}"],
    [
        "95% confidence interval",
        f"[{ci_low:.2%}, {ci_high:.2%}]"
    ],
    ["Cohen's h", f"{cohens_h:.4f}"],
    ["Significance level", f"{alpha:.2f}"]
]

test_table = Table(
    test_data,
    colWidths=[2.7 * inch, 2.5 * inch],
    repeatRows=1
)

test_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#49358F")
        ),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ])
)

story.append(test_table)
story.append(Spacer(1, 12))


# Interpretation
story.append(
    Paragraph("5. Interpretation", styles["Heading1"])
)

story.append(
    Paragraph(
        f"Variant B converted {xB} of {nB} visitors ({pB:.2%}), "
        f"compared with {xA} of {nA} visitors ({pA:.2%}) for "
        f"Control A. The observed absolute lift was "
        f"{difference:.2%}, and the relative lift was "
        f"{relative_lift:.2%}. The p-value was {p_value:.4f}, "
        f"which is {'below' if significant else 'above'} the "
        f"0.05 significance level. The 95% confidence interval "
        f"[{ci_low:.2%}, {ci_high:.2%}] "
        f"{'does not include' if ci_low > 0 or ci_high < 0 else 'includes'} "
        f"zero. Therefore, the result is: {decision}.",
        styles["BodyText"]
    )
)

story.append(Spacer(1, 10))


# Executive decision memo
story.append(
    Paragraph("6. Executive Decision Memo", styles["Heading1"])
)

story.append(
    Paragraph(
        "<b>Objective:</b> Determine whether the modified checkout "
        "experience should replace the existing checkout page.",
        styles["BodyText"]
    )
)

story.append(Spacer(1, 5))

story.append(
    Paragraph(
        "<b>Finding:</b> Variant B recorded a higher observed "
        "conversion rate than Control A, but the difference was "
        "not statistically significant at the 5% level.",
        styles["BodyText"]
    )
)

story.append(Spacer(1, 5))

story.append(
    Paragraph(
        "<b>Recommendation:</b> " + recommendation,
        styles["BodyText"]
    )
)

story.append(Spacer(1, 5))

story.append(
    Paragraph(
        "<b>Further work:</b> Estimate an appropriate sample size "
        "for a follow-up experiment and investigate results by "
        "device type and traffic source before making a final "
        "rollout decision.",
        styles["BodyText"]
    )
)

story.append(Spacer(1, 10))


# Conclusion
story.append(
    Paragraph("7. Conclusion", styles["Heading1"])
)

story.append(
    Paragraph(
        "Variant B showed a higher observed conversion rate, "
        "but the experiment does not provide sufficient evidence "
        "of a difference at the 5% significance level. The null "
        "hypothesis cannot be rejected. Further testing is "
        "recommended before declaring Variant B the winner.",
        styles["BodyText"]
    )
)


# =========================================================
# 9. GENERATE PDF
# =========================================================

pdf.build(story)

print("\n" + "=" * 55)
print("       FILES GENERATED SUCCESSFULLY")
print("=" * 55)

print("1. ab_test_report.pdf")
print("2. conversion_comparison.png")
print("3. ab_test_summary.csv")

print("\nOpen ab_test_report.pdf to check the final layout.")
