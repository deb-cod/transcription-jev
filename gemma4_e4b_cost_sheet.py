from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side
import csv

latency_seconds = 4.914
watts = 100
price_per_kwh = 0.15

# Formula:
# energy_kwh = watts * latency_seconds / 3,600,000
# cost_per_question = energy_kwh * price_per_kwh
cost_per_question = (watts * latency_seconds / 3_600_000) * price_per_kwh
cost_10k = cost_per_question * 10_000
cost_1m = cost_per_question * 1_000_000

rows = [
    [
        "Backend",
        "Model",
        "Latency (s)",
        "Power draw (W)",
        "Electricity price ($/kWh)",
        "Energy per question (kWh)",
        "Cost per question ($)",
        "Cost per 10,000 questions ($)",
        "Cost per 1,000,000 questions ($)",
        "Notes",
    ],
    [
        "OpenJev native Gemma (experimental)",
        "gemma4-e4b-native",
        latency_seconds,
        watts,
        price_per_kwh,
        "=B2*A2/3600000",
        "=F2*C2",
        "=G2*10000",
        "=G2*1000000",
        "Native Gemma via llama.cpp; from README_INFERENCE_BENCHMARK.md and README_INFERENCE_COSTS.md",
    ],
]

wb = Workbook()
ws = wb.active
ws.title = "OpenJev Gemma4 E4B"
for row in rows:
    ws.append(row)

header_fill = PatternFill("solid", fgColor="D9EAF7")
header_font = Font(bold=True)
thin = Side(style="thin", color="DDDDDD")
border = Border(left=thin, right=thin, top=thin, bottom=thin)
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.border = border
for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=10):
    for cell in row:
        cell.border = border

# Set formats
ws["A2"].number_format = "@"
ws["B2"].number_format = "@"
ws["C2"].number_format = "0.000"
ws["D2"].number_format = "0"
ws["E2"].number_format = "0.00"
ws["F2"].number_format = "0.000000"
ws["G2"].number_format = "0.000000"
ws["H2"].number_format = "0.000"
ws["I2"].number_format = "0.00"

# Explanation sheet
info = wb.create_sheet("How the cost is calculated")
info.append(["Step", "Formula", "Value"])
info.append(["Latency used", "Measured latency for OpenJev native Gemma", "4.914 s"])
info.append(["Power draw", "Assumed request power", "100 W"])
info.append(["Energy per question", "watts × latency_seconds / 3,600,000", f"{(watts * latency_seconds / 3_600_000):.12f} kWh"])
info.append(["Cost per question", "energy_kwh × electricity_price_per_kwh", f"${cost_per_question:.12f}"])
info.append(["Cost per 10,000 questions", "cost_per_question × 10,000", f"${cost_10k:.3f}"])
info.append(["Cost per 1,000,000 questions", "cost_per_question × 1,000,000", f"${cost_1m:.2f}"])
info.append(["Source refs", "README_INFERENCE_BENCHMARK.md and README_INFERENCE_COSTS.md", "OpenJev native Gemma benchmark"])

for cell in info[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.border = border
for row in info.iter_rows(min_row=2, max_row=info.max_row, min_col=1, max_col=3):
    for cell in row:
        cell.border = border

# Resize columns
for sheet in [ws, info]:
    for col in sheet.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                value = str(cell.value)
                if len(value) > max_length:
                    max_length = len(value)
            except Exception:
                pass
        sheet.column_dimensions[column].width = min(max_length + 2, 40)

wb.save("gemma4_e4b_cost_per_question.xlsx")

csv_rows = [
    [
        "Backend",
        "Model",
        "Latency (s)",
        "Power draw (W)",
        "Electricity price ($/kWh)",
        "Energy per question (kWh)",
        "Cost per question ($)",
        "Cost per 10,000 questions ($)",
        "Cost per 1,000,000 questions ($)",
        "Notes",
    ],
    [
        "OpenJev native Gemma (experimental)",
        "gemma4-e4b-native",
        4.914,
        100,
        0.15,
        0.0001365,
        0.00002048,
        0.205,
        20.48,
        "Native Gemma via llama.cpp; from README_INFERENCE_BENCHMARK.md and README_INFERENCE_COSTS.md",
    ],
]
with open("gemma4_e4b_cost_per_question.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("Created gemma4_e4b_cost_per_question.xlsx and gemma4_e4b_cost_per_question.csv")
print(f"Cost per question: ${cost_per_question:.8f}")
print(f"Cost per 10,000: ${cost_10k:.3f}")
print(f"Cost per 1,000,000: ${cost_1m:.2f}")
