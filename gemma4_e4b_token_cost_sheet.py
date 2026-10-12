from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side
import csv

latency_seconds = 4.914
watts = 100
price_per_kwh = 0.15

# Base electricity-only estimate for the native Gemma 4 E4B route.
# cost_per_question = watts * latency / 3_600_000 * electricity_price
cost_per_question = (watts * latency_seconds / 3_600_000) * price_per_kwh

# Example token assumptions for presentation. These are not cloud billing values.
# If one native request processes 1,000 tokens, then:
# cost_per_token = cost_per_question / tokens_per_question
for_tokens = {
    250: cost_per_question / 250,
    500: cost_per_question / 500,
    1000: cost_per_question / 1000,
    2000: cost_per_question / 2000,
    5000: cost_per_question / 5000,
    10000: cost_per_question / 10000,
}

rows = [
    ["Measurement", "Value", "Formula / note"],
    ["Latency (s)", latency_seconds, "Measured OpenJev native Gemma 4 E4B latency"],
    ["Power draw (W)", watts, "Assumed local system draw while processing"],
    ["Electricity price ($/kWh)", price_per_kwh, "Illustrative local tariff"],
    ["Energy per question (kWh)", (watts * latency_seconds / 3_600_000), "watts × latency_seconds / 3,600,000"],
    ["Cost per question ($)", cost_per_question, "energy_per_question × electricity_price"],
    ["Cost per 250 tokens ($)", for_tokens[250], "Example: 250 tokens/request"],
    ["Cost per 500 tokens ($)", for_tokens[500], "Example: 500 tokens/request"],
    ["Cost per 1,000 tokens ($)", for_tokens[1000], "Example: 1,000 tokens/request"],
    ["Cost per 2,000 tokens ($)", for_tokens[2000], "Example: 2,000 tokens/request"],
    ["Cost per 5,000 tokens ($)", for_tokens[5000], "Example: 5,000 tokens/request"],
    ["Cost per 10,000 tokens ($)", for_tokens[10000], "Example: 10,000 tokens/request"],
    ["Notes", "Local inference has no cloud API fee. These are electricity-only estimates.", "Not a billing quote; excludes startup, downloads, hardware, and maintenance."],
]

wb = Workbook()
ws = wb.active
ws.title = "Gemma4 E4B token cost"
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
for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=3):
    for cell in row:
        cell.border = border

for row in range(2, ws.max_row + 1):
    ws[f"B{row}"].number_format = "0.000000"

# Write CSV with same numbers
csv_rows = [
    ["Measurement", "Value", "Formula / note"],
    ["Latency (s)", 4.914, "Measured OpenJev native Gemma 4 E4B latency"],
    ["Power draw (W)", 100, "Assumed local system draw while processing"],
    ["Electricity price ($/kWh)", 0.15, "Illustrative local tariff"],
    ["Energy per question (kWh)", 0.0001365, "watts × latency_seconds / 3,600,000"],
    ["Cost per question ($)", 0.00002047, "energy_per_question × electricity_price"],
    ["Cost per 250 tokens ($)", 0.0000000819, "Example: 250 tokens/request"],
    ["Cost per 500 tokens ($)", 0.0000000409, "Example: 500 tokens/request"],
    ["Cost per 1,000 tokens ($)", 0.0000000205, "Example: 1,000 tokens/request"],
    ["Cost per 2,000 tokens ($)", 0.0000000102, "Example: 2,000 tokens/request"],
    ["Cost per 5,000 tokens ($)", 0.0000000041, "Example: 5,000 tokens/request"],
    ["Cost per 10,000 tokens ($)", 0.0000000020, "Example: 10,000 tokens/request"],
    ["Notes", "Local inference has no cloud API fee. These are electricity-only estimates.", "Not a billing quote; excludes startup, downloads, hardware, and maintenance."],
]

wb.save("gemma4_e4b_token_cost.xlsx")
with open("gemma4_e4b_token_cost.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("Created gemma4_e4b_token_cost.xlsx and gemma4_e4b_token_cost.csv")
print(f"Cost per question = ${cost_per_question:.8f}")
print(f"Cost per 1,000 tokens (example assumption) = ${for_tokens[1000]:.10f}")
