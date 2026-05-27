from fpdf import FPDF
from datetime import datetime
import os

# Define paths
plots_path = "C:/Users/heave/Videos/data/Sales Walmart/walmart-recruiting-store-sales-forecasting/plots/"
output_pdf = os.path.join(plots_path, "dashboard_report.pdf")

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Retail Sales Forecasting Dashboard Report", ln=True, align="C")
pdf.ln(10)

# Date
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, f"Generated on: {datetime.today().strftime('%Y-%m-%d')}", ln=True)

# Description
pdf.multi_cell(0, 10, "This report provides an overview of a retail sales forecasting dashboard built with Streamlit and powered by an XGBoost model. It covers evaluation metrics, visual insights, and features used to predict weekly sales for Walmart.")

# Model Evaluation
pdf.ln(5)
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "1. Model Evaluation Metrics:", ln=True)
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, "• Mean Absolute Error (MAE): $12,094.57", ln=True)
pdf.cell(0, 10, "• Root Mean Squared Error (RMSE): $18,765.31", ln=True)

# Add Plots
def add_plot(title, filename):
    pdf.ln(8)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, title, ln=True)
    pdf.image(os.path.join(plots_path, filename), w=180)

# Section 2: Visual Insights
pdf.ln(5)
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "2. Visual Insights", ln=True)

add_plot("Feature Importance", "feature_importance.png")
add_plot("Store-Level Sales Comparison", "store_sales_comparison.png")
add_plot("Prediction Residual Distribution", "residual_distribution.png")
add_plot("Sales by Store Type", "sales_by_store_type.png")
add_plot("Total Weekly Sales Over Time", "totalweeklysalesovertime.png")
add_plot("Holiday vs Non-Holiday Sales", "sales_by_holiday_nonholiday1.png")
add_plot("Correlation Matrix", "Matrix.png")

# Footer
pdf.ln(10)
pdf.set_font("Arial", "I", 10)
pdf.cell(0, 10, "End of Report", align="C")

# Save PDF
pdf.output(output_pdf)
print(f"PDF report saved to: {output_pdf}")
