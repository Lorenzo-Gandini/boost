import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def load_json_data(folder_path):
    """
    Load JSON files from a specified folder and return structured data.

    Parameters:
    - folder_path: Path to the folder containing JSON files.

    Returns:
    - A list of dictionaries with 'filename' and 'content' keys.
    """
    data_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    data_list.append({
                        "filename": filename,
                        "content": json_data
                    })
                except json.JSONDecodeError:
                    print(f"Error decoding JSON file: {filename}")

    return data_list

def format_json_data(json_data, title):
    """
    Format JSON data into a structured format with sections and detailed tables.
    """
    elements = []
    styles = getSampleStyleSheet()

    # Add title
    elements.append(Paragraph(title, styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Mapping for setting names
    setting_mapping = {
        "Setting_1": "Custom Setting",
        "Setting_2": "Suggested Setting"
    }

    for key, value in json_data.items():
        # Replace "Setting_1" and "Setting_2" with mapped names
        key_display = setting_mapping.get(key, key)

        # Add subsection title
        elements.append(Paragraph(f"<b>{key_display}</b>", styles['Heading3']))
        elements.append(Spacer(1, 8))

        if isinstance(value, dict):  # If the value is a dictionary
            # Create a table for the dictionary's content
            table_data = [["Metric", "Value"]]  # Table header
            for sub_key, sub_value in value.items():
                # Handle numeric values and others
                formatted_value = str(round(sub_value, 2)) if isinstance(sub_value, (int, float)) else str(sub_value)
                table_data.append([sub_key.replace('_', ' ').capitalize(), formatted_value])

            # Create and style the table
            table = Table(table_data, colWidths=[200, 300])
            table.setStyle(
                TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ])
            )
            elements.append(table)
            elements.append(Spacer(1, 12))

    return elements

def add_table_with_check(elements, table_data, max_rows_per_page=20):
    """
    Add table to the PDF, splitting it across multiple pages if necessary.
    """
    headers = table_data[0]  # Header row
    rows = table_data[1:]    # Data rows

    for i in range(0, len(rows), max_rows_per_page):
        chunk = [headers] + rows[i:i + max_rows_per_page]
        table = Table(chunk, colWidths=[200, 300])
        table.setStyle(
            TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ])
        )
        elements.append(table)
        elements.append(Spacer(1, 24))  # Add space after the table
        if i + max_rows_per_page < len(rows):  # Add a page break if more rows remain
            elements.append(PageBreak())

def generate_pdf(json_list, description, output_filename):
    """
    Generate a PDF with formatted JSON data.
    """
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Add description
    elements.append(Paragraph(description, styles['Title']))
    elements.append(Spacer(1, 24))

    # Add each JSON file's content
    for index, json_obj in enumerate(json_list, start=1):
        title = f"Data Set {index}: {json_obj['filename']}"
        elements.extend(format_json_data(json_obj['content'], title))
        if index < len(json_list):
            elements.append(PageBreak())

    doc.build(elements)

def generate_report(athlete):
    """
    Generate a detailed report for the athlete.
    """
    folder_path = f"output/{athlete}/stats/"
    json_list = load_json_data(folder_path)
    description = (
        f"This PDF contains detailed analysis and statistics for athlete {athlete}. "
        "The data is organized by key metrics and settings."
    )
    output_filename = f"output/{athlete}/stats_report.pdf"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    generate_pdf(json_list, description, output_filename)
    print(f"Report generated successfully: {output_filename}")

generate_report("Gandini Lorenzo")
