import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import json

def load_json_files(folder_path):
    json_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                json_list.append(json.load(f))
    return json_list

def generate_pdf(json_list, description, output_filename):
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Add description
    elements.append(Paragraph(description, styles['Title']))
    elements.append(Spacer(1, 12))

    # Add JSON content
    for json_obj in json_list:
        json_str = json.dumps(json_obj, indent=4)
        elements.append(Paragraph(json_str, styles['BodyText']))
        elements.append(Spacer(1, 12))

    doc.build(elements)


def generate_report(athlete):
    folder_path = f"output/{athlete}/stats/"
    json_list = load_json_files(folder_path)
    description = f"This PDF contains a list of JSON objects with statistics for athlete {athlete}."
    output_filename = f"output/{athlete}/stats_report.pdf"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    generate_pdf(json_list, description, output_filename)
    print(f"Report generated successfully: {output_filename}")
