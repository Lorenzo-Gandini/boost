import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json

def generate_pdf(json_list, description, output_filename):
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter

    # Add description text
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 50, description)

    # Add JSON content
    y_position = height - 100
    for json_obj in json_list:
        json_str = json.dumps(json_obj, indent=4)
        for line in json_str.split('\n'):
            if y_position < 50:
                c.showPage()
                y_position = height - 50
                c.setFont("Helvetica", 12)
            c.drawString(100, y_position, line)
            y_position -= 15

    c.save()

def generate_report(athlete):
    # Example usage
    json_list = [
        {"name": "John Doe", "age": 30, "city": "New York"},
        {"name": "Jane Smith", "age": 25, "city": "Los Angeles"}
    ]
    description = "This PDF contains a list of JSON objects with user information."
    
    # Build the output folder and file path
    output_folder = os.path.join("output", athlete)
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    output_filename = os.path.join(output_folder, "report.pdf")

    generate_pdf(json_list, description, output_filename)