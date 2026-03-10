import os
from jinja2 import Template

# WKHTMLTOPDF_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
# config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)


def save_html_report(template_path, output_path, context):
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    # Add base path for images
    context["base_path"] = os.path.abspath("reports/outputs")

    html = template.render(context)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] HTML report saved: {output_path}")

