from markdown_pdf import MarkdownPdf
from markdown_pdf import Section

pdf = MarkdownPdf(toc_level=2)

with open("report.md", "r", encoding="utf-8") as f:
    text = f.read()

pdf.add_section(Section(text))
pdf.save("report/HW3_Report.pdf")
print("PDF generated successfully.")
