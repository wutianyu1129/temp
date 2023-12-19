from swfparse import SWFFile
import fitz  # PyMuPDF
import os

# File paths
swf_file_path = '/mnt/data/charpter1.swf'
pdf_file_path = '/mnt/data/charpter1_converted.pdf'

# Check if the SWF file contains images
swf_file = SWFFile(open(swf_file_path, 'rb'))
images = swf_file.extract_images()

# Create a PDF document
pdf_document = fitz.open()

# Add each image to the PDF document
for img in images:
    img_path = f'/mnt/data/{img.name}'
    with open(img_path, 'wb') as f:
        f.write(img.bytes)

    # Insert the image into the PDF
    pdf_page = pdf_document.new_page(width=595, height=842)  # A4 size page
    pdf_page.insert_image(pdf_page.rect, filename=img_path)

    # Remove the temporary image file
    os.remove(img_path)

# Save the PDF document
pdf_document.save(pdf_file_path)
pdf_document.close()

pdf_file_path
