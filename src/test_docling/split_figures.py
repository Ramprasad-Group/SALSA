from pathlib import Path

# Core Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem, TextItem, DoclingDocument # For advanced features
import matplotlib.pyplot as plt

def inspect_pictures_with_images(doc: DoclingDocument, image_size=(6, 4)):
    """Display pictures inline with their text content"""
    for idx, picture in enumerate(doc.pictures):
        print(f"\n{'='*60}")
        print(f"Picture {idx}")
        print(f"{'='*60}")

        # Display the image
        try:
            img = picture.get_image(doc)
            if img:
                plt.figure(figsize=image_size)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Picture {idx}")
                plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")

        # Display metadata
        caption = picture.caption_text(doc)
        if caption:
            print(f"\nCaption: {caption}")

        if hasattr(picture, 'prov') and picture.prov:
            print(f"Location: Page {picture.prov[0].page_no}")

        # Display embedded text
        print("\nEmbedded text elements:")
        text_found = False
        for item, level in doc.iterate_items(root=picture, traverse_pictures=True):
            if isinstance(item, TextItem):
                print(f"{'  ' * (level + 1)}- {item.text}")
                text_found = True

        if not text_found:
            print("  (No text elements found)")

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

test_pdf = "/home/wschertzer3/AEM_OCR/src/test_docling/c5ta01420d.pdf"

# # Simple conversion

# # Create a converter instance
# converter = DocumentConverter()

# # Convert a document
# result = converter.convert(test_pdf)
# doc = result.document

# # Document metadata - important for tracking
# print(f"Document title: {doc.name}")
# print(f"Number of pages: {len(doc.pages)}")
# print(f"Number of tables: {len(doc.tables)}")
# print(f"Number of pictures: {len(doc.pictures)}")

# Configure pipeline to extract images
IMAGE_RESOLUTION_SCALE = 2.0  # 2x resolution (144 DPI)

pipeline_options = PdfPipelineOptions(
    images_scale=IMAGE_RESOLUTION_SCALE,
    generate_page_images=True,
    generate_picture_images=True,
)

converter_with_images = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Convert with image extraction
img_result = converter_with_images.convert(test_pdf)
img_doc = img_result.document

# Create images subdirectory
images_dir = output_dir / "images"
images_dir.mkdir(exist_ok=True)

# Save page images
for page_no, page in img_doc.pages.items():
    page_image_filename = images_dir / f"page_{page_no}.png"
    with page_image_filename.open("wb") as fp:
        page.image.pil_image.save(fp, format="PNG")

print(f"Saved {len(img_doc.pages)} page images")



# Extract all figures and tables as separate images
table_counter = 0
picture_counter = 0

for element, level in img_doc.iterate_items():
    if isinstance(element, TableItem):
        table_counter += 1
        image_filename = images_dir / f"table_{table_counter}.png"
        with image_filename.open("wb") as fp:
            element.get_image(img_doc).save(fp, "PNG")

    elif isinstance(element, PictureItem):
        picture_counter += 1
        image_filename = images_dir / f"figure_{picture_counter}.png"
        with image_filename.open("wb") as fp:
            element.get_image(img_doc).save(fp, "PNG")

print(f"Extracted {table_counter} tables and {picture_counter} figures as images")


# Use the simple inline display
# inspect_pictures_with_images(img_doc)


# # Configure enrichment pipeline

# prompt = "Given a paper as context, extract the final polymer structure of C18BQAPPO-1. " \
#         f"If there is no reaction, respond 'No Reaction Observed'. Here is the paper: {md_out[8600:14000]}"

# print(prompt)

# enrichment_options = PdfPipelineOptions(
#     do_picture_description=True,
#     picture_description_options=PictureDescriptionVlmOptions(  # model & prompt choice
#         repo_id="ibm-granite/granite-vision-3.1-2b-preview",
#         # prompt="Give a detailed description of what is depicted in the image",
#         prompt=prompt,
#     ),
#     generate_picture_images=True,
#     images_scale=2.0,
# )

# converter_enriched = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(pipeline_options=enrichment_options)
#     }
# )

# enr_result = converter_enriched.convert(aem_paper)
# enr_doc = enr_result.document

# # print(enr_doc)

# html_buffer = []
# # display pic 5 pictures, captions and annotations:
# for pic in enr_doc.pictures:
#     html_item = (
#         f"<h3>Picture <code>{pic.self_ref}</code></h3>"
#         # f'<img src="{pic.image.uri!s}" /><br />'
#         f"<h4>Caption</h4>{pic.caption_text(doc=doc)}<br />"
#     )
#     for annotation in pic.annotations:
#         if not isinstance(annotation, PictureDescriptionData):
#             continue
#         html_item += (
#             f"<h4>Annotations ({annotation.provenance})</h4>{annotation.text}<br />\n"
#         )
#     html_buffer.append(html_item)
# print(html_buffer)

