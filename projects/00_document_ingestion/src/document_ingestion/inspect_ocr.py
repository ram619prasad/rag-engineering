from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

image_dir = Path(
    "/Users/sidvik/Learning/rag-engineering/datasets/sample/extracted_images"
)

for image_path in sorted(image_dir.glob("page3_image_*")):
    print(f"\n--- OCR: {image_path.name} ---")

    with Image.open(image_path) as image:
        image = image.convert("L")
        image = image.resize(
            (image.width * 2, image.height * 2)
        )
        image = ImageEnhance.Contrast(image).enhance(2)
        image = image.filter(ImageFilter.SHARPEN)

        text = pytesseract.image_to_string(
            image,
            config="--psm 11"
        )

        print(text.strip() or "[No text recognized]")