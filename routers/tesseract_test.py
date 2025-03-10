import pytesseract
from PIL import Image, ImageDraw, ImageFont

# Create a blank white image
width, height = 400, 100
image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)

# Draw some text on the image
text_to_draw = "Hello!, pytesseract!"
# Use a basic font; you can specify a ttf file if desired
draw.text((10, 30), text_to_draw, fill='black')

# Save the image to disk (optional)
image.save("test_image.png")

# Now use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(image)

print("Extracted Text:")
print(extracted_text)
