from PIL import Image
import numpy as np

def detect_dominant_color(image_path):
    image = Image.open(image_path)
    image = image.resize((50, 50))
    pixels = np.array(image).reshape(-1, 3)
    avg_color = np.mean(pixels, axis=0)
    r, g, b = map(int, avg_color)
    
    if r > 200 and g < 100 and b < 100:
        return "red"
    elif g > 200 and r < 100:
        return "green"
    elif b > 200 and r < 100:
        return "blue"
    elif r > 200 and g > 200 and b < 100:
        return "yellow"
    elif r > 180 and g > 180 and b > 180:
        return "white"
    elif r < 80 and g < 80 and b < 80:
        return "black"
    else:
        return "neutral"

