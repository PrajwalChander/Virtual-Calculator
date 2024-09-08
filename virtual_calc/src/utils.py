import cv2
import numpy as np
import os

def resize_image(image, width=None, height=None):
    """
    Resize an image to the given width and/or height while maintaining aspect ratio.
    
    Args:
        image (numpy.ndarray): The input image.
        width (int, optional): The desired width. Defaults to None.
        height (int, optional): The desired height. Defaults to None.
        
    Returns:
        numpy.ndarray: The resized image.
    """
    if width is None and height is None:
        return image

    (h, w) = image.shape[:2]
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def save_image(image, path):
    """
    Save an image to a given path.
    
    Args:
        image (numpy.ndarray): The image to save.
        path (str): The path to save the image.
    """
    cv2.imwrite(path, image)

def load_image(path):
    """
    Load an image from a given path.
    
    Args:
        path (str): The path to the image file.
        
    Returns:
        numpy.ndarray: The loaded image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")
    return cv2.imread(path)

def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
              font_scale=1, color=(0, 0, 0), thickness=2):
    """
    Draw text on an image.
    
    Args:
        image (numpy.ndarray): The image to draw text on.
        text (str): The text to draw.
        position (tuple): The position to draw the text (x, y).
        font (int, optional): The font type. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): The scale of the font. Defaults to 1.
        color (tuple, optional): The color of the text in BGR. Defaults to (0, 0, 0).
        thickness (int, optional): The thickness of the text. Defaults to 2.
    """
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def preprocess_image(image):
    """
    Preprocess an image for input into the model.
    
    Args:
        image (numpy.ndarray): The image to preprocess.
        
    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image to the required size for the model
    resized = cv2.resize(gray, (224, 224))
    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0
    # Expand dimensions to match input shape (1, 224, 224, 1)
    preprocessed = np.expand_dims(normalized, axis=(0, -1))
    return preprocessed
