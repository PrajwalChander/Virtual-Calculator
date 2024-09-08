import google.generativeai as genai
from PIL import Image
import cv2

class GeminiAPI:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def image_to_pil(self, image):
        # Convert OpenCV image (numpy array) to PIL image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        return pil_image

    def solve(self, image):
        pil_image = self.image_to_pil(image)
        prompt = ("Analyze the given image and solve any mathematical equations present. "
                  "If there is an equals sign, provide the result. "
                  "If there is a question mark or an incomplete part of the equation, provide the missing value or complete the equation."
                  "Display the final answer only.Dont give a lot of other text.")

        # Here, we assume that the generate_content function accepts both text and image
        response = self.model.generate_content([prompt, pil_image])
        
        # Handle the response properly
        if hasattr(response, 'text'):
            return response.text
        else:
            return "Error: Unable to retrieve response text"
