from models.google_api import GoogleAPI

def capture_and_send_image(canvas):
    response = GoogleAPI().generate_content(canvas)
    return response
