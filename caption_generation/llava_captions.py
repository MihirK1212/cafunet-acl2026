import base64
import requests
import json


class LlavaCaptionGeneration():
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.model = "llava"
        self.api_url="http://localhost:11434/api/generate"
        self.prompt = "Provide an informative, not too long, not too short, caption for this image"


    def encode_image_to_base64(self):
        # Open the image file and encode it to base64
        with open(self.image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    def stream_caption_from_api(self) -> str:
        base64_image = self.encode_image_to_base64()

        # Prepare the request payload
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [base64_image]
        }
        
        # Send the request using requests library and stream the response
        response = requests.post(self.api_url, json=payload, stream=True)
        
        # Parse the streaming response
        caption = ""
        for line in response.iter_lines():
            if line:
                # Load the line as JSON
                json_line = json.loads(line.decode('utf-8'))
                # Append the 'response' part to the caption
                caption += json_line.get("response", "")
                # If the stream is done, stop
                if json_line.get("done"):
                    break
        return caption 
    
    def get_caption(self) -> str:
        try:
            return self.stream_caption_from_api()
        except Exception as E:
            print("ERROR Could not Generate Caption", E)
            return "ERROR Could not Generate Caption"