import base64
from PIL import Image
import io
import os
import json

def decode_comfyui_json_and_save_image(json_filepath, output_filename="comfyui_generated_image.png"):
    """
    Reads a ComfyUI JSON response file, extracts the base64 image string, decodes it, and saves it as an image file.

    Args:
        json_filepath (str): The path to the input JSON file.
        output_filename (str): The name for the output image file.
    """
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        
        # Extract the base64 string from the ComfyUI response structure
        # Handles both: data['output']['message'] (Runpod API format) and data['output']['images'][0]['data']
        base64_url = None
        
        if 'output' in data:
            if 'message' in data['output']:
                # Runpod API format: contains data:image/png;base64, prefix
                base64_url = data['output']['message']
            elif 'images' in data['output'] and len(data['output']['images']) > 0:
                # Alternative format with images array
                base64_url = data['output']['images'][0]['data']

        if not base64_url:
            print("Error: Image data not found in the JSON output.")
            return

        # Remove data URI prefix if present
        if "," in base64_url:
            _, encoded_data = base64_url.split(",", 1)
        else:
            encoded_data = base64_url

        # Decode base64 to bytes
        image_data = base64.b64decode(encoded_data)
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        image.save(output_filename)

        print(f"ComfyUI image successfully saved as '{output_filename}'")
        print(f"Image path: {os.path.abspath(output_filename)}")

    except FileNotFoundError:
        print(f"Error: The file '{json_filepath}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_filepath}'.")
    except base64.binascii.Error as e:
        print(f"Error decoding base64 string: {e}")
        print("Please ensure the input is a valid base64 string.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Process the comfyui_output.json file
decode_comfyui_json_and_save_image("comfyui_output.json", "comfyui_generated_image.png")