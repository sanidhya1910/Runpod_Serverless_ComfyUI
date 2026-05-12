import requests
import json
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from decode_comfyui_image import decode_comfyui_json_and_save_image

load_dotenv()

ENDPOINT_ID = "og3rnr9mee0yom"
API_KEY = os.getenv("RUNPOD_API_KEY")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
WORKFLOW_FILE = "comfyui_flux_workflow.json"
OUTPUT_JSON_FILE = "comfyui_output.json"
OUTPUT_IMAGE_FILE = "comfyui_generated_image.png"
POLL_INTERVAL = 5 
MAX_WAIT_TIME = 1800 
def load_workflow(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Workflow file '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{filepath}'.")
        sys.exit(1)

def update_prompt_in_workflow(workflow, prompt):
    try:
        # Update the positive prompt in node 6
        workflow['input']['workflow']['6']['inputs']['text'] = prompt
        return workflow
    except KeyError:
        print("Error: Could not find the prompt node in the workflow.")
        sys.exit(1)

def submit_job(workflow):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    try:
        print("Submitting job to Runpod Serverless...")
        response = requests.post(
            f"{BASE_URL}/run",
            headers=headers,
            json=workflow,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        job_id = result.get('id')
        status = result.get('status')
        
        if not job_id:
            print("Error: No job ID received from the API.")
            print(f"Response: {result}")
            sys.exit(1)
        
        print(f"  Job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Status: {status}")
        
        return job_id
    
    except requests.exceptions.RequestException as e:
        print(f"Error submitting job: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        sys.exit(1)

def poll_job_status(job_id):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    start_time = time.time()
    poll_count = 0
    
    print("\nMonitoring job progress...")
    
    while True:
        elapsed_time = time.time() - start_time
        
        if elapsed_time > MAX_WAIT_TIME:
            print(f"\nError: Job did not complete within {MAX_WAIT_TIME} seconds (30 minutes).")
            print("Results are automatically deleted after 30 minutes.")
            sys.exit(1)
        
        try:
            response = requests.get(
                f"{BASE_URL}/status/{job_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            status = result.get('status')
            
            poll_count += 1
            
            if status == "IN_QUEUE":
                print(f"  [{poll_count}] Status: IN_QUEUE - Waiting for processing...")
            elif status == "IN_PROGRESS":
                delay_time = result.get('delayTime', 0)
                print(f"  [{poll_count}] Status: IN_PROGRESS - Processing (queue wait: {delay_time}ms)...")
            elif status == "COMPLETED":
                execution_time = result.get('executionTime', 0)
                delay_time = result.get('delayTime', 0)
                print(f"\n  Job completed!")
                print(f"  Execution time: {execution_time}ms")
                print(f"  Queue delay: {delay_time}ms")
                print(f"  Total time: {int(elapsed_time)}s")
                return result
            elif status == "FAILED":
                print(f"\nError: Job failed!")
                print(f"Response: {result}")
                sys.exit(1)
            else:
                print(f"  [{poll_count}] Status: {status}")
            
            time.sleep(POLL_INTERVAL)
        
        except requests.exceptions.RequestException as e:
            print(f"Error checking job status: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            sys.exit(1)

def save_output_json(result, filepath):
    """Save the full job result to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Response saved to '{filepath}'")
        return filepath
    except Exception as e:
        print(f"Error saving output JSON: {e}")
        sys.exit(1)

def generate_image(prompt, output_image_name=None):
    """Main function to generate an image from a prompt."""
    
    # Validate API key
    if not API_KEY:
        print("Error: RUNPOD_API_KEY not found in .env file.")
        print("Please create a .env file with: RUNPOD_API_KEY=your_actual_api_key")
        sys.exit(1)
    
    print("=" * 60)
    print("ComfyUI FLUX Image Generator")
    print("=" * 60)
    print(f"\nPrompt: {prompt}")
    print(f"Endpoint: {BASE_URL}/run")
    
    # Load and update workflow
    workflow = load_workflow(WORKFLOW_FILE)
    workflow = update_prompt_in_workflow(workflow, prompt)
    
    # Submit job
    job_id = submit_job(workflow)
    
    # Poll for completion
    result = poll_job_status(job_id)
    
    # Save output JSON
    save_output_json(result, OUTPUT_JSON_FILE)
    
    # Decode and save image
    image_file = output_image_name or OUTPUT_IMAGE_FILE
    print(f"\nDecoding image...")
    decode_comfyui_json_and_save_image(OUTPUT_JSON_FILE, image_file)
    

    print("Image generation complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use command-line argument as prompt
        prompt = " ".join(sys.argv[1:])
        output_image = sys.argv[2] if len(sys.argv) > 2 else None
        generate_image(prompt, output_image)
    else:
        # Interactive prompt
        print("ComfyUI FLUX Image Generator")
        print("-" * 60)
        prompt = input("Enter your image prompt: ").strip()
        
        if not prompt:
            print("Error: Prompt cannot be empty.")
            sys.exit(1)
        
        output_image = input("Enter output filename (press Enter for default 'comfyui_generated_image.png'): ").strip()
        
        generate_image(prompt, output_image if output_image else None)
