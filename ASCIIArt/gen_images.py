from tqdm import tqdm
import time
import argparse
import json
import hashlib
import requests


def get_sha1(image):
    bytes = image
    readable_hash = hashlib.sha1(bytes).hexdigest()
    return readable_hash

def save_image(image, filename):
    with open(filename, 'wb') as f:
        f.write(image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('api_key_filepath', type=str, help='Path to the file containing the API key')
    # parser.add_argument('prompt', type=str, help='Prompt for the image generation')
    parser.add_argument("-r", "--resume")
    args = parser.parse_args()

    with open(args.api_key_filepath, 'r') as f:
        api_key = f.read().strip()

    with open("prompts.json") as infile:
        prompts = json.load(infile)

    if args.resume:
        images = json.load("images.json")
    else:
        images = {}
        
    for prompt in tqdm(prompts):
        if '|'.join([prompt["subject"], prompt["style"], prompt["prompt"]]) in images:
            continue
        headers = {
            'authorization': api_key,
            'accept': 'image/*'
        }

        data = {
            'prompt': prompt["prompt"],
            'aspect_ratio': '1:1',
            'mode': 'text-to-image',
            'model': 'sd3',
            'output_format': 'png'
        }

        files = {'none': None}  # The API expects at least one file to be sent

        response = requests.post('https://api.stability.ai/v2beta/stable-image/generate/sd3', headers=headers, files=files, data=data)
        response.raise_for_status()

        image = response.content
        sha1_hash = get_sha1(image)
        filename = f'{sha1_hash}.png'
        save_image(image, filename)
        images['|'.join([prompt["subject"], prompt["style"], prompt["prompt"]])] = sha1_hash
        if len(images) % 20 == 0:
            with open("images.json.wip", "w") as outfile:
                json.dump(images, outfile)
        time.sleep(1)
    with open("images.json", "w") as outfile:
        json.dump(images, outfile)
                
if __name__ == '__main__':
    main()

