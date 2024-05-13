import argparse
import subprocess
import json
import os

def convert_image_to_ascii(filepath):
    command = f"ascii-image-converter {filepath} -f"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if error:
        print(f"Error: {error}")
        return None
    return output.decode()

def convert_images(json_file, output_file):
    with open(json_file, "r") as f:
        images = json.load(f)

    conversions = {}
    for img in images:
        filename = img["filename"]
        filepath = os.path.join(args.prefix, filename)
        ascii_art = convert_image_to_ascii(filepath)
        if ascii_art:
            conversions[filename] = ascii_art

    with open(output_file, "w") as f:
        json.dump(conversions, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to ASCII art.")
    parser.add_argument("prefix", help="The path prefix for the image files.")
    args = parser.parse_args()

    convert_images("train.json", "train_aic_conversions.json")
    convert_images("val.json", "val_aic_conversions.json")
