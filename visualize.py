import os
import csv
from enhancement import run_realesrgan
from spotting import run_on_images
import numpy as np
import shutil
import torch
# Setup directories - You can change the directory names as per use
input_dir = "inputs"
output_dir = "results"
enhanced_dir = "processed"
os.makedirs(enhanced_dir, exist_ok=True)

image_confidences = []

for i, filename in enumerate(os.listdir(input_dir)):
    input_file = os.path.join(input_dir, filename)
    base_filename, file_extension = os.path.splitext(filename)
    # Define the output file name for the enhanced image
    enhanced_output_file = os.path.join(enhanced_dir, f"{base_filename}_enhanced{file_extension}")
    # Enhance the image before processing
    run_realesrgan(input_file, enhanced_dir, f"{base_filename}_enhanced{file_extension}")
    # Now process the enhanced image
    previous_confidence = run_on_images(enhanced_output_file, output_dir)
    print("-------------------------------------Clearing torch cache----------------------------------------")
    torch.cuda.empty_cache()

