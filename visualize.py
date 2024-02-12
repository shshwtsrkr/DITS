import os
import csv
from enhancement import run_realesrgan
from spotting import run_on_images
import numpy as np
import shutil
import torch

input_dir = "inputs"
output_dir = "outputs"
processed_dir = "processed"
threshold_confidence = 1.0
max_iterations = 5
os.makedirs(processed_dir, exist_ok=True)

best_outputs = []
image_confidences = []

for i, filename in enumerate(os.listdir(input_dir)):
    input_file = os.path.join(input_dir, filename)
    base_filename, file_extension = os.path.splitext(filename)

    previous_confidence = run_on_images(input_file, output_dir)
    previous_output = os.path.join(output_dir, f"{base_filename}_out{file_extension}")
    best_output = previous_output
    best_confidence = previous_confidence
    final_confidences = ','.join(map(str, best_confidence))
    image_confidences.append((filename, final_confidences))
    print("-------------------------------------Clearing torch cache----------------------------------------")
    torch.cuda.empty_cache()

# Write image names and final confidences to a CSV file
with open('results/image_confidences_deepsolo.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image Name", "Confidences"])  # Write header
    writer.writerows(image_confidences)
