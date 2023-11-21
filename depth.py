import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the MiDaS large model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.to('cpu')
midas.eval()

# Input transformation pipeline for MiDaS large
transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

# Directory containing the images
input_dir = 'enter path to your input images'
output_dir = 'enter path for output folder'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize a dictionary to store depth data
depth_data = {}

# Get list of all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process and display each image and its depth map
for file_name in image_files:
    img_path = os.path.join(input_dir, file_name)

    # Read and convert the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the original image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Transform the image for MiDaS
    input_batch = transform(img_rgb).to('cpu')

    # Predict the depth
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

    # Compute depth statistics
    avg_depth = np.mean(output)
    max_depth = np.max(output)
    min_depth = np.min(output)

    # Store the depth data
    depth_data[file_name] = {
        'average_depth': avg_depth,
        'maximum_depth': max_depth,
        'minimum_depth': min_depth
    }

    # Display the depth map
    plt.subplot(1, 2, 2)
    plt.imshow(output, cmap='inferno')
    plt.title('Depth Map')
    plt.axis('off')

    plt.show()

    # Print depth statistics
    print(f"Depth Data:")
    print(f"  Average Depth: {avg_depth}")
    print(f"  Maximum Depth: {max_depth}")
    print(f"  Minimum Depth: {min_depth}")
    print("\n")

# Save the depth data to a file
output_data_path = os.path.join(output_dir, 'depth_data.txt')
with open(output_data_path, 'w') as file:
    for key, value in depth_data.items():
        file.write(f'{key}: {value}\n')

print("Depth data saved in:", output_data_path)
