import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.metrics import silhouette_score

# Folder setup
input_folder = 'Images'
output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)

# K values for segmentation
Ks = [2, 3, 4, 5]

# finding les images
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(image_files)} images: {image_files}")

for idx, image_file in enumerate(image_files, 1):
    print(f"\nProcessing image {idx}/{len(image_files)}: {image_file}")
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Warning: {image_file} could not be read. Skipping.")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # resiziha kima habit (it will take so long on large images hhh)
    pixel_vals = image.reshape((-1, 3)).astype(np.float32)

    for k in Ks:
        print(f"  - Running KMeans for K={k}...", end='', flush=True)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)

        
        base_name = os.path.splitext(image_file)[0]
        output_filename = f"{base_name}_segmented_K{k}.png"
        output_path = os.path.join(output_folder, output_filename)
        plt.imsave(output_path, segmented_image)

        # silhouette score
        try:
            sil_score = silhouette_score(pixel_vals, labels.flatten())
            print(f" Done! Saved as {output_filename} | Silhouette Score: {sil_score:.4f}")
        except Exception as e:
            print(f" Done! Could not calculate silhouette score: {e}")
