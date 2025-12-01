import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import csv
from sklearn.metrics import silhouette_score

# Folder setup
input_folder = 'Images'
base_output_folder = 'Final_output'
best_output_folder = os.path.join(base_output_folder, 'best')
os.makedirs(base_output_folder, exist_ok=True)
os.makedirs(best_output_folder, exist_ok=True)

Ks = [2, 3, 4, 5]

# Enhanced preprocessing options
USE_DENOISING = True
USE_CONTRAST_ENHANCEMENT = True
USE_BLUR = False
BLUR_KERNEL = (5, 5)

# Postprocessing options
USE_MORPHOLOGY = False  # Disabled for now
MORPH_KERNEL_SIZE = 5

# Test all feature types
FEATURE_TYPES = ['rgb', 'hsv', 'rgb+xy', 'hsv+xy']
NORMALIZE_FEATURES = False

# CSV to store all scores
csv_filename = 'all_outputs.csv'
csv_best_filename = 'best_outputs.csv'
all_results = []
best_results = {}  # Key: (image_file, K) -> dict(method, score, image)

def preprocess_image(image):
    """Apply preprocessing to improve segmentation quality"""
    processed = image.copy()
    
    if USE_DENOISING:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    
    if USE_CONTRAST_ENHANCEMENT:
        lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
    
    if USE_BLUR:
        processed = cv2.GaussianBlur(processed, BLUR_KERNEL, 0)
    
    return processed

def extract_features(image, feature_type):
    """Extract features based on feature type"""
    h, w, c = image.shape
    
    if feature_type == 'rgb':
        features = image.reshape(-1, 3)
    elif feature_type == 'hsv':
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        features = hsv.reshape(-1, 3)
    elif feature_type == 'rgb+xy':
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        X_norm = X.astype(np.float32) / w
        Y_norm = Y.astype(np.float32) / h
        features = np.concatenate([
            image.reshape(-1, 3).astype(np.float32),
            X_norm.reshape(-1, 1),
            Y_norm.reshape(-1, 1)
        ], axis=1)
    elif feature_type == 'hsv+xy':
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        X_norm = X.astype(np.float32) / w
        Y_norm = Y.astype(np.float32) / h
        features = np.concatenate([
            hsv.reshape(-1, 3).astype(np.float32),
            X_norm.reshape(-1, 1),
            Y_norm.reshape(-1, 1)
        ], axis=1)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    if NORMALIZE_FEATURES:
        features = features.astype(np.float32)
        for i in range(features.shape[1]):
            min_val = features[:, i].min()
            max_val = features[:, i].max()
            if max_val > min_val:
                features[:, i] = (features[:, i] - min_val) / (max_val - min_val)
    
    return features

# Find all valid image files
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(image_files)} images: {image_files}")
print(f"\nTesting all feature types: {FEATURE_TYPES}")
print(f"Preprocessing: Denoising={USE_DENOISING}, Contrast={USE_CONTRAST_ENHANCEMENT}, Blur={USE_BLUR}\n")

for feature_type in FEATURE_TYPES:
    output_folder = os.path.join(base_output_folder, feature_type)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Testing feature type: {feature_type.upper()}")
    print(f"{'='*60}")
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\nProcessing image {idx}/{len(image_files)}: {image_file}")
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Warning: {image_file} could not be read. Skipping.")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        
        # Preprocessing
        processed_image = preprocess_image(image)
        
        # Extract features
        features = extract_features(processed_image, feature_type)
        features = features.reshape(-1, features.shape[-1]).astype(np.float32)
        features = np.ascontiguousarray(features)
        
        for k in Ks:
            print(f"  - Running KMeans for K={k}...", end='', flush=True)
            
            # Try multiple initializations
            best_labels = None
            best_centers = None
            best_score = -1
            
            for init in range(3):
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
                _, labels, centers = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                try:
                    score = silhouette_score(features, labels.flatten())
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_centers = centers
                except:
                    if best_labels is None:
                        best_labels = labels
                        best_centers = centers
            
            labels = best_labels
            centers = best_centers
            
            # Reconstruct segmented image
            if feature_type in ['hsv', 'hsv+xy']:
                centers_hsv = centers[:, :3] if centers.shape[1] > 3 else centers
                if NORMALIZE_FEATURES:
                    centers_hsv[:, 0] = centers_hsv[:, 0] * 179
                    centers_hsv[:, 1:] = centers_hsv[:, 1:] * 255
                centers_hsv_uint8 = np.clip(centers_hsv, 0, 255).astype(np.uint8).reshape(1, -1, 3)
                centers_rgb = cv2.cvtColor(centers_hsv_uint8, cv2.COLOR_HSV2RGB)
                centers_rgb = centers_rgb.reshape(-1, 3)
            else:
                centers_rgb = centers[:, :3] if centers.shape[1] > 3 else centers
                if NORMALIZE_FEATURES:
                    centers_rgb = np.clip(centers_rgb * 255, 0, 255)
            
            centers_rgb = np.uint8(centers_rgb)
            segmented_data = centers_rgb[labels.flatten()]
            segmented_image = segmented_data.reshape(image.shape)
            
            # Save segmented image for current feature type
            base_name = os.path.splitext(image_file)[0]
            output_filename = f"{base_name}_segmented_K{k}.png"
            output_path = os.path.join(output_folder, output_filename)
            plt.imsave(output_path, segmented_image)
            
            # Calculate silhouette score
            sil_score = None
            try:
                sil_score = silhouette_score(features, labels.flatten())
                all_results.append([image_file, feature_type, k, sil_score])
                print(f" Done! Silhouette Score: {sil_score:.4f}")
            except Exception as e:
                print(f" Done! Could not calculate silhouette score: {e}")

            # Track best result per image & K
            key = (image_file, k)
            if sil_score is not None:
                current_best = best_results.get(key)
                if current_best is None or sil_score > current_best['score']:
                    best_results[key] = {
                        'score': sil_score,
                        'method': feature_type,
                        'image': segmented_image.copy()
                    }

# Save best images
print(f"\n{'='*60}")
print("Saving best results per image and K...")
print(f"{'='*60}")

best_records = []
for (image_file, k), data in best_results.items():
    base_name = os.path.splitext(image_file)[0]
    best_filename = f"{base_name}_best_K{k}.png"
    best_path = os.path.join(best_output_folder, best_filename)
    plt.imsave(best_path, data['image'])
    best_records.append([image_file, data['method'], k, data['score']])
    print(f"✓ Saved {best_filename} (method={data['method']}, score={data['score']:.4f})")

# Save all results to CSV
print(f"\n{'='*60}")
print(f"Saving all results to {csv_filename}...")
print(f"{'='*60}")

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'Feature_Type', 'K', 'Silhouette_Score'])
    writer.writerows(all_results)

print(f"\n✓ Successfully saved {len(all_results)} results to {csv_filename}")
print(f"✓ Results saved in folders: {[os.path.join(base_output_folder, ft) for ft in FEATURE_TYPES]}")

# Save best results summary
with open(csv_best_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'Best_Feature_Type', 'K', 'Silhouette_Score'])
    writer.writerows(best_records)

print(f"✓ Best results saved to {best_output_folder}")
print(f"✓ Best scores summary saved to {csv_best_filename}")

