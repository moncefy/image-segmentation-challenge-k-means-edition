# Image Segmentation Challenge – K-Means Edition

## What does this do?
This project uses K-Means clustering to automatically segment (separate) different objects or regions in images based on color. The script loads your images, resizes them, and applies K-Means clustering to group similar pixels together for K=2, 3, 4, and 5 (number of clusters). For each setting, it saves a segmented result image.

## How is the score computed?
The script calculates the **silhouette score** for each segmentation result. The silhouette score measures how well pixels are clustered, with values close to 1 meaning best separation and values near 0 meaning poor clustering. You will see the silhouette scores printed in the terminal for each image and K value.

## Required Libraries
- numpy
- opencv-python
- matplotlib
- scikit-learn

You can copy the following command and paste it in your terminal to install all required libraries:
```
pip install numpy opencv-python matplotlib scikit-learn
```

## How to Run
1. Place your images in the `Images` folder (JPG/PNG format).
2. The script will attempt to create its output folders automatically (e.g., `Final_output`). **If you encounter any errors or due to OS limitations, you may need to manually create these folders before running.**
3. Open a terminal in this project directory.
4. Execute the main segmentation script:
   ```
   python segment_all_features.py
   ```
5. `segment_all_features.py` runs RGB, HSV, RGB+XY, HSV+XY (with preprocessing) for each image and K; saves all results under `Final_output/<method>` and stores the best-scoring segmentation per (image, K) in `Final_output/best`.
6. Silhouette scores:
   - All runs: `all_outputs.csv`
   - Best-only summary: `best_outputs.csv`
