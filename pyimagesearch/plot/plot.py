import matplotlib.pyplot as plt
import os

def plot_classes(main_folder):
    # Get the list of subdirectories

    subdirs = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]

    # Initialize a dictionary to store the count of images per class
    class_counts = {}

    # Iterate through each subdir and count the number of images
    for subdir in subdirs:
        class_name = subdir
        subdir_path = os.path.join(main_folder, subdir)
        # Count the number of files with common image extensions (e.g., jpg, png)
        num_images = len([f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = num_images

    if "train" in main_folder:
        split = "Train"
    elif "val" in main_folder:
        split = "Validation"
    else:
        split = "Test"

    # Plotting the histogram
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(f'Number of Images per Class {split}')
    plt.show()
