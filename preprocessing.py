from PIL import Image
import numpy as np
from skimage.transform import resize
import pandas as pd
import os
from multiprocessing import Pool

def preprocess_image_parallel(img_path):
    # Same image preprocessing code as before
    image = Image.open(img_path)
    if image.mode != 'L':
        image = image.convert('L')
    image = np.array(image)
    image = image.astype(np.float32)
    image = resize(image, (256, 256), anti_aliasing=True)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_data_parallel(csv_path, png_dir, df):
    # Read the CSV and create a list of image paths
    df['File Name'] = df['File Name'].apply(lambda x: str(x).split('_')[0])
    image_filenames = {f.split('_')[0]: f for f in os.listdir(png_dir) if f.endswith(".png")}
    image_paths = [os.path.join(png_dir, image_filenames.get(row['File Name'], '')) for index, row in df.iterrows()]

    total_images = len(image_paths)
    print(f"Total images to preprocess: {total_images}")

    preprocessed_images = []
    preprocessed_labels = []

    with Pool(processes=4) as pool:  # Adjust the number of processes as needed
        for idx, image in enumerate(pool.imap_unordered(preprocess_image_parallel, image_paths)):
            label_str = df.iloc[idx]['Bi-Rads']
            
            # Factorize the label_str to get a unique integer for each unique label
            label_int = pd.factorize(df['Bi-Rads'])[0][idx]
            
            preprocessed_images.append(image)
            preprocessed_labels.append(label_int)
            print(f"Processed image {idx + 1}/{total_images}, LabelMap: {label_int}, Label: {label_str} ")

    # Saving preprocessed data...
    preprocessed_images = np.array(preprocessed_images, dtype=np.float32)
    preprocessed_labels = np.array(preprocessed_labels, dtype=np.int64)
    np.save('preprocessed_images.npy', preprocessed_images)
    np.save('preprocessed_labels.npy', preprocessed_labels)
    print("Preprocessing complete.")

def main():
    csv_path = 'E:\\INbreast Release 1.0\\INbreast.csv'
    png_dir = 'E:\\INbreast Release 1.0\\AllPNGs\\InBreast Mammo Images'
    
    df = pd.read_csv(csv_path, sep=';')  # Define df here
    
    preprocess_data_parallel(csv_path, png_dir, df)  # Pass df as an argument

if __name__ == "__main__":
    main()
