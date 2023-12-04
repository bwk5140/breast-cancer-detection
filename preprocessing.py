# preprocessing.py
from PIL import Image
import numpy as np
from skimage.transform import resize
import pandas as pd
import os

def preprocess_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'L':
        image = image.convert('L')
    image = np.array(image)
    image = image.astype(np.float32)  # Convert image to float32
    image = resize(image, (256, 256), anti_aliasing=True)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    return image

def main():
    csv_path = 'E:\\INbreast Release 1.0\\INbreast.csv'
    png_dir = 'E:\\INbreast Release 1.0\\AllPNGs\\InBreast Mammo Images'

    print("Reading CSV file...")
    df = pd.read_csv(csv_path, sep=';')
    df['File Name'] = df['File Name'].apply(lambda x: str(x).split('_')[0])
    unique_labels = df['Bi-Rads'].unique()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    df['Bi-Rads'] = df['Bi-Rads'].map(label_to_int)

    image_filenames = {f.split('_')[0]: f for f in os.listdir(png_dir) if f.endswith(".png")}
    filtered_data = []
    for index, row in df.iterrows():
        file_id = row['File Name']
        if file_id in image_filenames:
            filtered_data.append({
                'File Name': image_filenames[file_id], 
                'Bi-Rads Label': label_to_int.get(row['Bi-Rads'], None)
            })

    filtered_df = pd.DataFrame(filtered_data)

    preprocessed_images = []
    preprocessed_labels = []
    for index, row in filtered_df.iterrows():
        img_path = os.path.join(png_dir, row['File Name'])
        print(f"Preprocessing image {index + 1}/{len(filtered_df)}: {img_path}")
        image = preprocess_image(img_path)
        label = row['Bi-Rads Label']
        preprocessed_images.append(image)
        preprocessed_labels.append(label)

    print("Saving preprocessed data...")
    preprocessed_images = np.array(preprocessed_images, dtype=np.float32)
    preprocessed_labels = np.array(preprocessed_labels, dtype=np.int64)
    np.save('preprocessed_images.npy', preprocessed_images)
    np.save('preprocessed_labels.npy', preprocessed_labels)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
