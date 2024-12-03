import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import ConcatDataset
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms


__all__ = ['CustomDataset', 'create_combined_dataset', 'load_json_data',
           'df_transformer', 'load_img_data', 'init_dataset']


class CustomDataset(Dataset):
    """
    A custom dataset class for handling image and tabular data.

    Attributes:
        image_files (list): A list of paths to image files.
        dataframe (DataFrame): A pandas DataFrame containing related tabular
        data.
        transform (Compose): A torchvision transforms composition to apply to
        the images.

    Methods:
        __len__: Returns the length of the dataset.
        __getitem__: Retrieves an image and its associated data by index.
    """
    def __init__(self, image_files, dataframe):
        self.image_files = image_files
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((366, 366)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Load the image as a PIL image
        image = Image.open(self.image_files[idx]).convert('L')
        image = self.transform(image)

        # Get the other data from the dataframe
        storm_id = torch.tensor([self.dataframe.iloc[idx]['id']],
                                dtype=torch.float32)
        relative_time = torch.tensor([self.dataframe.iloc[idx]
                                      ['relative_time']], dtype=torch.float32)
        ocean = torch.tensor([self.dataframe.iloc[idx]['ocean']],
                             dtype=torch.float32)
        wind_speed = torch.tensor([self.dataframe.iloc[idx]['wind_speed']],
                                  dtype=torch.float32)
        return image, storm_id, relative_time, ocean, wind_speed


def create_combined_dataset(parent_folder_path, initial_count, transform=None):
    """
    Creates a combined dataset from multiple subfolders within a parent folder.

    Args:
        parent_folder_path (str): The path to the parent folder.
        initial_count (int): An initial count value for indexing.
        transform (optional, transforms.Compose): A torchvision transform to
        apply to images.

    Returns:
        ConcatDataset: A concatenated dataset comprising datasets from each
        subfolder.

    Example:
        >>> combined_dataset = create_combined_dataset('path/to/parent', 0)
        >>> len(combined_dataset)
        50  # Assuming 50 datasets were combined
    """
    all_datasets = []
    count = initial_count

    for subfolder_name in os.listdir(parent_folder_path):
        subfolder_path = os.path.join(parent_folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            current_dataset = CustomDataset(subfolder_path, count,
                                            transform=transform)
            all_datasets.append(current_dataset)
            count += 1

    combined_dataset = ConcatDataset(all_datasets)
    return combined_dataset


def load_json_data(folder_path='./data/'):
    """
    Loads JSON data from a specified folder.

    Args:
        folder_path (str, optional): The path to the folder containing JSON
        files.

    Returns:
        tuple: A tuple containing a DataFrame of all data and a dictionary of
        file names.

    Example:
        >>> df, file_dict = load_json_data('./data/')
        >>> df.columns
        ['id', 'relative_time', 'ocean', 'wind_speed']  # Example output
    """
    feature_files = []
    label_files = []
    file_name_dict = {}

    for name in os.listdir(folder_path):
        if name == ".DS_Store":
            continue
        count = []
        directory_path = folder_path + name
        for file in os.listdir(directory_path):
            if len(file) < 7:
                continue
            img = directory_path + "/" + name + "_" + file[4:7] \
                + ".jpg"
            fea = directory_path + "/" + name + "_" + file[4:7] \
                + "_features.json"
            lab = directory_path + "/" + name + "_" + file[4:7] \
                + "_label.json"
            if not (os.path.exists(img) and os.path.exists(fea) and os.path.
                    exists(lab)):
                continue
            if file.endswith("_features.json"):
                feature_files.append(os.path.join(directory_path, file))
                if int(file[4:7]) not in count:
                    count.append(int(file[4:7]))
            elif file.endswith("_label.json"):
                label_files.append(os.path.join(directory_path, file))
                if int(file[4:7]) not in count:
                    count.append(int(file[4:7]))
        count.sort()
        file_name_dict[name] = count
    feature_files.sort()
    label_files.sort()
    all_data = []
    for feature_file, label_file in zip(feature_files, label_files):
        with open(feature_file, 'r') as file:
            feature_data = json.load(file)
        with open(label_file, 'r') as file:
            label_data = json.load(file)
        merged_data = {**feature_data, **label_data}
        all_data.append(merged_data)

    return (pd.DataFrame(all_data), file_name_dict)


def df_transformer(df):
    """
    Transforms a DataFrame by encoding categorical variables and converting to
    float.

    Args:
        df (DataFrame): The DataFrame to transform.

    Returns:
        tuple: A tuple containing the transformed DataFrame and the
        LabelEncoder used.

    Example:
        >>> transformed_df, encoder = df_transformer(df)
        >>> transformed_df.dtypes
        id              float64
        relative_time   float64
        ocean           float64
        wind_speed      float64
    """
    label_encoder = LabelEncoder()
    df['id'] = label_encoder.fit_transform(df['storm_id'])
    df = df.drop(["storm_id"], axis=1).copy()
    df = df.astype(float)
    return (df, label_encoder)


def load_img_data(file_name_dict, folder_path='./data/'):
    """
    Loads image data based on a dictionary of file names.

    Args:
        file_name_dict (dict): A dictionary with keys as file names and values
        as indices.
        folder_path (str, optional): The path to the folder containing images.

    Returns:
        list: A list of paths to image files.

    Example:
        >>> img_files = load_img_data(file_dict)
        >>> len(img_files)
        100  # Assuming 100 image files were loaded
    """
    all_image_files = []
    sorted_keys = sorted(file_name_dict.keys())
    for name in sorted_keys:
        print(name)
        if name == ".DS_Store":
            continue
        image_pattern = folder_path + name + "/" + name + "_{:03d}.jpg"
        image_indices = file_name_dict[name]
        all_image_files = all_image_files + [image_pattern.format(i)
                                             for i in image_indices]

    return all_image_files


def init_dataset(img_files, json_df):
    """
    Initializes a CustomDataset with image files and a DataFrame.

    Args:
        img_files (list): A list of paths to image files.
        json_df (DataFrame): A DataFrame with associated data for the images.

    Returns:
        CustomDataset: The initialized CustomDataset.

    Example:
        >>> dataset = init_dataset(img_files, json_df)
        >>> len(dataset)
        100  # Assuming the dataset contains 100 items
    """
    return CustomDataset(img_files, json_df)
