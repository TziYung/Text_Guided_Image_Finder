import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
def process_image(img_path_list: list, img_size: tuple, norm = True) -> list:
    # img_size is the width and length of the image in tuple
    processed_images = []
    
    for img_path in img_path_list:

        try:
            # read, resize image, and convert image from bgr to rgb due to the reason
            # that opencv read image in the pattern of bgr
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if norm:
                img = (img - 127.5) / 127.5
            processed_images.append(img)

        except Exception as e:
            print(e)
            print(f"Can't load image from: {img_path}")

    return processed_images
class Loader(tf.keras.utils.Sequence):
    def __init__(self, text_path,tkzr, img_dir_path, img_size, batch_size = 16, img_path_column = 0, text_column = -1):
        super().__init__()
        self.text_path = text_path
        self.text_info = self.load_information(self.text_path)
        self.tkzr = tkzr
        self.img_size = img_size
        self.batch_size = batch_size
        self.img_dir_path = img_dir_path
        self.img_path_column = img_path_column
        self.text_column = text_column
        self.MAX_LEN = self.text_info.iloc[:, text_column].str.len().max()

    def load_information(self, text_path):
        info_table = pd.read_csv(self.text_path, index_col = False)
        # Drop any row that contains null value and drop reset the index
        info_table.dropna(axis='index',how='any', inplace = True)
        info_table.reset_index(inplace = True, drop = True)
        
        return info_table

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size

        text = self.text_info.iloc[start_index:end_index, self.text_column]
        text = self.tkzr(text.to_list(), max_length = self.MAX_LEN, truncation = True, padding = True)
        text = [np.array(value) for value in text.values()]

        img_path  = self.text_info.iloc[start_index:end_index, self.img_path_column]
        img_path = [os.path.join(self.img_dir_path, filepath) for filepath in img_path]
        images = process_image(img_path, (self.img_size, self.img_size))
        images = np.array(images)
        
        return list(zip(text[0], text[1], images))
    def __len__(self):
        length = self.text_info.shape[0]
        return int(length / self.batch_size) + min(length % self.batch_size, 1)
