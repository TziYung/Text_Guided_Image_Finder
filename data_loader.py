import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
def process_image(img_path_list: list, img_size: tuple, norm = False) -> list:
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
    def __init__(self, table_path,tkzr, img_dir_path, img_size, batch_size = 16, img_path_column = 0, text_column = -1):
        super().__init__()
        self.table_path = table_path
        self.table = self.load_information(self.table_path)
        self.tkzr = tkzr
        self.img_size = img_size
        self.batch_size = batch_size
        self.img_dir_path = img_dir_path
        self.img_path_column = img_path_column
        self.text_column = text_column
        if type(self.table.iloc[0, text_column]) == str:
            # The length of longest sentence
            self.MAX_LEN = self.table.iloc[:, text_column].str.len().max()
        else:
            self.MAX_LEN = self.table.iloc[:, text_column].apply(lambda x : len(max(x, key = len))).max()
            
        self.epoch = 0
    

    def load_information(self, table_path):
        if table_path[-4:] == ".csv":
            info_table = pd.read_csv(table_path, index_col = False)
        elif table_path[-8:] == ".parquet":
            info_table = pd.read_parquet(table_path)
        # Drop any row that contains null value and drop reset the index
        info_table.dropna(axis='index',how='any', inplace = True)
        info_table.reset_index(inplace = True, drop = True)
        
        return info_table

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size

        text = self.table.iloc[start_index:end_index, self.text_column]
        if type(text.iloc[0]) == str:
            text = text.to_list()
        else:
            text = [n[self.epoch % len(n)] for n in text]

        img_path  = self.table.iloc[start_index:end_index, self.img_path_column]
        img_path = [os.path.join(self.img_dir_path, filepath) for filepath in img_path]
        
        return self.process(text, img_path), None, None
    def on_epoch_end(self):
        self.epoch += 1
    def __len__(self):
        length = self.table.shape[0]
        return int(length / self.batch_size + min(length % self.batch_size, 1))
    def process(self, text, img_path):
        text = self.tkzr(text, max_length = self.MAX_LEN, truncation = True, padding = True)
        input_id, attention_mask = [np.array(value) for value in text.values()]
        images = process_image(img_path, (self.img_size, self.img_size))
        images = np.array(images)
        return input_id, attention_mask, images
        
