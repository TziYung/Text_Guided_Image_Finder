import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from data_loader import Loader
import model
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--table_path", required = True, type = str)
    parser.add_argument("-i", "--image_dir_path", required = True, type = str)
    parser.add_argument("-b", "--batch_size", default = 4, type = int)
    parser.add_argument("-d", "--dim", default = 16, type = int)
    parser.add_argument("-ah", "--attention_head", default = 4, type = int)
    parser.add_argument("-ad", "--attention_dim", default = 4, type = int)
    parser.add_argument("-e", "--epochs", default = 2, type = int)
    parser.add_argument("--text_column", default = -1, type = int)
    parser.add_argument("--img_path_column", default = 0, type = int)

    args = parser.parse_args()

    MODEL_NAME = "distilbert-base-uncased"
    tkzr = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    text_encoder = TFDistilBertModel.from_pretrained(MODEL_NAME)

    image_encoder = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top = False,
        weights='imagenet',
        pooling=None,
        include_preprocessing=True
    )

    loader = Loader(table_path = args.table_path,
        tkzr = tkzr,
        img_dir_path = args.image_dir_path,
        img_size = 224,
        batch_size = args.batch_size,
        img_path_column = args.img_path_column,
        text_column = args.text_column
    )
    clip = model.CLIP(text_encoder, image_encoder, dim = args.dim, attention_head = args.attention_head, attention_dim = args.attention_dim)
    clip.compile(tf.keras.optimizers.Adam(learning_rate = 1e-4))
    clip.fit(loader, epochs = args.epochs)



