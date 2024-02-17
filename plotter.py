from matplotlib import pyplot as plt
import tensorflow as tf
import math

def plot(array, image_list, text_list, name):
    
    fig, ax  = plt.subplots(nrows = len(array), ncols = len(array[0]) + 1, figsize = (30, 10), dpi = 600)
    for index, image in enumerate(image_list):
        ax[index, 0].imshow(image, aspect = "auto")
    array = tf.math.square(array - tf.reduce_min(array))
    max_, min_ = tf.reduce_max(array), tf.reduce_min(array)
    array = (array - min_)/ (max_ - min_)
    for y_index, value_list in enumerate(array):
        for x_index, value in enumerate(value_list):
            value = tf.ones(image_list[0].shape, dtype = "float32") * value
            ax[y_index, x_index + 1].imshow(value, aspect = "auto")

    for y in range(len(array)):
        for x in range(len(array) + 1):
            ax[y, x] .axis("off")
    for index, text in enumerate(["Image"] + text_list):
        if index > 0:
            text = text.split(" ")
            text = [" ".join(text[n * 5 : (n + 1) * 5]) for n in range(math.ceil(len(text)/ 5))]
            text = "\n".join(text)
        ax[0, index].set_title(text, fontsize = 9)
    
    fig.subplots_adjust(hspace = 0, wspace = 0)
    fig.savefig(f"{name}.png")
    plt.close(fig)
