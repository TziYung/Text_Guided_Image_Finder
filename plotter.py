from matplotlib import pyplot as plt
import tensorflow as tf

def plot(array, image_list, text_list, name):
    
    fig, ax  = plt.subplots(nrows = len(array), ncols = len(array[0]) + 1)
    for index, image in enumerate(image_list):
        ax[index, 0].imshow(image, aspect = "auto")
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
        ax[0, index].set_title(text)
    
    fig.subplots_adjust(hspace = 0, wspace = 0)
    fig.savefig(f"{name}.png")
    plt.close(fig)


