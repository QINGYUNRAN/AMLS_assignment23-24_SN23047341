# import libraries
import matplotlib.pyplot as plt
import numpy as np





# load the data
data_path_A = r"Datasets/PneumoniaMNIST/pneumoniamnist.npz"
data_path_B = r"Datasets/PathMNIST/pathmnist.npz"

data_a = np.load(data_path_A)
data_b = np.load(data_path_B)
print(data_a['train_images'].shape, data_a['train_labels'].shape)
print(data_b['train_images'].shape, data_b['train_labels'].shape)
print(data_a.files, data_b.files)
def visualize_image(image, title):
    if len(image.shape) == 2:
        c = 'gray'
    else:
        c = 'viridis'
    plt.imshow(image, cmap=c)
    plt.axis('off')
    plt.title(title)
    plt.show()

visualize_image(data_a['train_images'][0], data_a['train_labels'][0])
visualize_image(data_b['train_images'][10], data_b['train_labels'][10])


#train A
