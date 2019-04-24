#! /usr/bin/env
from bagnets.utils import plot_heatmap, generate_heatmap_pytorch
import numpy as np
import bagnets.pytorch
from foolbox.utils import imagenet_example
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

pytorch_model = bagnets.pytorch.bagnet33(pretrained=True).cuda()

original, label = imagenet_example(data_format='channels_first')
original=np.expand_dims(original,0)
label = np.expand_dims(label,0)

# preprocess sample image
sample = original / 255.
sample -= np.array([0.485, 0.456, 0.406])[:, None, None]
sample /= np.array([0.229, 0.224, 0.225])[:, None, None]

# generate heatmap

heatmap = generate_heatmap_pytorch(pytorch_model, sample, label, 33)

# plot heatmap\n",
fig = figure(figsize=(8, 4))
original_image = original[0].transpose([1,2,0])
ax = plt.subplot(121)
ax.set_title('original')
ax.imshow(original_image / 255.)
plt.axis('off')
ax = plt.subplot(122)
ax.set_title('heatmap')
plot_heatmap(heatmap, original_image, ax, dilation=0.5, percentile=99, alpha=.25)
plt.axis('off')
plt.show()
