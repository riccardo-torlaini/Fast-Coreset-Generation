import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob


for filename in glob.glob("outputs\**", recursive=True):
    if '.npy' in filename:

        data = np.load(filename, allow_pickle=True)
        img_array = data.item()
        assert len(img_array.keys()) == 2
        keys = list(img_array.keys())
        k0, k1 = keys[0], keys[1]
        if 'acc' in keys:
            plt.figure()
            plt.plot(img_array[k0], img_array[k1], '*')
            plt.title('{}: {}, {}'.format(filename[len('outputs\\bico\\'):], k0, k1))
            plt.xlabel(k0)
            plt.ylabel(k1) 
            plt.show()
        print(filename, img_array[k0].shape, img_array[k1].shape)
        # try:
        #     img_array = np.load(filename, allow_pickle=True)
        #     plt.imshow(img_array, cmap="rainbow_r")
        #     img_name = filename+".png"
        #     # matplotlib.image.imsave(img_name, img_array)
        # except TypeError:
        #     print(np.load(filename, allow_pickle=True))
            
        # except Exception as e:
        #     print(e)
        