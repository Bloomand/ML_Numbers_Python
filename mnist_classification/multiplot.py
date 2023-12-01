import numpy as np
import os
import matplotlib.pyplot as plt
def multiplot(b):
    for i in range(0, b):
        path = os.listdir('test')
        rimage = np.random.choice(path)
        path2 = plt.imread('test/' + rimage)
        rimage = np.random.choice(path)
        path3 = plt.imread('test/' + rimage)
        rimage = np.random.choice(path)

        path4 = plt.imread('test/' + rimage)

        fig, axes = plt.subplots(1, 3)
        path2[:, 0] = 0
        axes[0].imshow(path2)
        path3[:, 1] = 0
        axes[1].imshow(path3)

        path4[:, 2]
        axes[2].imshow(path4)

        for ax in axes:
            ax.set_yticks([])

        fig.set_figwidth(12)
        fig.set_figheight(6)

        plt.show()

if __name__=='__main__':
    multiplot(int(input()))