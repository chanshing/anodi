""" An example of ANODI evaluation with MDS visualization. In this toy
problem, we evaluate rotated patches extracted from a reference exemplar
image (default='channel.png') """

import argparse
import skimage.io as io
from skimage.transform import rotate
import matplotlib.pyplot as plt

import binary_anodi as anodi

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='channel.png')
args = parser.parse_args()

# load reference
img0 = io.imread(args.image)
img0 = anodi.otsu(img0)  # convert to binary

NSAMPLE = 10  # note: can be expensive with large samples

imgs1 = anodi.extract_patches_2d(img0, (64, 64), max_patches=NSAMPLE)
print "Extracted {} patches of size {}x{}".format(*imgs1.shape)

imgs2 = anodi.extract_patches_2d(rotate(img0, 90), (64, 64), max_patches=NSAMPLE)
print "Extracted {} patches of size {}x{}".format(*imgs2.shape)

imgs3 = anodi.extract_patches_2d(rotate(img0, 180), (64, 64), max_patches=NSAMPLE)
print "Extracted {} patches of size {}x{}".format(*imgs3.shape)

# --- ANODI evaluations
print "[  0 rotation] inconsistency: {:.4f} | diversity: {:.4f}".format(*anodi.anodi(img0, imgs1))
print "[ 90 rotation] inconsistency: {:.4f} | diversity: {:.4f}".format(*anodi.anodi(img0, imgs2))
print "[180 rotation] inconsistency: {:.4f} | diversity: {:.4f}".format(*anodi.anodi(img0, imgs3))

# --- MDS on compound set of images (incl. reference)
imgs_all = [img for imgs in [imgs1, imgs2, imgs3] for img in imgs] + [img0]
pos = anodi.mds(imgs_all)

# MDS visualization
plt.figure()
plt.scatter(pos[:NSAMPLE,0], pos[:NSAMPLE,1], c='C0', marker='o', label='0 rotation')
plt.scatter(pos[NSAMPLE:NSAMPLE*2,0], pos[NSAMPLE:NSAMPLE*2,1], c='C1', marker='d', label='90 rotation')
plt.scatter(pos[NSAMPLE*2:NSAMPLE*3,0], pos[NSAMPLE*2:NSAMPLE*3,1], c='C2', marker='s', label='180 rotation')
plt.scatter(pos[-1,0], pos[-1,1], c='k', marker='^', s=50, label='reference')
plt.title('MDS visualization')
plt.legend()
plt.savefig('mds.png')

# show patches
fig, axs = plt.subplots(3,4)
for i,imgs in enumerate([imgs1, imgs2, imgs3]):
    for ax, img in zip(axs[i], imgs):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
axs[0,0].set_ylabel('0 rot')
axs[1,0].set_ylabel('90 rot')
axs[2,0].set_ylabel('180 rot')
fig.suptitle('patches')
plt.savefig('patches.png')
