"""
A module to evaluate quality and diversity of binary images using the ANODI
method https://link.springer.com/article/10.1007/s11004-013-9482-1

Requires scikit-learn and scikit-image.

For each image, a histogram of binary patterns -- so-called multipoint
histograms -- is computed. A 'distance' between two images is then defined as
the Jensen-Shannon divergence between their multipoint histograms. This
distance is used to assess the quality and diversity of a set of images:

- The quality of a set of images is given by an inconsistency score, defined
as the average distance between the images and a reference image.

- The diversity score for a set of images is the average distance within the
set.
"""
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

from sklearn import manifold
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.filters import threshold_otsu

__all__ = ['bin2dec', 'jsd', 'jsdm', 'multipoint_histogram', 'otsu', 'anodi', 'mds']

def bin2dec(x):
    """ Convert an array 'x' representing a binary sequence to decimal, e.g.:
    001 -> 1
    100 -> 4
    011 -> 3
    If 'x' is a 2D array, convert each row.
    https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
    """
    return x.dot(1 << np.arange(x.shape[-1] -1, -1, -1))

def jsd(p, q):
    """ Jensen-Shannon divergence """
    p = p.astype(np.float) / np.sum(p)
    q = q.astype(np.float) / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def jsdm(ps):
    """ Jensen-Shannon dissimilarity matrix for a list of distributions """
    return squareform(pdist(ps, metric=jsd))

def multipoint_histogram(img, patch_size=4):
    """
    Count occurances for each possible binary pattern within a patch size.
    For instance, there are 16 possible patterns for a 2x2 patch:

    `[[0,0],[0,0]]`,

    `[[0,0],[0,1]]`,

    `...`,

    `[[1,1],[1,1]]`.

    To associate an ID to each pattern, we note that each pattern corresponds
    to a binary sequence. We convert it to decimal and use it as its ID, e.g.:

    `[[0,0],[0,0]] -> 0000 -> 0`

    `[[0,1],[0,0]] -> 0100 -> 4`

    `[[1,0],[1,1]] -> 1011 -> 11`

    `...`

    We then count the occurences of each integer.

    Note that the number of patterns grows as 2^(n^2), so multipoint
    histograms become impractical for large patch sizes.
    """
    patches = extract_patches_2d(img, (patch_size, patch_size))
    # each patch is a binary sequence
    patches = patches.reshape(patches.shape[0], -1)
    # convert now to decimal
    patches = bin2dec(patches.reshape(patches.shape[0], -1))
    return np.bincount(patches.astype(np.int), minlength=2**(patch_size**2))

def otsu(img):
    """ Convert image to binary using Otsu's method """
    return img > threshold_otsu(img)

def anodi(img0, imgs, patch_size=4):
    """
    Compute inconsistency and diversity scores for a group of images:

    - The inconsistency is the average distance between the reference image
    'img0' and images in the list 'imgs'.

    - The diversity is the average distance between images within 'imgs'.

    The distance between two images is defined as the Jensen-Shannon
    divergence between their multipoint histograms. See
    https://link.springer.com/article/10.1007/s11004-013-9482-1

    Inputs
    ------
    img0 : 2D array
        Reference image

    imgs : list of 2D arrays
        List of images to be assessed

    Returns
    -------
    inconsistency : float
        Average distance between imgs and img0
    diversity : float
        Average distance between images within imgs
    """
    hist0 = multipoint_histogram(img0, patch_size=patch_size)
    hists = [multipoint_histogram(img, patch_size=patch_size) for img in imgs]
    inconsistency = np.mean([jsd(hist0, h) for h in hists])
    diversity = np.mean(pdist(hists, metric=jsd))
    return inconsistency, diversity

def mds(imgs, patch_size=4):
    """
    Multidimensional scaling for a set of images. The distance between two
    images is defined as the Jensen-Shannon divergence between their
    multipoint histograms.
    https://en.wikipedia.org/wiki/Multidimensional_scaling
    """
    hists = [multipoint_histogram(img, patch_size=patch_size) for img in imgs]
    mat = jsdm(hists)
    pos = manifold.MDS(n_components=2, dissimilarity='precomputed').fit_transform(mat)
    return pos
