# Evaluating textures using analysis of distances ([ANODI](https://link.springer.com/article/10.1007/s11004-013-9482-1))

A module to evaluate quality and diversity of textures using the ANODI method. This code supports binary images only, but can be extended to continuous images using clustering as shown in the [paper](https://link.springer.com/article/10.1007/s11004-013-9482-1).

_Requires scikit-learn and scikit-image._

## Analysis of distances

### Multipoint histograms
For each binary image, a histogram of binary patterns so &mdash; called multipoint histograms &mdash; is computed. In essence, we count the occurances for each possible pattern within a patch size. For instance, there are 16 possible patterns for a 2x2 patch:

`[[0,0],[0,0]]`,

`[[0,0],[0,1]]`,

`...`,

`[[1,1],[1,1]]`.

To associate an ID to each pattern, we note that each pattern corresponds to a binary sequence. We then convert it to decimal and use it as its ID, e.g.:

`[[0,0],[0,0]] -> 0000 -> 0`

`[[0,1],[0,0]] -> 0100 -> 4`

`[[1,0],[1,1]] -> 1011 -> 11`

`...`

We then count the occurences of each integer.

### Distance between images
Having defined multipoint histograms, a "distance" between two images is then defined as the Jensen-Shannon
divergence between their multipoint histograms. 
We use it to assess quality and diversity for a set of images:

- The quality of a set of images is given by an inconsistency score, defined as the average distance between the images and a reference image.

- The diversity score for a set of images is defined as the average distance between images within the set.

### Multidimensional scaling

An useful technique to visualize high-dimensional vectors (in this case, images) in a scatterplot. It aims to map  a set of high-dimensional vectors to low dimensions in a way that preserves distances.

See [https://en.wikipedia.org/wiki/Multidimensional_scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling)

## Example
We analyze simple rotated patches extracted from a reference image:

<img src="https://i.imgur.com/TpFVyIh.png" width=200>

<img src="https://i.imgur.com/WL19s4Q.png" width=200>

Let's visualize them using muldimensional scaling. As expected, unrotated and 180 rotated patches are closer to reference.

<img src="https://i.imgur.com/bQvNYF0.png" width=400>

The ANODI scores summarize the results.

|              | inconsistency | diversity |
|--------------|---------------|-----------|
| 0 rotation   | 0.0219        | 0.0362    |
| 90 rotation  | 0.1166        | 0.0620    |
| 180 rotation | 0.0676        | 0.0726    |


## Note: multiresolution analysis
In computing the multipoint histograms, the number of patterns grows very quickly with the patch size as 2^(n^2), so multipoint histograms become impractical for large patch sizes. For example, for a 4x4 patch, there are 2^16=65536 possible patterns.
The paper suggests to instead shrink the image, performing the analysis multple times for multiple resolutions.
