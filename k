import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

# Load the digits dataset 
digits = load_digits() 
images = digits.images
image = images[12] # Different image


# Flatten image for clustering 
flat_image = image.reshape(-1, 1)

# Apply KMeans 
n_clusters = 4
kmeans = KMeans(n_clusters= n_clusters, random_state=0)
kmeans.fit(flat_image)
clustered = kmeans.labels_


# Reshape back to image
segmented_image = clustered.reshape(image.shape)


# Plot
fig, ax = plt.subplots(1, 2, figsize=(6, 4)) 
ax[0].imshow(image, cmap='gray')
 
 
ax[0].set_title("Original Image") 
ax[0].axis('off')

ax[1].imshow(segmented_image, cmap='nipy_spectral') # more colorful segments ax[1].set_title(f"K-Means Segmented (k={n_clusters})")
ax[1].axis('off')


plt.tight_layout()
plt.show() 
