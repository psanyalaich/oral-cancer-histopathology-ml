import matplotlib.pyplot as plt
from src.preprocessing import load_image
from src.preprocessing import segment_tissue
from src.features import extract_features

# load image
img = load_image(
    "data/100x/normal/Normal_100x_1.jpg"
)

# tissue mask
mask = segment_tissue(img)

# extract feature
features = extract_features(img, mask)

print("Feature Vector:")
print(features)

print("\nFeature Shape:")
print(features.shape)

# visualize
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Tissue Mask")
plt.axis("off")

plt.show()