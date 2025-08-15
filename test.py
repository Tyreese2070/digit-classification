import matplotlib.pyplot as plt

def display_digit(image, label, flattened=True):
    if flattened:
        image = image.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()