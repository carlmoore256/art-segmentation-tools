from scipy.signal import convolve2d
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_image_gradients(image):
    if len(image.shape) == 3:
        image = image[:, :, 0]
    # Sobel filter
    kern = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

    grad_x = convolve2d(image, kern, mode='same')
    grad_y = convolve2d(image, kern.T, mode='same')
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradients = np.arctan2(grad_y, grad_x)
    return {
        'grad_x': grad_x,
        'grad_y': grad_y,
        'magnitude': magnitude,
        'gradients': gradients
    }


def plot_image_gradients(gradients):
    plt.figure(figsize=(20, 36))
    plt.subplot(1, 4, 1)
    plt.title("x sobel")
    plt.imshow(gradients['grad_x'])
    plt.subplot(1, 4, 2)
    plt.title("y sobel")
    plt.imshow(gradients['grad_y'])
    plt.subplot(1, 4, 3)
    plt.title("x+y magnitude")
    plt.imshow(gradients['magnitude'])
    plt.subplot(1, 4, 4)
    plt.title("gradients")
    plt.imshow(gradients['gradients'])
    plt.show()


def edge_mask(image, threshold=0.5, kern_size=3, iterations=3, blur=0):
    gradients = get_image_gradients(image)
    edge = gradients['magnitude'] > threshold
    edge[edge > threshold] = 1
    edge[edge <= threshold] = 0
    edge = cv2.dilate((edge * 255).astype(np.uint8),
                      np.ones((kern_size, kern_size), np.uint8), iterations=iterations)
    if blur > 0:
        edge = cv2.GaussianBlur(edge, (blur, blur), 0)
    return edge / 255


def normalize_image_percentile(image, lower_percentile=1, upper_percentile=99):
    # Convert the image to a numpy array if it's not already
    image_array = np.array(image)
    # Calculate the lower and upper bounds based on percentiles
    lower_bound = np.percentile(image_array[image_array > 0], lower_percentile)
    upper_bound = np.percentile(image_array[image_array > 0], upper_percentile)
    # Clip the image to these bounds
    clipped_image = np.clip(image_array, lower_bound, upper_bound)
    # Normalize to the range [0, 1]
    normalized_image = (clipped_image - lower_bound) / \
        (upper_bound - lower_bound)
    # Replace any NaN values with 0 (if division by zero occurred)
    normalized_image = np.nan_to_num(normalized_image)
    return normalized_image


def crop_depth_map(depth_map, bbox, mask, edge_removal=False, lower_percentile=1, upper_percentile=99):
    depth = depth_map.copy()[int(bbox[2]):int(bbox[2])+mask.shape[0], int(bbox[0]):int(bbox[0])+mask.shape[1]]
    # make sure depth map and mask are the same size
    if depth.shape[0] != mask.shape[0] or depth.shape[1] != mask.shape[1]:
        raise Exception(
            f"[!] Depth map and mask are not the same size: {depth.shape} vs {mask.shape}") 
    # apply mask
    depth = mask.apply(depth)

    # get the average depth of the mask thats nonzero
    avg_depth = np.mean(depth[depth > 0])

    if edge_removal:
        # remove edge artifacts
        try:
            edge = edge_mask(depth, threshold=0.5, kern_size=3,
                            iterations=3, blur=21)
            edge = 1-edge
            depth = depth * np.expand_dims(edge, axis=-1)
        except:
            print(f'[!] Failed to remove edge artifacts: {depth.shape} | {mask.shape}')
    try:
        depth = normalize_image_percentile(
            depth, lower_percentile, upper_percentile)
    except:
        print(f'[!] Failed to normalize depth map: {depth.shape}')
    return depth, avg_depth


def generate_normal_map(image, ksize=3, scale=1, delta=0, norm_range=(0, 1)):
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter to compute gradients
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize,
                       scale=scale, delta=delta)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize,
                       scale=scale, delta=delta)

    # Normalize the gradients
    norm_grad_x = cv2.normalize(
        grad_x, None, alpha=norm_range[0], beta=norm_range[1], norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_grad_y = cv2.normalize(
        grad_y, None, alpha=norm_range[0], beta=norm_range[1], norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Combine to form a normal map
    normal_map = np.dstack(
        ((norm_grad_x + 1) / 2, (norm_grad_y + 1) / 2, np.ones_like(norm_grad_x)))

    # Convert back to uint8
    normal_map = (255 * normal_map).astype(np.uint8)

    return normal_map


def plot_histogram(image, exclude_zeros=True):
    if exclude_zeros:
        image = image[image > 0]
    histogram, bin_edges = np.histogram(
        image, bins=255, range=(np.min(image), np.max(image)))
    plt.figure(figsize=(10, 4))
    plt.plot(bin_edges[0:-1], histogram)  # Exclude the last bin edge
    plt.title("Histogram of Image")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
