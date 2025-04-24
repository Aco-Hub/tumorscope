import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, measure
from typing import Tuple

def find_polarisation(
        image: np.ndarray, 
        mask: np.ndarray,
        display:bool = False) -> Tuple[[Tuple[int, int]], Tuple[float, float]]:
    
    """
    Finds the global maximum and centroid of a mask in an image and optionally displays the results.

    Parameters:
    - image (np.ndarray): The input image as a 2D NumPy array.
    - mask (np.ndarray): The binary mask as a 2D NumPy array.
    - display (bool): Flag to display the image, mask, and annotations. Default is False.

    Returns:
    - Tuple[Optional[Tuple[int, int]], Tuple[float, float]]: A tuple containing:
        - The coordinates of the global maximum as a tuple of integers (row, column), or None if no local maxima are found.
        - The coordinates of the centroid of the mask as a tuple of floats (row, column).

    Description:
    This function performs the following steps:
    1. Identifies local maxima in the input image.
    2. Determines the global maximum among these local maxima.
    3. Calculates the centroid of the provided binary mask.
    4. Optionally displays the image, mask, global maximum, centroid, and an arrow between them.

    The function is useful for analyzing features in images, such as identifying the brightest spot and the center of a region of interest.
    """

    # Find local maxima in the image
    local_maxima_coords = feature.peak_local_max(image, min_distance=10)
    # Find the global maximum among the local maxima
    if len(local_maxima_coords) > 0:
        maxima_values = [image[coord[0], coord[1]] for coord in local_maxima_coords]
        global_max_index = np.argmax(maxima_values)
        global_max_coord = local_maxima_coords[global_max_index]
    else:
        global_max_coord = None

    # Find the centroid of the mask using region properties
    properties = measure.regionprops(mask)
    centroid = properties[0].centroid

    if display:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Image')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Mask')
        
        ax[2].imshow(image, cmap='gray')
        ax[2].plot(centroid[1], centroid[0], 'bo')
        ax[2].plot(global_max_coord[1], global_max_coord[0], 'ro')
        # Define the start and end points of the arrow
        start_point = (centroid[1], centroid[0])
        end_point = (global_max_coord[1], global_max_coord[0])
        plt.annotate(
            '', xy=end_point, xytext=start_point,
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )
        ax[2].set_title('Annotated Image')
    
    return global_max_coord, centroid

if __name__ == "__main__":
    pass