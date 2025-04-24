import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage.measure import regionprops, regionprops_table

from typing import Tuple, List

def distance(point1, point2):
    point2point_distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return point2point_distance

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
    - Tuple[Optional[Tuple[int, int]], Tuple[float, float], List[Tuple[int, int]]]: A tuple containing:
        - The coordinates of the global maximum as a tuple of integers (row, column), or None if no local maxima are found.
        - The coordinates of the centroid of the mask as a tuple of floats (row, column).
        - A list of local maxima coordinates as tuples of integers (row, column).

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
    properties = regionprops(mask)
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
    
    return global_max_coord, centroid, local_maxima_coords

def find_centrosome(
        image_tcell: np.ndarray, 
        mask_tcell: np.ndarray, 
        cancer_cell_centroid_pos: List[float]) -> Tuple[float, float]:
    
    """
    Finds the centrosome position in a T-cell image based on the closest local maximum to a given cancer cell centroid.

    Parameters:
    - image_tcell (np.ndarray): The input T-cell image as a 2D NumPy array.
    - mask_tcell (np.ndarray): The binary mask for the T-cell image as a 2D NumPy array.
    - cancer_cell_centroid_pos (List[float]): The coordinates of the cancer cell centroid as a list of floats [x, y].

    Returns:
    - Tuple[float, float]: The coordinates of the centrosome as a tuple of floats (x, y), which is the closest local maximum to the cancer cell centroid.

    Description:
    This function performs the following steps:
    1. Applies the T-cell mask to the T-cell image.
    2. Uses the `find_polarisation` function to find local maxima in the masked T-cell image.
    3. Calculates the distance from the cancer cell centroid to each local maximum.
    4. Identifies and returns the coordinates of the local maximum that is closest to the cancer cell centroid.

    The function is useful for identifying the centrosome position in T-cell images based on proximity to a given cancer cell centroid.
    """

    _, _, maxima = find_polarisation(image_tcell*mask_tcell, mask_tcell)

    dist0 = 999
    index0 = 0
    for index, point in enumerate(maxima):
        dist = distance(cancer_cell_centroid_pos, point)
        if dist < dist0:
            dist0 = dist
            index0 = index

    return maxima[index0]

def compute_alignement(
        image_cancer: np.ndarray, 
        image_tcell: np.ndarray, 
        mask_cancer: np.ndarray, 
        mask_tcell: np.ndarray, 
        display:bool = False) -> float:
    
    """
    Computes the alignment angle between vectors formed by the centroids of cancer and T-cell regions and the centrosome.

    Parameters:
    - image_cancer (np.ndarray): The input cancer image as a 2D NumPy array.
    - image_tcell (np.ndarray): The input T-cell image as a 2D NumPy array.
    - mask_cancer (np.ndarray): The binary mask for the cancer image as a 2D NumPy array.
    - mask_tcell (np.ndarray): The binary mask for the T-cell image as a 2D NumPy array.
    - display (bool): Flag to display the T-cell image with annotations. Default is False.

    Returns:
    - float: The alignment angle in degrees between the vectors formed by the centroids and the centrosome.

    Description:
    This function performs the following steps:
    1. Computes the centroids, bounding boxes, and equivalent diameters of the regions in the cancer and T-cell masks.
    2. Identifies the centrosome position in the T-cell image using the `find_centrosome` function.
    3. Creates vectors from the centroids of the T-cell and cancer regions to the centrosome.
    4. Calculates the dot product and magnitudes of these vectors.
    5. Computes the angle between the vectors in radians and converts it to degrees.
    6. Optionally displays the T-cell image with annotations showing the centroids and centrosome.

    The function is useful for analyzing the spatial relationship between cancer cells, T-cells, and the centrosome in images.
    """

    properties_tcell = regionprops_table(mask_tcell, properties = ('Centroid', 'BoundingBox', 'EquivDiameter'))
    properties_cancer = regionprops_table(mask_cancer, properties = ('Centroid', 'BoundingBox', 'EquivDiameter'))
    
    point_a = [properties_tcell['Centroid-0'][0], properties_tcell['Centroid-1'][0]]
    point_b = [properties_cancer['Centroid-0'][0], properties_cancer['Centroid-1'][0]]
    point_c = find_centrosome(image_tcell*mask_tcell, mask_tcell, point_b)

    # Create vectors from the points
    vector_ab = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
    vector_ac = np.array([point_c[0] - point_a[0], point_c[1] - point_a[1]])

    # Calculate the dot product of the vectors
    dot_product = np.dot(vector_ab, vector_ac)

    # Calculate the magnitudes of the vectors
    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_ac = np.linalg.norm(vector_ac)

    # Compute the angle in radians then degrees
    angle_radians = np.arccos(dot_product / (magnitude_ab * magnitude_ac))
    angle_degrees = np.degrees(angle_radians)

    if display:
        print(angle_degrees)
        plt.imshow(image_tcell, cmap='gray')
        plt.plot(point_a[1], point_a[0], 'ro')
        plt.plot(point_b[1], point_b[0], 'ro')
        plt.plot(point_c[1], point_c[0], 'bo')

    return angle_degrees

if __name__ == "__main__":
    pass