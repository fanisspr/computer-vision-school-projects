'''
In this project we use an image containing cells to find different information about the cells. 
We can use either N3.png or NF3.png. The 2 images are the same, but N3 has noise, which we must remove.

Here is a summary of what the code does:
-Read an image file in grayscale.
-If the filename is 'N3.png', apply median blur to the image.
-Threshold the image to create a binary image.
-Find the contours in the binary image.
-Draw the contours on the original image.
-Remove any contours that are on the border of the image.
-Draw bounding boxes around the remaining contours.
-Calculate the size and average grayscale value of the objects within the bounding boxes.
'''

import cv2
import numpy as np


filename = 'images/N3.png'


def median_blur(img, kernel=(5, 5)):
    """
    Applies median blur to an image.

    Parameters:
        img (numpy.ndarray): The input image.
        kernel (tuple): The size of the kernel to use for the blur.

    Returns:
        numpy.ndarray: The blurred image.
    """
    # Create an output image with the same shape as the input image
    output = np.zeros(img.shape, "uint8")

    # Calculate padding for the input image
    pad_height = int((kernel[0] - 1) / 2)
    pad_width = int((kernel[1] - 1) / 2)

    # Pad the input image
    padded_image = np.zeros(
        (img.shape[0] + (2 * pad_height), img.shape[1] + (2 * pad_width)), "uint8")
    padded_image[pad_height:padded_image.shape[0] - pad_height,
                 pad_width:padded_image.shape[1] - pad_width] = img

    # Iterate through the image pixels
    for px_x in range(0, img.shape[0]):
        for px_y in range(0, img.shape[1]):
            # Sort the values in the kernel
            sorted_mat = np.sort(
                padded_image[px_x:px_x + kernel[0], px_y:px_y + kernel[1]],
                axis=None)
            # Take the median value and set it as the output pixel value
            output[px_x, px_y] = sorted_mat[len(sorted_mat)//2]
    return output


def sum_of_rect(img, row, col, w, h):
    """
    Calculates the sum of all pixels within a rectangle in an image.

    Parameters:
    img (numpy.ndarray): The input image.
    row (int): The row index of the top-left corner of the rectangle.
    col (int): The column index of the top-left corner of the rectangle.
    w (int): The width of the rectangle.
    h (int): The height of the rectangle.

    Returns:
        int: The sum of all pixel values within the rectangle.
    """
    # Create a new image with the same dimensions as the input image, but with an additional
    # row and column for the cumulative sum
    D = np.zeros((row+h, col+w))
    D[0, 0] = img[0, 0]
    for c in range(1, col + w):
        D[0, c] = img[0, c] + D[0, c-1]
    for r in range(1, row + h):
        D[r, 0] = img[r, 0] + D[r-1, 0]
    for r in range(1, row + h):
        for c in range(1, col + w):
            D[r, c] = img[r, c] + D[r-1, c] + D[r, c-1] - D[r-1, c-1]

    # Calculate the sum of the rectangle based on the cumulative sum image
    if col == 0 and row == 0:
        rect_sum = D[h-1, w-1]
    elif row == 0:
        C = D[h-1, col-1]
        rect_sum = D[h-1, col+w-1] - C
    elif col == 0:
        B = D[row-1, w-1]
        rect_sum = D[row+h-1, w-1] - B
    else:
        A = D[row-1, col-1]
        B = D[row-1, col + w-1]
        C = D[row + h-1, col-1]
        rect_sum = D[row+h-1, col+w-1] - B - C + A
    return rect_sum


def plot_image(image, title):
    """Plots an image using OpenCV's GUI window.

    This function displays the image in a GUI window and waits for a key press before closing the window.

    Args:
    image (numpy.ndarray): The image to display.
    title (str): The title of the window.
    """
    cv2.namedWindow(title, )
    cv2.imshow(title, image)
    cv2.waitKey(0)


# Read the image as grayscale
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# plot_image(img, 'main')

# If the filename is 'N3.png', apply median blur to the image
if(filename == 'images/N3.png'):
    img = median_blur(img, (5, 5))
    cv2.imwrite(f'{filename}-medianblur.png', img)

# Threshold the image to create a binary image
img_bi = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite(f'{filename}-binary.png', img_bi)

# Find the contours in the binary image.
contours, _ = cv2.findContours(
    img_bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Print the number of contours found
print("Found %d objects." % len(contours))
# Print the size of each contour
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))

# Draw the contours on the original image.
cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
cv2.imwrite(f'{filename}-contours.png', img)

k = 0
temp = list(contours)
for c in contours:
    k += 1
    (x, y, w, h) = cv2.boundingRect(c)
    # Remove any contours that are on the border of the image.
    if(x == 0 or (x+w) == img_bi.shape[1] or y == 0 or (y+h) == img_bi.shape[0]):
        del temp[k]
        k -= 1
        continue

    # Draw bounding boxes around the remaining contours.
    bounds = cv2.rectangle(img, (x, y),
                           (x + w, y + h), (255, 0, 0), 1)
    cv2.imwrite(f'{filename}-boundaries.png', bounds)

contours = temp
# Print the number of bounded objects found
print("Found %d objects." % len(contours))
# Calculate the size and average grayscale value of the objects within the bounding boxes.
for (i, c) in enumerate(contours):
    print("\tSize of object %d: %d" % (i, cv2.contourArea(c)))
    (x, y, w, h) = cv2.boundingRect(c)  # WARNING x=col, y=row
    sum = sum_of_rect(img, y, x, w, h)
    gray = sum / (h*w)
    print("\tAverage value of gray in bounding box of contour %d: %d " % (i, gray))
