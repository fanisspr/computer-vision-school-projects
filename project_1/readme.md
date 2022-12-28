# Project 1

In this project we use an image containing cells to find different information about the cells, like contours, number of cells, area of cells and borderboxes. 
The input images and the output images are contained in the images folder.
We can use either N3.png or NF3.png. The 2 images are the same, but N3 has noise, which we must remove.

Here is a summary of what the code does:
- Read an image file in grayscale.
- If the filename is 'N3.png', apply median blur to the image.
- Threshold the image to create a binary image.
- Find the contours in the binary image.
- Draw the contours on the original image.
- Remove any contours that are on the border of the image.
- Draw bounding boxes around the remaining contours.
- Calculate the size and average grayscale value of the objects within the bounding boxes.
