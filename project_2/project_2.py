import cv2
import numpy as np


def match(feature_vectors1, feature_vectors2):
    """
    Finds the nearest neighbor for each feature vector in feature_vectors1 in feature_vectors2.
    Only returns a match if the distance to the second nearest neighbor is more than 50% of the distance to the nearest neighbor.

    Parameters:
    feature_vectors1 (numpy array): NxM array of N feature vectors with M dimensions.
    feature_vectors2 (numpy array): NxM array of N feature vectors with M dimensions.

    Returns:
    matches (list): List of cv2.DMatch objects, containing the indices of the matching feature vectors and the distance between them.
    """
    num_vectors1 = feature_vectors1.shape[0]
    num_vectors2 = feature_vectors2.shape[0]

    matches = []
    match_count = 0
    for i in range(num_vectors1):
        fv = feature_vectors1[i, :]
        # Calculate the difference between the current feature vector and all feature vectors in feature_vectors2
        diff = feature_vectors2 - fv
        diff = np.abs(diff)
        # Calculate the distances between the current feature vector and all feature vectors in feature_vectors2
        distances = np.sum(diff, axis=1)

        # Find the index of the feature vector in feature_vectors2 with the smallest distance to the current feature vector
        nearest_neighbor_index = np.argmin(distances)
        nearest_neighbor_distance = distances[nearest_neighbor_index]

        # Set the distance of the nearest neighbor to infinity so it is not considered in the next nearest neighbor search
        distances[nearest_neighbor_index] = np.inf

        # Find the index of the feature vector in feature_vectors2 with the second smallest distance to the current feature vector
        second_nearest_neighbor_index = np.argmin(distances)
        second_nearest_neighbor_distance = distances[second_nearest_neighbor_index]

        # Only append a match if the distance to the second nearest neighbor is more than 50% of the distance to the nearest neighbor
        if nearest_neighbor_distance / second_nearest_neighbor_distance < 0.5:
            matches.append(cv2.DMatch(i, nearest_neighbor_index,
                           nearest_neighbor_distance))
            match_count += 1
    print('matches: ', match_count)
    return matches


def resize(image, scale_percent):
    """
    Resizes the input image by a given percentage.

    Parameters:
    image (numpy array): The image to be resized, represented as a Numpy array.
    scale_percent (float): The percentage by which to scale the image.

    Returns:
    resized (numpy array): The resized image, represented as a Numpy array.
    """
    # Calculate the new dimensions of the image
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)

    # Resize the image using the INTER_AREA interpolation method
    resized = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return resized


def crop(image):
    """
    Crops the input image to the bounding box of the largest connected component in the image.

    Parameters:
    image (numpy array): The image to be cropped, represented as a Numpy array.

    Returns:
    cropped (numpy array): The cropped image, represented as a Numpy array.
    """
    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest connected component
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop the image to the bounding box
    cropped = image[y:y+h-1, x:x+w-1]
    return cropped


def stitching_right_to_left(image1, image2, features):
    """
    Stitches two images together using the right image as the reference frame.

    Parameters:
    image1 (numpy array): The first image to be stitched, represented as a Numpy array.
    image2 (numpy array): The second image to be stitched, represented as a Numpy array.

    Returns:
    stitched (numpy array): The stitched image, represented as a Numpy array.
    """
    # Detect keypoints and compute descriptors for both images
    kp1 = features.detect(image1)
    desc1 = features.compute(image1, kp1)
    kp2 = features.detect(image2)
    desc2 = features.compute(image2, kp2)

    # Find matches between the two images in both directions
    matches1 = match(desc1[1], desc2[1])
    matches2 = match(desc2[1], desc1[1])

    # Filter matches using cross-checking
    cross_check = []
    match_count = 0
    img_pt1 = []
    img_pt2 = []
    for m1 in matches1:
        for m2 in matches2:
            # Check if the matches are mutual.
            # If the matches have the same indices in both lists, it is a cross-check match
            if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                match_count += 1
                img_pt1.append(kp1[m1.queryIdx].pt)
                img_pt2.append(kp2[m2.queryIdx].pt)
                cross_check.append(m1)
    print(f'Number of cross-checked matches left to right: {len(cross_check)}')
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    # Draw the mutual matches between the two images
    matched_image = cv2.drawMatches(
        image1, desc1[0], image2, desc2[0], cross_check, None)
    cv2.namedWindow('matchesR-L')
    cv2.imshow('matchesR-L', matched_image)
    cv2.waitKey(0)

    # Find the homography matrix using RANSAC
    M, mask = cv2.findHomography(img_pt1, img_pt2, cv2.RANSAC)

    # Warp image1 using the homography matrix
    stitched = cv2.warpPerspective(
        image1, M, (image2.shape[1]+500, image2.shape[0]+200))
    # Overlay image2 on the warped image1
    stitched[0: image2.shape[0], 0: image2.shape[1]] = image2
    # Crop the stitched image to remove any black borders
    stitched = crop(stitched)
    cv2.namedWindow('RightToLeft')  # , cv2.WINDOW_NORMAL
    cv2.imshow('RightToLeft', stitched)
    cv2.waitKey(0)

    return stitched


def stitching_left_to_right(img1, img2, features):
    """Stitch the images together using a left-to-right stitching method.

    This function detects and matches keypoints in the two images, computes a homography
    matrix to transform the second image to align with the first image, and then warps and
    combines the two images.

    Args:
        img1 (ndarray): The first image.
        img2 (ndarray): The second image.

    Returns:
        ndarray: The stitched image.
    """
    # Detect and compute keypoints and descriptors for both images
    kp1 = features.detect(img1)
    desc1 = features.compute(img1, kp1)
    kp2 = features.detect(img2)
    desc2 = features.compute(img2, kp2)

    # Find matches between the two images
    matches1 = match(desc1[1], desc2[1])
    matches2 = match(desc2[1], desc1[1])

    # Find cross-checked matches
    cross_check = []
    img_pt1 = []
    img_pt2 = []
    for m2 in matches2:
        for m1 in matches1:
            if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                img_pt1.append(kp1[m1.queryIdx].pt)
                img_pt2.append(kp2[m2.queryIdx].pt)
                cross_check.append(m2)
    print(f'Number of cross-checked matches left to right: {len(cross_check)}')

    # Convert keypoint lists to numpy arrays
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    # Compute the homography matrix
    M, mask = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)
    T = np.array([[1, 0, 300], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    M = np.matmul(T, M)

    # Warp the second image to align with the first image
    img2_warp = cv2.warpPerspective(
        img2, M, (img1.shape[1] + 8000, img1.shape[0] + 8000))
    img2_warp = crop(img2_warp)

    # Draw and display the matches
    dimg = cv2.drawMatches(img2, desc2[0], img1, desc1[0], cross_check, None)
    cv2.namedWindow('matchesL-R')
    cv2.imshow('matchesL-R', dimg)
    cv2.waitKey(0)

    # Display the warped image
    cv2.namedWindow('LeftToRight')
    cv2.imshow('LeftToRight', img2_warp)
    cv2.waitKey(0)

    # Detect and compute keypoints and descriptors for the warped image
    kp2_warp = features.detect(img2_warp)
    desc2_warp = features.compute(img2_warp, kp2_warp)

    # Find matches between the first image and the warped image
    matches1_2warp = match(desc1[1], desc2_warp[1])
    matches2warp_1 = match(desc2_warp[1], desc1[1])

    # Find cross-checked matches
    cross_check = []
    img_pt1 = []
    img_pt2_warp = []
    for m2 in matches2warp_1:
        for m1 in matches1_2warp:
            if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                img_pt1.append(kp1[m1.queryIdx].pt)
                img_pt2_warp.append(kp2_warp[m2.queryIdx].pt)
                cross_check.append(m2)
    print(
        f'Number of cross-checked matches left_warped to right: {len(cross_check)}')

    # Convert keypoint lists to numpy arrays
    img_pt1 = np.array(img_pt1)
    img_pt2_warp = np.array(img_pt2_warp)

    # Compute the homography matrix
    M, mask = cv2.findHomography(img_pt1, img_pt2_warp, cv2.RANSAC)

    # Draw and display the matches
    dimg = cv2.drawMatches(
        img2_warp, desc2_warp[0], img1, desc1[0], cross_check, None)
    cv2.namedWindow('matchesLwarped-R')
    cv2.imshow('matchesLwarped-R', dimg)
    cv2.waitKey(0)

    # Warp the first image to align with the second image
    img1_warp = cv2.warpPerspective(
        img1, M, (img2_warp.shape[1], img2_warp.shape[0]))

    # Blend the two images together
    img1_warp[img1_warp == 0] = img2_warp[img1_warp == 0]

    img1_warp = crop(img1_warp)

    # Display the final stitched image
    cv2.namedWindow('Stitched')
    cv2.imshow('Stitched', img1_warp)
    cv2.waitKey(0)

    # alternative way to blend images
    # img1_warp = cv2.warpPerspective(
    #     img1, M, (img2.shape[1] + 800, img2.shape[0] + 800))

    # for i in range(img2_warp.shape[0]):
    #     for j in range(img2_warp.shape[1]):
    #         if img2_warp[i, j] != 0:
    #             img1_warp[i, j] = img2_warp[i, j]
    # img1_warp = crop(img1_warp)

    # Save the final stitched image
    # cv2.imwrite('Stitched.jpg', img1_warp)
    return img1_warp


def stitching_last(img1, img2, features):  # img2 = lefToRIght, img1 = RightToLeft
    """
    Stitch two images together by aligning them using keypoint matching and computing a homography matrix.

    Parameters:
    img1 (ndarray): The first image to be stitched.
    img2 (ndarray): The second image to be stitched.

    Returns:
    ndarray: The stitched image.
    """

    # Resize images for faster processing
    img1 = resize(img1, 50)
    img2 = resize(img2, 50)

    # Detect and compute keypoints and descriptors for both images
    kp1 = features.detect(img1)
    desc1 = features.compute(img1, kp1)
    kp2 = features.detect(img2)
    desc2 = features.compute(img2, kp2)

    # Find matches between the two images
    matches1 = match(desc1[1], desc2[1])
    matches2 = match(desc2[1], desc1[1])

    # Find cross-checked matches
    cross_check = []
    img_pt1 = []
    img_pt2 = []
    for m2 in matches2:
        for m1 in matches1:
            if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                img_pt1.append(kp1[m1.queryIdx].pt)
                img_pt2.append(kp2[m2.queryIdx].pt)
                cross_check.append(m2)
    print(
        f'Number of cross-checked matches left to right (last): {len(cross_check)}')

    # Convert lists to numpy arrays
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    # Draw and display the matches
    dimg = cv2.drawMatches(img2, desc2[0], img1, desc1[0], cross_check, None)
    cv2.namedWindow('matchesLast')
    cv2.imshow('matchesLast', dimg)
    cv2.waitKey(0)

    # Compute the homography matrix
    M, mask = cv2.findHomography(img_pt1, img_pt2, cv2.RANSAC)

    # Warp the first image to align with the second image
    img = cv2.warpPerspective(
        img1, M, (img1.shape[1]+1000, img1.shape[0]+1000))
    # Overlap the two images
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if img2[i, j] != 0:
                img[i, j] = img2[i, j]

    # # Warp the first image to align with the second image
    # img = cv2.warpPerspective(
    #     img1, M, (img2.shape[1], img2.shape[0]))

    # # Blend the two images together
    # img[img == 0] = img2[img == 0]

    # Crop the final image
    img = crop(img)

    # Display the final stitched image
    cv2.namedWindow('Panorama', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Panorama', 1280, 720)
    cv2.imshow('Panorama', img)
    cv2.waitKey(0)

    return img


features = cv2.SIFT_create()

files = ['project_2/database/yard-00.png', 'project_2/database/yard-01.png', 'project_2/database/yard-02.png', 'project_2/database/yard-03.png',
         'project_2/database/my00.jpg', 'project_2/database/my01.jpg', 'project_2/database/my02.jpg', 'project_2/database/my03.jpg']  # add myphotos here

scale_factor_yard = 40
scale_factor_my = 20
images = []
for (i, img) in enumerate(files):
    if i < 4:
        images.append(
            resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), scale_factor_yard))
    else:
        images.append(
            resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), scale_factor_my))

Panorama1 = stitching_last(stitching_right_to_left(
    images[0], images[1], features), stitching_left_to_right(images[2], images[3], features), features)

Panorama2 = stitching_last(stitching_right_to_left(
    images[4], images[5], features), stitching_left_to_right(images[6], images[7], features), features)

# Save the final stitched images
cv2.imwrite(f'project_2/yard_panorama.jpg', Panorama1)
cv2.imwrite(f'project_2/city_panorama.jpg', Panorama2)
