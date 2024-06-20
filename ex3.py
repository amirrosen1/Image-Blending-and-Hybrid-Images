import cv2
import numpy as np


def generate_gaussian_pyramid(img, levels):
    """
    Generates a Gaussian pyramid for an image.
    :param img: An image to generate the pyramid for.
    :param levels: Number of levels in the pyramid.
    :return: A list of images representing the Gaussian pyramid.
    """

    # Convert the image to float32
    img = img.astype(np.float32)
    gaussian_pyramid = [img]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        gaussian_pyramid.append(img)
    return gaussian_pyramid


def unified_laplacian_pyramid(input, levels=None):
    """
    Generates a Laplacian pyramid either directly from an image or from a pre-generated Gaussian pyramid.
    :param input: An image or a pre-generated Gaussian pyramid.
    :param levels: Number of levels in the pyramid if the input is an image. Ignored if input is a Gaussian pyramid.
    :return: A list of images representing the Laplacian pyramid.
    """

    # If the input is an image, generate a Gaussian pyramid first and then a Laplacian pyramid from it,
    # otherwise assume the input is a Gaussian pyramid and generate a Laplacian pyramid directly.
    if isinstance(input, np.ndarray):
        gaussian_pyramid = generate_gaussian_pyramid(input, levels)
    else:
        gaussian_pyramid = input
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        higher = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], higher)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def blend_pyramids(lap_pyramid1, lap_pyramid2, mask_pyramid):
    """
    Blends two Laplacian pyramids based on a mask pyramid.
    :param lap_pyramid1: A Laplacian pyramid of the first image.
    :param lap_pyramid2: A Laplacian pyramid of the second image.
    :param mask_pyramid: A Gaussian pyramid of the mask.
    :return: A blended Laplacian pyramid.
    """

    blended_pyramid = []
    for i in range(len(lap_pyramid1)):
        lap1 = lap_pyramid1[i]
        lap2 = lap_pyramid2[i]
        mask_image = mask_pyramid[i]
        mask_resized = cv2.resize(mask_image, (lap1.shape[1], lap1.shape[0]), interpolation=cv2.INTER_LINEAR)
        # If the mask is not 3 channels, make it so by repeating the last dimension.
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]
        # If the mask has only one channel, repeat it three times to make it 3 channels.
        if mask_resized.shape[2] == 1:
            mask_resized = np.repeat(mask_resized, 3, axis=2)
        # Blend the layers with the resized mask.
        blended = lap1 * mask_resized + lap2 * (1 - mask_resized)
        blended_pyramid.append(blended)
    return blended_pyramid


def combine_pyramids(lap_pyramid1, lap_pyramid2, cutoff_level):
    """
    Combines two Laplacian pyramids based on a cutoff level.
    :param lap_pyramid1: A Laplacian pyramid of the first image.
    :param lap_pyramid2: A Laplacian pyramid of the second image.
    :param cutoff_level: A level at which to switch from the first to the second pyramid.
    :return: A combined Laplacian pyramid.
    """

    combined_pyramid = []
    for i in range(len(lap_pyramid1)):
        # If the current level is less than the cutoff level, use the first pyramid, otherwise use the second pyramid.
        if i < cutoff_level:
            combined_pyramid.append(lap_pyramid1[i])
        else:
            combined_pyramid.append(lap_pyramid2[i])
    return combined_pyramid


def blend_or_combine_pyramids(lap_pyramid1, lap_pyramid2, method="blend", mask_pyramid=None, cutoff_level=None):
    """
    A universal function to blend or combine two Laplacian pyramids based on the specified method.
    :param lap_pyramid1: A Laplacian pyramid of the first image.
    :param lap_pyramid2: A Laplacian pyramid of the second image.
    :param method: A string specifying the operation to perform on the pyramids. Can be 'blend' or 'combine'.
    :param mask_pyramid: A Gaussian pyramid of the mask, used if method is 'blend'.
    :param cutoff_level: A level at which to switch from the first to the second pyramid, used if method is 'combine'.
    :return: Resulting Laplacian pyramid based on the specified method.
    """

    # Choose the appropriate method based on the input parameters.
    if method == "blend" and mask_pyramid is not None:
        return blend_pyramids(lap_pyramid1, lap_pyramid2, mask_pyramid)
    elif method == "combine" and cutoff_level is not None:
        return combine_pyramids(lap_pyramid1, lap_pyramid2, cutoff_level)


def reconstruct_from_laplacian_pyramid(lap_pyramid):
    """
    Reconstructs an image from its Laplacian pyramid.
    :param lap_pyramid: A Laplacian pyramid to reconstruct the image from.
    :return: A reconstructed image.
    """

    # Start with the smallest image in the pyramid.
    reconstructed_image = lap_pyramid[-1]
    for i in range(len(lap_pyramid) - 2, -1, -1):
        height, width = lap_pyramid[i].shape[:2]
        up = cv2.pyrUp(reconstructed_image, dstsize=(width, height))
        reconstructed_image = up + lap_pyramid[i]
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    return reconstructed_image


def blend_images(image1, image2, mask_image):
    """
    Blends two images based on a mask using Laplacian and Gaussian pyramids.
    :param image1: A first image to blend.
    :param image2: A second image to blend.
    :param mask_image: A mask to use for blending.
    :return: A blended image.
    """

    # Ensure the second image and the mask are the same size as the first image.
    levels = 6
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    mask_image = cv2.resize(mask_image, (width, height))
    if mask_image.ndim == 2:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    mask_image = mask_image / 255.0
    mask_pyramid = generate_gaussian_pyramid(mask_image, levels)
    lap_pyramid1 = unified_laplacian_pyramid(image1, levels)
    lap_pyramid2 = unified_laplacian_pyramid(image2, levels)
    blended_pyramid = blend_pyramids(lap_pyramid1, lap_pyramid2, mask_pyramid)
    blended_image_result = reconstruct_from_laplacian_pyramid(blended_pyramid)
    return np.clip(blended_image_result, 0, 255).astype(np.uint8)


def hybrid_images(image1, image2):
    """
    Generates a hybrid image from two images using Laplacian and Gaussian pyramids.
    :param image1: A first image to use for the hybrid.
    :param image2: A second image to use for the hybrid.
    :return: A hybrid image.
    """

    # Ensure the second image is the same size as the first image.
    levels = 6
    cut_off = 3
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    gp1 = generate_gaussian_pyramid(image1, levels)
    gp2 = generate_gaussian_pyramid(image2, levels)
    lp1 = unified_laplacian_pyramid(gp1)
    lp2 = unified_laplacian_pyramid(gp2)
    # Combine the pyramids based on the cutoff level.
    combined_pyramid = combine_pyramids(lp1, lp2, cut_off)
    # Reconstruct the hybrid image from the combined pyramid.
    hybrid_image_result = reconstruct_from_laplacian_pyramid(combined_pyramid)
    # Convert the hybrid image to grayscale.
    hybrid_image_result = cv2.cvtColor(hybrid_image_result, cv2.COLOR_BGR2GRAY)
    return hybrid_image_result


first_image_for_blending = cv2.imread('Barcelona.jpg')
second_image_for_blend = cv2.imread('scarynight.png')
mask = cv2.imread('maskbarcelonainvert.jpg')
blended_image = blend_images(first_image_for_blending, second_image_for_blend, mask)
cv2.imwrite('blended_image_barcelona_final.jpg', blended_image)


first_image_for_hybrid = cv2.imread('ed_sheeren.jpg')
second_image_for_hybrid = cv2.imread('brown_dog.jpg')
hybrid_image = hybrid_images(first_image_for_hybrid, second_image_for_hybrid)
cv2.imwrite('hybrid_image_final.jpg', hybrid_image)
