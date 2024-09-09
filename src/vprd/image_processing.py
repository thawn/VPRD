from pathlib import Path
import numpy as np
import pyclesperanto as cle
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
from vprd.data import ensure_local, pixels_to_mev, pixels_to_fs, time_energy_aspect_ratio, GlobalStandardScaler
# from vprd.data import mev_to_J as convert_mev  # calculate potential in V
from vprd.data import nanocoulomb_to_coulomb as convert_nanocoulomb  # calculate charge in C
# from vprd.data import nanocoulomb_to_e as convert_nanocoulomb  # calculate charge in e
from vprd.data import mev_to_ev as convert_mev  # calculate potential in eV
from typing import List, Tuple
from scipy.optimize import curve_fit


def show_phase_image(image, ax: matplotlib.axes.Axes = None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel(r'$\Delta$ energy (MeV)')
    extent = [0, pixels_to_fs(image.shape[1]), 0, pixels_to_mev(image.shape[0])]
    ax.imshow(image, extent=extent, aspect=time_energy_aspect_ratio, **kwargs)


def destripe_image(image: np.ndarray, sigma: float = 100) -> np.ndarray:
    """
    Destripes the given image. The striped bacground is estimated by applying a highly assymetric Gaussian blur and a top-hat filter to the image.

    Parameters
    ----------
    image: numpy.ndarray
        The input image to be destriped.

    Returns
    -------
    numpy.ndarray
        The destriped image.
    """
    destriping_background = cle.gaussian_blur(image, sigma_x=0, sigma_y=sigma)
    destriping_background = cle.top_hat(destriping_background, radius_x=1, radius_y=0)
    return cle.subtract_images(image, destriping_background)


def measure_bounding_box_size(image: np.ndarray, padding: int = 50, min_radius: int = 5) -> np.ndarray:
    """
    Measures the size of the bounding box around objects in the given image.

    The bounding box always has the full height of the phase image, because we only need to de-jitter the time axis.

    Parameters
    ----------
    image: numpy.ndarray
        The input image to measure the bounding box size.
    padding: int, optional
        The amount of padding to add to the objects before measuring the bounding box size. Default is 50 pixels.
    min_radius: int, optional
        The minimum radius of objects to keep in the image. Objects with a radius smaller than this value will be removed. Default is 5 pixels.

    Returns
    -------
    numpy.ndarray
        The bounding box coordinates [x1, y1, z1, x2, y2, z2] of the objects in the image after applying padding.
    """
    blurred_image = cle.gaussian_blur(image, sigma_x=2, sigma_y=2)
    thresholded = cle.threshold_otsu(blurred_image)
    # removes objects with a radius of less than `min_radius`
    eroded = cle.minimum(thresholded, radius_x=min_radius, radius_y=min_radius)
    new_radius = min_radius + padding  # compensate for shrinking objects by min_radius in the previous step
    dilated = cle.maximum(eroded, radius_x=new_radius, radius_y=new_radius)  # grows objects by `padding` pixels
    bounding_box = np.asarray(cle.bounding_box(dilated)).astype(int)
    # we need the full height, because that is the energy axis
    bounding_box[1] = 0
    bounding_box[4] = image.shape[0]
    return bounding_box


def calculate_bounding_box_width_height(bounding_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the widths and heights of bounding boxes.

    Parameters
    ----------
    bounding_boxes: numpy.ndarray
        An array of bounding boxes in the format [x1, y1, z1, x2, y2, z2].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the widths and heights of the bounding boxes.
    """

    widths = bounding_boxes[:, 3] - bounding_boxes[:, 0]
    heights = bounding_boxes[:, 4] - bounding_boxes[:, 1]
    return widths, heights


def print_min_max_sizes(widths: np.ndarray, heights: np.ndarray):
    """
    Prints the minimum and maximum bounding box sizes.

    Parameters
    ----------
    widths: numpy.ndarray
        An array of bounding box widths.
    heights: numpy.ndarray
        An array of bounding box heights.
    """
    print(f"min bounding box: height: {heights.min()}, width: {widths.min()}")
    print(f"max bounding box: height: {heights.max()}, width: {widths.max()}")


def normalize_bounding_box(box: np.ndarray, target_height: int, target_width: int,
                           max_height: int, max_width: int) -> np.ndarray:
    """
    Normalize the bounding box to a target width and height.

    Parameters
    ----------
    box: numpy.ndarray
        The bounding box coordinates [x1, y1, z1, x2, y2, z2].
    target_width: int
        The desired width of the bounding box.
    target_height: int
        The desired height of the bounding box.

    Returns
    -------
    numpy.ndarray
        The normalized bounding box coordinates [x1, y1, z1, x2, y2, z2].

    """
    # get the size of the bounding box
    width = box[3] - box[0]
    height = box[4] - box[1]

    assert width <= max_width, "Bounding box width is larger than the image width."
    assert height <= max_height, "Bounding box height is larger than the image height."

    # calculate the number of pixels to add or remove
    delta_width = target_width - width
    delta_height = target_height - height

    # calculate the new bounding box
    new_box = np.array(box)
    new_box[0] -= delta_width / 2
    new_box[1] -= delta_height / 2
    new_box[3] += delta_width / 2
    new_box[4] += delta_height / 2

    # shift the bounding box to the right if it's too far to the left
    if new_box[0] < 0:
        new_box[3] -= new_box[0]
        new_box[0] = 0

    # shift the bounding box to the left if it's too far to the right
    if new_box[3] > max_width:
        new_box[0] -= new_box[3] - max_width
        new_box[3] = max_width

    # shift the bounding box down if it's too far to the top
    if new_box[1] < 0:
        new_box[4] -= new_box[1]
        new_box[1] = 0

    # shift the bounding box up if it's too far to the bottom
    if new_box[4] > max_height:
        new_box[1] -= new_box[4] - max_height
        new_box[4] = max_height

    return new_box.astype(int)


def crop_image_with_bounding_box(image: np.ndarray, bounding_box: np.ndarray) -> np.ndarray:
    """
    Crop the given image using the provided bounding box.

    Parameters
    ----------
    image: numpy.ndarray
        The input image to be cropped.
    bounding_box: numpy.ndarray
        A list or tuple containing the coordinates of the bounding box in the format [x1, y1, z1, x2, y2, z2].

    Returns
    -------
    numpy.array
        The cropped image.
    """
    return image[bounding_box[1]:bounding_box[4], bounding_box[0]:bounding_box[3]]


def verify_image_sizes(images: List[np.ndarray], target_height: int, target_width: int) -> bool:
    """
    Verifies if the images in the given list have the specified target width and height.

    Parameters
    ----------
    images: List[np.ndarray]
        A list of images to be verified.
    target_width: int
        The target width of the images.
    target_height: int
        The target height of the images.

    Returns
    -------
    bool
        True if all images have the correct size, False otherwise.
    """
    ok = True
    for i, img in enumerate(images):
        if img.shape[0] != target_height or img.shape[1] != target_width:
            print(f"Image {i} has wrong size: {img.shape}")
            ok = False
    return ok


def calculate_target_shape(bounding_boxes: List[np.ndarray]) -> Tuple[int, int]:
    """
    Calculate the target width and height of the bounding box for the given list of bounding boxes so that the largest signal still fits in the bounding box.

    Parameters
    ----------
    bounding_boxes: List[np.ndarray]
        A list of bounding boxes to calculate the target width and height for.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the target width and height of the bounding box.
    """

    widths, heights = calculate_bounding_box_width_height(bounding_boxes)
    target_width = np.max(widths)
    target_height = np.max(heights)
    return target_height, target_width


def extract_normalized_bounding_boxes(images: List[np.ndarray], padding: int = 50, min_radius: int = 5) -> np.ndarray:
    """
    Extracts normalized bounding boxes from the given images.

    Parameters
    ----------
    images: List[np.ndarray]
        A list of input images as numpy arrays.

    Returns
    -------
    np.ndarray
        An array containing the normalized bounding boxes of the input images.

    """
    bounding_boxes = np.array([measure_bounding_box_size(img, padding, min_radius) for img in images])
    target_shape = calculate_target_shape(bounding_boxes)
    normalized_bounding_boxes = np.array([normalize_bounding_box(
        box, *target_shape, *images[i].shape) for i, box in enumerate(bounding_boxes)])
    return normalized_bounding_boxes


def extract_relevant_image_parts(images: List[np.ndarray]) -> np.ndarray:
    """
    Extracts the relevant parts of the input images based on their bounding boxes.

    Parameters
    ----------
    images: List[np.ndarray]
        A list of input images as numpy arrays.

    Returns
    -------
    np.ndarray
        An array containing the extracted relevant parts of the input images.

    Raises
    ------
    AssertionError
        If the extracted images have incorrect sizes.

    """
    normalized_bounding_boxes = extract_normalized_bounding_boxes(images)
    # crop the image
    normalized_images = [
        crop_image_with_bounding_box(
            img, normalized_bounding_box) for img, normalized_bounding_box in zip(
            images, normalized_bounding_boxes)]

    assert verify_image_sizes(
        normalized_images, *normalized_images[0].shape), "Images have wrong sizes."

    return np.array(normalized_images)


def extract_relevant_image_parts_from_numpy_file(image_file_path: Path) -> List[np.ndarray]:
    """
    Extracts relevant image parts from the given numpy image file.

    Parameters
    ----------
    image_file_path: pathlib.Path
        The path to the image file.

    Returns
    -------
    List[np.ndarray]
        A list of cropped images.
    """
    ensure_local(image_file_path)

    images = np.load(image_file_path, allow_pickle=True)

    image_list = [img['data'] for img in images]
    normalized_images = extract_relevant_image_parts(image_list)
    normalized_images_with_ids = [{'TrainId_image': img['TrainId_image'],
                                   'data': normalized_images[i].astype(np.uint8)} for i, img in enumerate(images)]
    return normalized_images_with_ids

####################################################################################################
# Extract energy data from images
####################################################################################################


def calibrated_lineplot(data: np.ndarray, ylabel: str = r'$\Delta$ energy (MeV)', **kwargs) -> np.ndarray:
    time_data = pixels_to_fs(np.arange(data.shape[0]))
    ax = sns.lineplot(x=time_data, y=data, **kwargs)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel(ylabel)
    return ax


def gaussian(x: np.ndarray, a: float, x0: float, sigma: float, y0: float) -> np.ndarray:
    """
    Calculate the value of a Gaussian function at a given point.

    Parameters
    ----------
    x: numpy.ndarray
        The input values.
    a: float
        The amplitude of the Gaussian curve.
    x0: float
        The mean of the Gaussian curve.
    sigma: float
        The standard deviation of the Gaussian curve.
    y0: float
        The y-offset of the Gaussian curve.

    Returns
    -------
    numpy.ndarray
        The values of the Gaussian function at the given points.
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + y0


def fit_gaussian(data: np.ndarray) -> np.ndarray:
    """
    Fits a Gaussian curve to the given data. Returns the optimized parameters of the Gaussian curve.

    Returns the amplitude, mean, standard deviation, and y-offset of the Gaussian curve.

    Parameters
    ----------
    data: numpy.ndarray
        The input data to fit the Gaussian curve to.

    Returns
    -------
    numpy.ndarray
        The optimized parameters of the Gaussian curve.

    """
    x = np.arange(len(data))
    a = np.max(data)  # amplitude
    x0 = np.argmax(data)  # mean
    sigma = 1  # standard deviation
    y0 = np.min(data)  # y-offset
    popt, pcov = curve_fit(gaussian, x, data, p0=[a, x0, sigma, y0])
    return popt


def fit_columns_within_bounding_box(image: np.ndarray, measured_charge: float,
                                    bounding_box: np.ndarray, average_width: int = 0) -> np.ndarray:
    """
    Fits a Gaussian distribution to the data in each column within a given bounding box of an image.

    Parameters
    ----------
    image: numpy.ndarray
        The input image.
    bounding_box:
        The bounding box coordinates [x1, y1, z1, x2, y2, z2] within which to fit the columns.
    average_width: int, optional
        The width of the window used to calculate the average data. Default is 0 (do not average).

    Returns
    -------
    numpy.ndarray
        An array of means, standard deviations and sum for each column within the bounding box.
    """
    destriped_image = destripe_image(image.astype(np.float32))
    destriped_image = np.asarray(calibrate_charge(destriped_image, measured_charge))
    means_stds_sums = np.zeros((bounding_box[3] - bounding_box[0], 3))
    average_offset = int((average_width - 1) / 2)
    # get the columns within the bounding box
    columns = np.arange(bounding_box[0], bounding_box[3])
    for i, column in enumerate(columns):
        # fit a Gaussian to the data in the column
        if average_offset > 0:
            data = destriped_image[:, column - average_offset:column + average_offset].mean(axis=1)
        else:
            data = destriped_image[:, column]
        try:
            popt = fit_gaussian(data)
        except RuntimeError:
            popt = np.zeros(4)
            popt[:] = np.nan
        means_stds_sums[i, 0:2] = popt[1:3]
        means_stds_sums[i, 2] = data.sum() / pixels_to_fs(1e-15)  # sum the energy data
    # flip the mean energy data
    means_stds_sums[:, 0] = destriped_image.shape[0] - means_stds_sums[:, 0]
    # calibrate the mean energy data to MeV
    means_stds_sums[:, 0:2] = pixels_to_mev(means_stds_sums[:, 0:2])
    return means_stds_sums


def get_total_intensity_count(image: cle.Array) -> float:
    """
    Calculates the total intensity count of the given image.
    Ignores background noise by applying a Gaussian blur and Otsu thresholding.

    Parameters
    ----------
    image: numpy.ndarray
        The input image.

    Returns
    -------
    float
        The total intensity count of the image.
    """
    blurred = cle.gaussian_blur(image, sigma_x=2.0, sigma_y=2.0)
    thresholded = cle.threshold_otsu(blurred)
    masked = cle.mask(image, thresholded)
    intensity_count = masked.sum()
    return intensity_count


def charge_to_coulomb(charge: np.ndarray) -> np.ndarray:
    """
    Converts the given charge in nanocoulomb to Coulomb.

    Parameters
    ----------
    charge: np.ndarray
        The charge in nanocoulomb.

    Returns
    -------
    np.ndarray
        The charge in Coulomb.
    """
    return charge / convert_nanocoulomb(1) / 1e9


def calculate_charge_calibration_factor(total_count: float, measured_charge: float) -> float:
    """
    Calculates the charge calibration factor for the given total intensity count and measured charge.

    Parameters
    ----------
    total_count: float
        The total intensity count of the image.
    measured_charge: float
        The measured charge of the image.

    Returns
    -------
    float
        The charge calibration factor (in number of electrons per count).
    """
    return convert_nanocoulomb(measured_charge) / total_count


def calculate_charge_calibration_factors(images: List[np.ndarray], measured_charge: List[float]) -> List[float]:
    """
    Calculates the charge calibration factors for the given images.

    Parameters
    ----------
    images: List[np.ndarray]
        A list of images to calculate the charge calibration factors for.
    measured_charge: List[float]
        A list of measured charges for the images.

    Returns
    -------
    List[float]
        A list of charge calibration factors (in number of electrons per count) for the images.
    """
    total_counts = [get_total_intensity_count(img) for img in images]
    return [calculate_charge_calibration_factor(count, charge) for count, charge in zip(total_counts, measured_charge)]


def calibrate_charge(image: np.ndarray, measured_charge: float) -> np.ndarray:
    """
    Calibrates the charge of the given image.

    Parameters
    ----------
    image: numpy.ndarray
        The input image to calibrate.
    measured_charge: float
        The measured charge of the image.

    Returns
    -------
    numpy.ndarray
        The calibrated image in Coulomb per pixel.
    """
    image = cle.push(image.astype(np.float32))
    total_count = get_total_intensity_count(image)
    charge_per_count = calculate_charge_calibration_factor(total_count, measured_charge)
    return cle.multiply_image_and_scalar(image, scalar=charge_per_count)


def calculate_electron_power(image: np.ndarray, measured_charge: float) -> np.ndarray:
    """
    Calculates the total power along the time axis of the given phase image in W.

    Parameters
    ----------
    image: numpy.ndarray
        The input phase image (axis[0] = time, axis[1] = energy).
    measured_charge: float
        The measured charge of the entire electron bunch in nanocoulomb (nC).

    Returns
    -------
    np.ndarray
        The electron power in W.
    """
    # push to GPU so it does not happen twice in the next steps
    image = cle.push(image.astype(np.float32))

    # weigh the image by the energy axis (pixel units)
    # the energy axis is flipped with respect to the image coordinates
    flipped_image = cle.flip(image, flip_x=False, flip_y=True)
    energy_pixels_weighted_pixels = cle.multiply_image_and_position(flipped_image, dimension=1)

    # sum the energy weighted image along the energy axis (pixel units)
    energy_pixels_weighted_sums_pixels = np.asarray(energy_pixels_weighted_pixels).sum(axis=0)

    # calculate the electrons per count
    total_count = get_total_intensity_count(image)
    charge_calibration_factor = calculate_charge_calibration_factor(total_count, measured_charge)
    current_factor = charge_calibration_factor / pixels_to_fs(1e-15)
    potential_factor = convert_mev(pixels_to_mev(1))
    calibration_factor = potential_factor * current_factor  # Behrens et al. 2014
    return energy_pixels_weighted_sums_pixels * calibration_factor


def extract_electron_power(images: List[np.ndarray], charges: List[float], padding: int = 10) -> np.ndarray:
    """
    Extracts power data from the given list of images.

    Parameters
    ----------
    images: List[np.ndarray]
        A list of images to extract power data from.
    padding: int, optional
        The amount of padding to add to the bounding box. Default is 10 pixels

    Returns
    -------
    numpy.ndarray:
        A 2D array where each row ist the relevant part of the electron power for each image.
    """

    # calculate energy curves from the phase images
    energy = np.array([calculate_electron_power(destripe_image(img.astype(np.float32)), charge)
                       for img, charge in zip(images, charges)])

    # de-jitter: align the energy curves by the peak location
    blurred = cle.gaussian_blur(energy, sigma_x=10.0, sigma_y=0.0)
    peaks = np.array([np.argmax(row) for row in blurred])
    median_peak_location = int(np.median(peaks))
    offsets = peaks - median_peak_location
    aligned_energy = np.array([np.roll(row, -offset) for row, offset in zip(energy, offsets)])

    # crop the energy curves
    # workaround for bug #3333 scale the image to a suitable integer range to make the thresholding work
    scaled_energy = cle.add_image_and_scalar(aligned_energy, scalar=-cle.minimum_of_all_pixels(aligned_energy))
    scaled_energy = cle.multiply_image_and_scalar(scaled_energy, scalar=255 / cle.maximum_of_all_pixels(scaled_energy))
    thresholded = cle.threshold_otsu(scaled_energy)
    bounding_box = [int(v) for v in cle.bounding_box(thresholded)]
    return aligned_energy[:, bounding_box[0] - padding:bounding_box[3] + padding]


def align_signals(lasing_off: np.ndarray, lasing_on: np.ndarray, peak_denoising_sigma: float = 10.0) -> int:
    """
    Aligns two signals by their peak positions after applying Gaussian blur.

    Used to align individual lasing on and off signals before subtracting them to calculate the photon power.
    Ensures that both arrays have the same length.

    Parameters:
    lasing_off (np.ndarray): The signal array when the laser is off.
    lasing_on (np.ndarray): The signal array when the laser is on.
    peak_denoising_sigma (float, optional): The sigma value for the Gaussian blur applied to denoise the peaks. Default is 10.0.

    Returns:
    tuple: A tuple containing the aligned lasing_off and lasing_on arrays.
    """
    lasing_on_mode = np.argmax(cle.gaussian_blur(lasing_on, sigma_x=peak_denoising_sigma))
    lasing_off_mode = np.argmax(cle.gaussian_blur(lasing_off, sigma_x=peak_denoising_sigma))
    offset = (lasing_off_mode - lasing_on_mode) // 2
    lasing_off = np.roll(lasing_off, -offset)
    lasing_on = np.roll(lasing_on, offset)
    if len(lasing_off) > len(lasing_on):
        lasing_off = lasing_off[:len(lasing_on)]
    else:
        lasing_on = lasing_on[:len(lasing_off)]
    return lasing_off, lasing_on


def extract_energy_data_from_images(images: List[np.ndarray]) -> np.ndarray:
    """
    Extracts energy data from the given images.

    Parameters
    ----------
    images: List[np.ndarray]
        A list of images to extract energy data from.

    Returns
    -------
    numpy.ndarray:
        An array of means and standard deviations for each column within the bounding box of each image.
    """
    normalized_bounding_boxes = extract_normalized_bounding_boxes(images, padding=0, min_radius=5)
    energy_data = np.array([fit_columns_within_bounding_box(img, box)
                           for img, box in zip(images, normalized_bounding_boxes)])
    return energy_data
