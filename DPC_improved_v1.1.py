"""
File: dead_pixel_correction.py
Description: Corrects the hot or dead pixels
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: xx-isp (ispinfinite@gmail.com)
------------------------------------------------------------
"""

from timeit import default_timer as timer
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, correlate


class DeadPixelCorrection:
    "Dead Pixel Correction"

    def __init__(self, img, sensor_info, parm_dpc, platform):
        self.img = img
        self.enable = parm_dpc["isEnable"]
        self.sensor_info = sensor_info
        self.parm_dpc = parm_dpc
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.bpp = self.sensor_info["bitdep"]
        self.threshold = self.parm_dpc["dp_threshold"]
        self.is_debug = self.parm_dpc["isDebug"]

    def padding(self):
        """Return a mirror padded copy of image."""

        img_pad = np.pad(self.img, (2, 2), "reflect")
        return img_pad

    def apply_dead_pixel_correction(self):
        """This function detects and corrects Dead pixels."""
        start_time = timer()

        height, width = self.sensor_info["height"], self.sensor_info["width"]

        # Mirror padding is applied to self.img.
        img_padded = np.float32(self.padding())
        dpc_img = np.empty((height, width), np.float32)

        # Get 3x3 neighbourhood of each pixel.
        # 5x5 matrix is defined as this window is extarcted from raw image.
        window = np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1],
            ]
        )

        max_value = maximum_filter(img_padded, footprint=window)
        min_value = minimum_filter(img_padded, footprint=window)

        # Condition 1: center_pixel needs to be corrected if it lies outside the
        # interval(min_value,max) of the 3x3 neighbourhood.
        # min_value < center_pixel < max_value--> no correction needed
        mask_cond1 = (
            np.where(min_value < img_padded, False, True)
            + np.where(img_padded < max_value, False, True)
        ).astype("int32")

        # Condition 2:
        # center_pixel is corrected only if the difference of center_pixel and every
        # neighboring pixel is greater than the specified threshold.
        # The two if conditions are used in combination to reduce False positives.

        # Kernels to compute the difference between center pixel and
        # each of the 8 neighbours.
        ker_top_left = np.array(
            [
                [-1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_top_mid = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_top_right = np.array(
            [
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_mid_left = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_mid_right = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_bottom_left = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
            ]
        )
        ker_bottom_mid = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
            ]
        )
        ker_bottom_right = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )

        # convolve each kernel over image to compute differences
        diff_top_left = np.abs(correlate(img_padded, ker_top_left))
        diff_top_mid = np.abs(correlate(img_padded, ker_top_mid))
        diff_top_right = np.abs(correlate(img_padded, ker_top_right))
        diff_mid_left = np.abs(correlate(img_padded, ker_mid_left))
        diff_mid_right = np.abs(correlate(img_padded, ker_mid_right))
        diff_bottom_left = np.abs(correlate(img_padded, ker_bottom_left))
        diff_bottom_mid = np.abs(correlate(img_padded, ker_bottom_mid))
        diff_bottom_right = np.abs(correlate(img_padded, ker_bottom_right))

        del (
            ker_top_left,
            ker_top_mid,
            ker_top_right,
            ker_mid_left,
            ker_mid_right,
            ker_bottom_left,
            ker_bottom_mid,
            ker_bottom_right,
        )

        # Stack all arrays
        diff_array = np.stack(
            [
                diff_top_left,
                diff_top_mid,
                diff_top_right,
                diff_mid_left,
                diff_mid_right,
                diff_bottom_left,
                diff_bottom_mid,
                diff_bottom_right,
            ],
            axis=2,
        )

        del (
            diff_top_left,
            diff_top_mid,
            diff_top_right,
            diff_mid_left,
            diff_mid_right,
            diff_bottom_left,
            diff_bottom_mid,
            diff_bottom_right,
        )

        mask_cond2 = np.all(np.where(diff_array > self.threshold, True, False), axis=2)
        detection_mask = mask_cond1 * mask_cond2

        # Compute gradients
        ker_v = np.array([[-1, 0, 2, 0, -1]]).T
        ker_h = np.array([[-1, 0, 2, 0, -1]])
        ker_left_dia = np.array(
            [
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
            ]
        )
        ker_right_dia = np.array(
            [
                [-1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )
        vertical_grad = np.abs(correlate(img_padded, ker_v))
        horizontal_grad = np.abs(correlate(img_padded, ker_h))
        left_diagonal_grad = np.abs(correlate(img_padded, ker_left_dia))
        right_diagonal_grad = np.abs(correlate(img_padded, ker_right_dia))

        del ker_v, ker_h, ker_left_dia, ker_right_dia

        min_grad = np.min_value(
            np.stack(
                [
                    vertical_grad,
                    horizontal_grad,
                    left_diagonal_grad,
                    right_diagonal_grad,
                ],
                axis=2,
            ),
            axis=2,
        )

        # corrected value is computed as the mean of the neighbours
        # in the direction of min_value gadient.
        ker_mean_v = np.array([[1, 0, 0, 0, 1]]).T / 2
        ker_mean_h = np.array([[1, 0, 0, 0, 1]]) / 2
        ker_mean_ldia = (
            np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ]
            )
            / 2
        )
        ker_mean_rdia = (
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
            / 2
        )

        mean_v = correlate(img_padded, ker_mean_v)
        mean_h = correlate(img_padded, ker_mean_h)
        mean_ldia = correlate(img_padded, ker_mean_ldia)
        mean_rdia = correlate(img_padded, ker_mean_rdia)

        del ker_mean_v, ker_mean_h, ker_mean_ldia, ker_mean_rdia

        # Corrected image has the corrected pixel values in place of a detected dead pixel
        # and 0 elsewhere
        corrected_img = np.zeros(img_padded.shape)

        corrected_v = np.where(min_grad == vertical_grad, mean_v, 0) * detection_mask
        corrected_h = np.where(min_grad == horizontal_grad, mean_h, 0) * detection_mask
        corrected_ldia = (
            np.where(min_grad == left_diagonal_grad, mean_ldia, 0) * detection_mask
        )
        corrected_rdia = (
            np.where(min_grad == right_diagonal_grad, mean_rdia, 0) * detection_mask
        )

        # This block ensures that each pixel is only corrected once.
        corrected_img = corrected_img + corrected_v
        corrected_img = np.where(corrected_img == 0, corrected_h, corrected_img)
        corrected_img = np.where(corrected_img == 0, corrected_ldia, corrected_img)
        corrected_img = np.where(corrected_img == 0, corrected_rdia, corrected_img)

        del mean_h, mean_v, mean_ldia, mean_rdia
        del corrected_v, corrected_h, corrected_ldia, corrected_rdia

        dpc_img = np.where(detection_mask, corrected_img, img_padded)

        # Remove padding
        dpc_img = dpc_img[2:-2, 2:-2]
        self.img = np.uint16(np.clip(dpc_img, 0, (2**self.bpp) - 1))

        end_time = timer()
        print("Time taken to execute DPC: ", end_time - start_time)

        if self.is_debug:
            print(
                "   - Number of corrected pixels = ",
                np.count_nonzero(detection_mask[2:-2, 2:-2]),
            )
            print("   - Threshold = ", self.threshold)
        return self.img

    def execute(self):
        """Execute DPC Module"""
        print("Dead Pixel Correction = " + str(self.enable))

        if self.enable:
            return self.img

        return self.apply_dead_pixel_correction()
