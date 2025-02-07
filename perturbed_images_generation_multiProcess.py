import argparse
import os
import pickle
import random
import numpy as np
import cv2
import time
from multiprocessing import Pool
import scipy.spatial.qhull as qhull  # Now we need this for Delaunay


def getDatasets(dir):
    """
    Returns a list of filenames in the given directory.

    Args:
        dir: The directory path.

    Returns:
        A list of strings representing the filenames.
    """
    return os.listdir(dir)

class perturbed(object):
    """
    Class to handle the image perturbation process.
    """
    def __init__(self, path, bg_path, save_path, save_suffix):
        """
        Initializes the perturbed object.

        Args:
            path: Path to the input image.
            bg_path: Path to the directory containing background images.
            save_path: Path to save the output images and data.
            save_suffix: Suffix for the output filenames.
        """
        self.path = path
        self.bg_path = bg_path
        self.save_path = save_path
        self.save_suffix = save_suffix

    def get_normalize(self, d):
        """
        Normalizes the input array by subtracting the mean and dividing by the standard deviation.

        Args:
            d: The input NumPy array.

        Returns:
            The normalized array.
        """
        E = np.mean(d)
        std = np.std(d)
        d = (d - E) / std
        return d

    def get_0_1_d(self, d, new_max=1, new_min=0):
        """
        Rescales the input array to the range [new_min, new_max].

        Args:
            d: The input NumPy array.
            new_max: The desired maximum value (default: 1).
            new_min: The desired minimum value (default: 0).

        Returns:
            The rescaled array.
        """
        d_min = np.min(d)
        d_max = np.max(d)
        d = ((d - d_min) / (d_max - d_min)) * (new_max - new_min) + new_min
        return d


    def get_pixel(self, p, origin_img):
        """
        Safely retrieves a pixel value from the image, handling out-of-bounds access.

        Args:
            p: The pixel coordinates (row, col) as a list or array.
            origin_img: The image as a NumPy array.

        Returns:
            The pixel value as a NumPy array, or [257, 257, 257] if out of bounds.
        """
        try:
            return origin_img[p[0], p[1]]
        except IndexError:
            return np.array([257, 257, 257])  # Use IndexError for clarity

    def get_coor(self, p, origin_label):
        """
        Safely retrieves coordinates from a label image, handling out-of-bounds access.

        Args:
            p: The pixel coordinates (row, col) as a list or array.
            origin_label: The label image as a NumPy array.

        Returns:
            The coordinates as a NumPy array, or [0, 0] if out of bounds.
        """
        try:
            return origin_label[p[0], p[1]]
        except IndexError:
            return np.array([0, 0])


    def is_perform(self, execution, inexecution):
        """
        Randomly decides whether to perform an action based on given probabilities.

        Args:
            execution: Probability of performing the action.
            inexecution: Probability of not performing the action.

        Returns:
            True if the action should be performed, False otherwise.
        """
        return random.choices([True, False], weights=[execution, inexecution])[0]


    def adjust_position_v2(self, x_min, y_min, x_max, y_max, new_shape):
        """
        Calculates padding to center a smaller region within a larger shape.

        Args:
            x_min: Minimum x-coordinate of the region.
            y_min: Minimum y-coordinate of the region.
            x_max: Maximum x-coordinate of the region.
            y_max: Maximum y-coordinate of the region.
            new_shape: The shape of the larger region (height, width).

        Returns:
            A tuple (f_g_0_0, f_g_1_0, new_x_max, new_y_max) representing the
            padding amounts and the adjusted maximum coordinates.
        """
        if (new_shape[0] - (x_max - x_min)) % 2 == 0:
            f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
            f_g_0_1 = f_g_0_0
        else:
            f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
            f_g_0_1 = f_g_0_0 + 1

        if (new_shape[1] - (y_max - y_min)) % 2 == 0:
            f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
            f_g_1_1 = f_g_1_0
        else:
            f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
            f_g_1_1 = f_g_1_0 + 1

        return f_g_0_0, f_g_1_0, new_shape[0] - f_g_0_1, new_shape[1] - f_g_1_1

    def adjust_position(self, x_min, y_min, x_max, y_max):
      """
      Adjust position
      """
      if (self.new_shape[0] - (x_max - x_min)) % 2 == 0:
          f_g_0_0 = (self.new_shape[0] - (x_max - x_min)) // 2
          f_g_0_1 = f_g_0_0
      else:
          f_g_0_0 = (self.new_shape[0] - (x_max - x_min)) // 2
          f_g_0_1 = f_g_0_0 + 1
      if (self.new_shape[1] - (y_max - y_min)) % 2 == 0:
          f_g_1_0 = (self.new_shape[1] - (y_max - y_min)) // 2
          f_g_1_1 = f_g_1_0
      else:
          f_g_1_0 = (self.new_shape[1] - (y_max - y_min)) // 2
          f_g_1_1 = f_g_1_0 + 1
      return f_g_0_0, f_g_1_0, self.new_shape[0] - f_g_0_1, self.new_shape[1] - f_g_1_1

    def interp_weights(self, xyz, uvw):
        """
        Calculates interpolation weights using Delaunay triangulation.

        Args:
            xyz: Coordinates of the triangle vertices.
            uvw: Coordinates of the point to interpolate.

        Returns:
            A tuple (vertices, weights) where vertices are the indices of the
            triangle vertices and weights are the barycentric coordinates.
        """
        tri = qhull.Delaunay(xyz)
        simplex = tri.find_simplex(uvw)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uvw - temp[:, 2]
        bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def interpolate(self, values, vtx, wts):
        """
        Performs interpolation using the given values, vertices, and weights.

        Args:
            values: Values at the vertices.
            vtx: Indices of the vertices.
            wts: Barycentric weights.

        Returns:
            The interpolated value.
        """
        return np.einsum('njk,nj->nk', np.take(values, vtx, axis=0), wts)

    def save_img(self, m, n, fold_curve='fold', repeat_time=4, relativeShift_position='relativeShift_v2'):
        """
        Main function to generate and save perturbed images.

        Args:
            m: Index of the input image.
            n: Index for the perturbation iteration.
            fold_curve: Type of perturbation ('fold' or 'curve').
            repeat_time: Number of perturbation iterations.
            relativeShift_position:  Method for applying shifts ('relativeShift_v2').
        """
        begin_time = time.time()  # Track execution time

        origin_img = cv2.imread(self.path, flags=cv2.IMREAD_COLOR)
        if origin_img is None:
            print(f"Error: Could not read image at {self.path}")
            return

        # --- Image Resizing (Preserving Original Logic) ---
        save_img_shape = [512*2, 480*2]
        reduce_value = np.random.choice([8*2, 16*2, 24*2, 32*2, 40*2, 48*2], p=[0.1, 0.2, 0.4, 0.1, 0.1, 0.1])
        base_img_shrink = save_img_shape[0] - reduce_value
        enlarge_img_shrink = [896*2, 768*2]

        im_lr = origin_img.shape[0]
        im_ud = origin_img.shape[1]
        aspect_ratio = round(im_lr / im_ud, 2)

        reduce_value_v2 = np.random.choice([4*2, 8*2, 16*2, 24*2, 28*2, 32*2, 48*2, 64*2], p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.08, 0.02])

        if im_lr > im_ud and aspect_ratio > 1.2:
            im_ud = min(int(im_ud / im_lr * base_img_shrink), save_img_shape[1] - reduce_value_v2)
            im_lr = save_img_shape[0] - reduce_value
        else:
            base_img_shrink = save_img_shape[1] - reduce_value
            im_lr = min(int(im_lr / im_ud * base_img_shrink), save_img_shape[0] - reduce_value_v2)
            im_ud = base_img_shrink

        self.origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)

        # --- Background Image Loading (Preserving Original Logic) ---
        perturbed_bg_filenames = getDatasets(self.bg_path)
        if not perturbed_bg_filenames:
            print(f"Error: No background images found in {self.bg_path}")
            return
        perturbed_bg_img_path = os.path.join(self.bg_path, random.choice(perturbed_bg_filenames))
        perturbed_bg_img = cv2.imread(perturbed_bg_img_path, flags=cv2.IMREAD_COLOR)
        if perturbed_bg_img is None:
            print(f"Error: Could not read background image at {perturbed_bg_img_path}")
            return

        perturbed_bg_img = cv2.resize(perturbed_bg_img, (save_img_shape[1], save_img_shape[0]), cv2.INTER_AREA)


        # --- Initialization ---
        mesh_shape = self.origin_img.shape[:2]
        self.synthesis_perturbed_img = np.full((enlarge_img_shrink[0], enlarge_img_shrink[1], 3), 257, dtype=np.int16)
        self.new_shape = self.synthesis_perturbed_img.shape[:2]


        origin_pixel_position = np.argwhere(np.zeros(mesh_shape, dtype=np.uint32) == 0).reshape(mesh_shape[0], mesh_shape[1], 2)
        pixel_position = np.argwhere(np.zeros(self.new_shape, dtype=np.uint32) == 0).reshape(self.new_shape[0], self.new_shape[1], 2)
        self.perturbed_xy_ = pixel_position.copy()

        self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))
        x_min, y_min, x_max, y_max = self.adjust_position_v2(0, 0, mesh_shape[0], mesh_shape[1], save_img_shape)
        origin_pixel_position += [x_min, y_min]


        x_min, y_min, x_max, y_max = self.adjust_position(0, 0, mesh_shape[0], mesh_shape[1])
        self.synthesis_perturbed_img[x_min:x_max, y_min:y_max] = self.origin_img
        self.synthesis_perturbed_label[x_min:x_max, y_min:y_max] = origin_pixel_position


        synthesis_perturbed_img_map = self.synthesis_perturbed_img.copy()
        synthesis_perturbed_label_map = self.synthesis_perturbed_label.copy()

        # --- Perturbation Parameters (Preserving Original Logic) ---
        alpha_perturbed = random.randint(8, 14) / 10
        self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = -1, -1, self.new_shape[0], self.new_shape[1]

        perturbed_time = 0
        fail_perturbed_time = 0
        is_normalizationFun_mixture = self.is_perform(0.1, 0.9)
        normalizationFun_0_1 = False

        if fold_curve == 'fold':
            fold_curve_random = True
            normalizationFun_0_1 = self.is_perform(0.2, 0.8)
            if is_normalizationFun_mixture:
                if self.is_perform(0.5, 0.5):
                    alpha_perturbed = random.randint(100, 140) / 100
                else:
                    alpha_perturbed = random.randint(80, 120) / 100
            else:
                if normalizationFun_0_1:
                    alpha_perturbed = random.randint(40, 50) / 100
                else:
                    alpha_perturbed = random.randint(70, 120) / 100
        else:  # curve
            fold_curve_random = self.is_perform(0.1, 0.9)
            alpha_perturbed = random.randint(80, 160) / 100
            is_normalizationFun_mixture = False

        # --- Main Perturbation Loop (Preserving Original Logic) ---
        for repeat_i in range(repeat_time):

            synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257, dtype=np.int16)
            synthesis_perturbed_label = np.zeros_like(self.synthesis_perturbed_label)

             # Perturbed point selection (Preserving Original)
            perturbed_p = np.array([
                random.randint((self.new_shape[0]-im_lr)//2*10, (self.new_shape[0]-(self.new_shape[0]-im_lr)//2) * 10) / 10,
                random.randint((self.new_shape[1]-im_ud)//2*10, (self.new_shape[1]-(self.new_shape[1]-im_ud)//2) * 10) / 10
            ])
            perturbed_pp = np.array([
                random.randint((self.new_shape[0]-im_lr)//2*10, (self.new_shape[0]-(self.new_shape[0]-im_lr)//2) * 10) / 10,
                random.randint((self.new_shape[1]-im_ud)//2*10, (self.new_shape[1]-(self.new_shape[1]-im_ud)//2) * 10) / 10
            ])

            perturbed_vp = perturbed_pp - perturbed_p
            perturbed_vp_norm = np.linalg.norm(perturbed_vp)

            perturbed_distance_vertex_and_line = np.dot((perturbed_p - pixel_position), perturbed_vp) / perturbed_vp_norm

            if fold_curve == 'fold' and self.is_perform(0.3, 0.7):
                perturbed_v = np.array([random.randint(-11000, 11000) / 100, random.randint(-11000, 11000) / 100])
            else:
                perturbed_v = np.array([random.randint(-9000, 9000) / 100, random.randint(-9000, 9000) / 100])

            if fold_curve == 'fold':
                if is_normalizationFun_mixture:
                    if self.is_perform(0.5, 0.5):
                        perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))
                    else:
                        perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), random.randint(1, 2))
                else:
                    if normalizationFun_0_1:
                        perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), 2)
                    else:
                        perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))

            else:  # curve
                if is_normalizationFun_mixture:
                    if self.is_perform(0.5, 0.5):
                        perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))
                    else:
                        perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), random.randint(1, 2))
                else:
                    if normalizationFun_0_1:
                        perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), 2)
                    else:
                        perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))

            if fold_curve_random:
                omega_perturbed = alpha_perturbed / (perturbed_d + alpha_perturbed)
            else:
                omega_perturbed = 1 - perturbed_d ** alpha_perturbed

            # Shadow effect (Preserving Original)
            if self.is_perform(0.6, 0.4):
                shadow_intensity = abs(np.linalg.norm(perturbed_v // 2)) * np.array([0.4 - random.random() * 0.1, 0.4 - random.random() * 0.1, 0.4 - random.random() * 0.1])
                shadow_mask = np.int16(np.round(omega_perturbed[x_min:x_max, y_min:y_max, np.newaxis] * shadow_intensity))
                synthesis_perturbed_img_map[x_min:x_max, y_min:y_max] = np.clip(synthesis_perturbed_img_map[x_min:x_max, y_min:y_max] - shadow_mask, 0, 255)


            if relativeShift_position in ['position', 'relativeShift_v2']:
                perturbed_xy_ = self.perturbed_xy_ + np.array([omega_perturbed * perturbed_v[0], omega_perturbed * perturbed_v[1]]).transpose(1, 2, 0)
                perturbed_xy_ = cv2.blur(perturbed_xy_, (17, 17))
                perturbed_xy_round_int = np.round(perturbed_xy_).astype(np.int32)

                # Optimized pixel mapping using NumPy indexing and try-except (Preserving the core logic)
                rows, cols = perturbed_xy_round_int.shape[:2]
                row_indices, col_indices = np.indices((rows, cols))

                try:
                    valid_mask = (perturbed_xy_round_int[:, :, 0] >= 0) & (perturbed_xy_round_int[:, :, 0] < synthesis_perturbed_img_map.shape[0]) & \
                                 (perturbed_xy_round_int[:, :, 1] >= 0) & (perturbed_xy_round_int[:, :, 1] < synthesis_perturbed_img_map.shape[1])

                    synthesis_perturbed_img[row_indices[valid_mask], col_indices[valid_mask]] = synthesis_perturbed_img_map[perturbed_xy_round_int[valid_mask, 0], perturbed_xy_round_int[valid_mask, 1]]
                    synthesis_perturbed_label[row_indices[valid_mask], col_indices[valid_mask]] = synthesis_perturbed_label_map[perturbed_xy_round_int[valid_mask, 0], perturbed_xy_round_int[valid_mask, 1]]

                except Exception as e: #Catching generic exception here is deliberate, the point is avoid crash
                  continue


            else:
                print('relativeShift_position error')
                exit()


# --- Validation and Clipping (Preserving Original Logic)---
            is_save_perturbed = False
            # These flags are not strictly needed anymore, but keep for clarity
            is_save_perturbed_1, is_save_perturbed_2, is_save_perturbed_3, is_save_perturbed_4 = False, False, False, False

            perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max = -1, -1, self.new_shape[0], self.new_shape[1]

            # Find bounding box of perturbed region (optimized)
            non_background_rows = np.where(np.sum(synthesis_perturbed_img, axis=(1, 2)) != 771 * self.new_shape[1])[0]
            non_background_cols = np.where(np.sum(synthesis_perturbed_img, axis=(0, 2)) != 771 * self.new_shape[0])[0]

            if non_background_rows.size > 0 and non_background_cols.size > 0:
                perturbed_x_min = max(0, non_background_rows[0]-1)
                perturbed_x_max = min(self.new_shape[0], non_background_rows[-1] + 1)
                perturbed_y_min = max(0, non_background_cols[0]-1)
                perturbed_y_max = min(self.new_shape[1], non_background_cols[-1] + 1)
                is_save_perturbed = True
            else:
                is_save_perturbed = False  # Explicitly set to False
                # We do NOT use 'continue' here.  We set is_save_perturbed
                # to False and let the later 'if is_save_perturbed:'
                # block handle skipping the rest of the processing.

            # We move the rest of the validation checks *inside* the
            # 'if is_save_perturbed:' block.  This ensures they are
            # only executed if we found a valid bounding box.
            if is_save_perturbed:
                if perturbed_y_min <= 0 or perturbed_y_max >= self.new_shape[1]-1 or perturbed_x_min <= 0 or perturbed_x_max >= self.new_shape[0]-1:
                    is_save_perturbed = False
                    # Again, no 'continue'.  We rely on the outer 'if'
                if perturbed_y_max - perturbed_y_min <= 1 or perturbed_x_max - perturbed_x_min <= 1:
                    is_save_perturbed = False
                    fail_perturbed_time += 1
                    # No 'continue' here either.

                mesh_0_b = int(round(im_lr*0.2))
                mesh_1_b = int(round(im_ud*0.2))
                mesh_0_s = int(round(im_lr*0.1))
                mesh_1_s = int(round(im_ud*0.1))

                if ((perturbed_x_max-perturbed_x_min) < (mesh_shape_[0]-mesh_0_s) or (perturbed_y_max-perturbed_y_min) < (mesh_shape_[1]-mesh_1_s) or (perturbed_x_max-perturbed_x_min) > (mesh_shape_[0]+mesh_0_b) or (perturbed_y_max-perturbed_y_min) > (mesh_shape_[1]+mesh_1_b)):
                    is_save_perturbed = False
                    # And no 'continue' here.

            if is_save_perturbed: # This if was already here
                # Reset and copy valid perturbed region
                self.synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257, dtype=np.int16)
                self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))

                synthesis_perturbed_img_repeat = synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :].copy()
                synthesis_perturbed_label_repeat = synthesis_perturbed_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :].copy()
                self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))

                if perturbed_x_max-perturbed_x_min > save_img_shape[0] or perturbed_y_max-perturbed_y_min > save_img_shape[1]:
                    synthesis_perturbed_img_repeat = cv2.resize(synthesis_perturbed_img_repeat, (im_ud, im_lr), interpolation=cv2.INTER_NEAREST)
                    synthesis_perturbed_label_repeat = cv2.resize(synthesis_perturbed_label_repeat, (im_ud, im_lr), interpolation=cv2.INTER_NEAREST)
                    self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(0, 0, im_lr, im_ud)

                else:
                    self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max)


                self.synthesis_perturbed_img[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max] = synthesis_perturbed_img_repeat
                self.synthesis_perturbed_label[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max] = synthesis_perturbed_label_repeat
                self.perturbed_xy_ = perturbed_xy_.copy()
                perturbed_time += 1

        if fail_perturbed_time == repeat_time:
            raise Exception('clip error')


      
# --- Perspective Transformation (Preserving Original Logic) ---
        perspective_shreshold = 280
        x_min_per, y_min_per, x_max_per, y_max_per = self.adjust_position(perspective_shreshold, perspective_shreshold, self.new_shape[0]-perspective_shreshold, self.new_shape[1]-perspective_shreshold)
        pts1 = np.float32([[x_min_per, y_min_per], [x_max_per, y_min_per], [x_min_per, y_max_per], [x_max_per, y_max_per]])
        e_1_ = x_max_per - x_min_per
        e_2_ = y_max_per - y_min_per
        e_3_ = e_2_
        e_4_ = e_1_
        perspective_shreshold_h = e_1_*0.02
        perspective_shreshold_w = e_2_*0.02

        # Simplified perspective point generation with checks
        def generate_pts2(curve_type):
            while True:
                if curve_type == 'curve' and self.is_perform(0.2, 0.8):
                    pts2 = np.float32([
                        [x_min_per + (random.random() - 1) * perspective_shreshold, y_min_per + (random.random() - 0.5) * perspective_shreshold],
                        [x_max_per + (random.random() - 1) * perspective_shreshold, y_min_per + (random.random() - 0.5) * perspective_shreshold],
                        [x_min_per + (random.random()) * perspective_shreshold, y_max_per + (random.random() - 0.5) * perspective_shreshold],
                        [x_max_per + (random.random()) * perspective_shreshold, y_max_per + (random.random() - 0.5) * perspective_shreshold]
                    ])
                else:
                    pts2 = np.float32([
                        [x_min_per + (random.random() - 0.5) * perspective_shreshold, y_min_per + (random.random() - 0.5) * perspective_shreshold],
                        [x_max_per + (random.random() - 0.5) * perspective_shreshold, y_min_per + (random.random() - 0.5) * perspective_shreshold],
                        [x_min_per + (random.random() - 0.5) * perspective_shreshold, y_max_per + (random.random() - 0.5) * perspective_shreshold],
                        [x_max_per + (random.random() - 0.5) * perspective_shreshold, y_max_per + (random.random() - 0.5) * perspective_shreshold]
                    ])

                e_1 = np.linalg.norm(pts2[0]-pts2[1])
                e_2 = np.linalg.norm(pts2[0]-pts2[2])
                e_3 = np.linalg.norm(pts2[1]-pts2[3])
                e_4 = np.linalg.norm(pts2[2]-pts2[3])
                if e_1_+perspective_shreshold_h > e_1 and e_2_+perspective_shreshold_w > e_2 and e_3_+perspective_shreshold_w > e_3 and e_4_+perspective_shreshold_h > e_4 and \
                    e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
                    abs(e_1-e_4) < perspective_shreshold_h and abs(e_2-e_3) < perspective_shreshold_w:
                    break
            return pts2

        pts2 = generate_pts2(fold_curve)

        M = cv2.getPerspectiveTransform(pts1, pts2)
        one = np.ones((self.new_shape[0], self.new_shape[1], 1), dtype=np.int16)
        matr = np.dstack((pixel_position, one))
        new = np.dot(M, matr.reshape(-1, 3).T).T.reshape(self.new_shape[0], self.new_shape[1], 3)
        x = new[:, :, 0]/new[:, :, 2]
        y = new[:, :, 1]/new[:, :, 2]
        perturbed_xy_round_int = np.dstack((x, y))
        perturbed_xy_round_int = np.around(cv2.blur(perturbed_xy_round_int, (17, 17))).astype(np.int32)
        perturbed_xy_round_int = (perturbed_xy_round_int - np.min(perturbed_xy_round_int.reshape(-1,2), axis = 0)).astype(np.int32)

        synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257, dtype=np.int16)
        synthesis_perturbed_label = np.zeros_like(self.synthesis_perturbed_label)

        # Optimized pixel mapping with correct error handling.
        rows, cols = perturbed_xy_round_int.shape[:2]
        row_indices, col_indices = np.indices((rows, cols))

        valid_mask = (perturbed_xy_round_int[:, :, 0] >= 0) & (perturbed_xy_round_int[:, :, 0] < self.synthesis_perturbed_img.shape[0]) & \
                     (perturbed_xy_round_int[:, :, 1] >= 0) & (perturbed_xy_round_int[:, :, 1] < self.synthesis_perturbed_img.shape[1])

        # Apply the transformation ONLY to valid pixels.
        try:
            synthesis_perturbed_img[row_indices[valid_mask], col_indices[valid_mask]] = self.synthesis_perturbed_img[perturbed_xy_round_int[valid_mask, 0], perturbed_xy_round_int[valid_mask, 1]]
            synthesis_perturbed_label[row_indices[valid_mask], col_indices[valid_mask]] = self.synthesis_perturbed_label[perturbed_xy_round_int[valid_mask, 0], perturbed_xy_round_int[valid_mask, 1]]
        except Exception as e:
            print(f"An unexpected error occurred during perspective pixel mapping: {e}")
            #  Crucially, we do NOT continue here.  We let the rest of the
            #  validation happen.  If the image is completely invalid
            #  due to the error, the later checks will catch it.
            #  If *some* pixels were mapped, we still proceed.
            pass # Use pass instead of continue

        # --- Validation for Perspective Transformation ---
        is_save_perspective_1, is_save_perspective_2, is_save_perspective_3, is_save_perspective_4 = False, False, False, False
        perspective_x_min, perspective_y_min, perspective_x_max, perspective_y_max = -1, -1, self.new_shape[0], self.new_shape[1]

        # Find bounding box of the perspective-transformed region (optimized)
        non_background_rows = np.where(np.sum(synthesis_perturbed_img, axis=(1, 2)) != 771 * self.new_shape[1])[0]
        non_background_cols = np.where(np.sum(synthesis_perturbed_img, axis=(0, 2)) != 771 * self.new_shape[0])[0]

        if non_background_rows.size > 0 and non_background_cols.size > 0:
            perspective_x_min = max(0, non_background_rows[0]-1)
            perspective_x_max = min(self.new_shape[0], non_background_rows[-1] + 1)
            perspective_y_min = max(0, non_background_cols[0]-1)
            perspective_y_max = min(self.new_shape[1], non_background_cols[-1] + 1)
            is_save_perspective = True
        else:
            is_save_perspective = False  # Explicitly set to False
            continue # Skip if no valid result

        if perspective_y_min <= 0 or perspective_y_max >= self.new_shape[1]-1 or perspective_x_min <= 0 or perspective_x_max >= self.new_shape[0]-1:
            is_save_perspective = False
            continue
        if perspective_y_max - perspective_y_min <= 1 or perspective_x_max - perspective_x_min <= 1:
            is_save_perspective = False
            continue
        if ((perspective_x_max-perspective_x_min) < (mesh_shape_[0]-mesh_0_s) or (perspective_y_max-perspective_y_min) < (mesh_shape_[1]-mesh_1_s) or (perspective_x_max-perspective_x_min) > (mesh_shape_[0]+mesh_0_b) or (perspective_y_max-perspective_y_min) > (mesh_shape_[1]+mesh_1_b)):
            is_save_perspective = False
            continue

        if is_save_perspective:
            # Reset and copy the valid perspective-transformed region
            self.synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257, dtype=np.int16)
            self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))

            synthesis_perturbed_img_repeat = synthesis_perturbed_img[perspective_x_min:perspective_x_max, perspective_y_min:perspective_y_max, :].copy()
            synthesis_perturbed_label_repeat = synthesis_perturbed_label[perspective_x_min:perspective_x_max, perspective_y_min:perspective_y_max, :].copy()

            if perspective_x_max - perspective_x_min > save_img_shape[0] or perspective_y_max - perspective_y_min > save_img_shape[1]:
                synthesis_perturbed_img_repeat = cv2.resize(synthesis_perturbed_img_repeat, (im_ud, im_lr), interpolation=cv2.INTER_NEAREST)
                synthesis_perturbed_label_repeat = cv2.resize(synthesis_perturbed_label_repeat, (im_ud, im_lr), interpolation=cv2.INTER_NEAREST)
                self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(0, 0, im_lr, im_ud)

            else:
                self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(perspective_x_min, perspective_y_min, perspective_x_max, perspective_y_max)

            self.synthesis_perturbed_img[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max] = synthesis_perturbed_img_repeat
            self.synthesis_perturbed_label[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max] = synthesis_perturbed_label_repeat

        # --- Final Clipping and Saving ---
        perfix_ = self.save_suffix + '_' + str(m) + '_' + str(n)

        if not is_save_perturbed and perturbed_time == 0:
            raise Exception('clip error')  # Consistent error handling
        else:
            is_save_perturbed = True

        if is_save_perturbed:
            self.new_shape = save_img_shape

            synthesis_perturbed_img = self.synthesis_perturbed_img[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max, :].copy()
            synthesis_perturbed_label = self.synthesis_perturbed_label[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max, :].copy()

            self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max)
            perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max = self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max

            # Randomly reduce margins (Preserving original logic)
            reduce_value_x = int(round(min((random.random()/2)*(self.new_shape[0]-(self.perturbed_x_max-self.perturbed_x_min)), min(reduce_value, reduce_value_v2))))
            reduce_value_y = int(round(min((random.random()/2)*(self.new_shape[1]-(self.perturbed_y_max-self.perturbed_y_min)), min(reduce_value, reduce_value_v2))))
            perturbed_x_min = max(perturbed_x_min-reduce_value_x, 0)
            perturbed_x_max = min(perturbed_x_max+reduce_value_x, self.new_shape[0])
            perturbed_y_min = max(perturbed_y_min-reduce_value_y, 0)
            perturbed_y_max = min(perturbed_y_max+reduce_value_y, self.new_shape[1])



            self.synthesis_perturbed_img = np.full((self.new_shape[0], self.new_shape[1], 3), 257, dtype=np.int16)
            self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))
            self.synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :] = synthesis_perturbed_img
            self.synthesis_perturbed_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :] = synthesis_perturbed_label
            pixel_position = np.argwhere(np.zeros(self.new_shape, dtype=np.uint32) == 0).reshape(self.new_shape[0], self.new_shape[1], 2)

            if relativeShift_position == 'relativeShift_v2':
                self.synthesis_perturbed_label -= pixel_position # Final label adjustment


            # --- Final Checks and Saving ---
            if np.sum(self.synthesis_perturbed_img[:, 0]) != 771 * self.new_shape[0] or np.sum(self.synthesis_perturbed_img[:, self.new_shape[1]-1]) != 771 * self.new_shape[0] or \
                    np.sum(self.synthesis_perturbed_img[0, :]) != 771 * self.new_shape[1] or np.sum(self.synthesis_perturbed_img[self.new_shape[0]-1, :]) != 771*self.new_shape[1]:
                is_save_perturbed = False


            if is_save_perturbed:
                label = np.zeros_like(self.synthesis_perturbed_img)
                foreORbackground_label = np.ones(self.new_shape, dtype=np.int16)

                self.synthesis_perturbed_label[np.sum(self.synthesis_perturbed_img, 2) == 771] = 0
                foreORbackground_label[np.sum(self.synthesis_perturbed_img, 2) == 771] = 0
                label[:, :, :2] = self.synthesis_perturbed_label
                label[:, :, 2] = foreORbackground_label

                # --- HSV Augmentation (Preserving Original Logic) ---
                if self.is_perform(0.1, 0.9):
                    if self.is_perform(0.2, 0.8):
                        synthesis_perturbed_img_clip_HSV = self.synthesis_perturbed_img.copy().astype(np.float32)
                        synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_RGB2HSV)
                        H_, S_, V_ = (random.random()-0.2)*20, (random.random()-0.2)/8, (random.random()-0.2)*20
                        synthesis_perturbed_img_clip_HSV[:, :, 0], synthesis_perturbed_img_clip_HSV[:, :, 1], synthesis_perturbed_img_clip_HSV[:, :, 2] = synthesis_perturbed_img_clip_HSV[:, :, 0]-H_, synthesis_perturbed_img_clip_HSV[:, :, 1]-S_, synthesis_perturbed_img_clip_HSV[:, :, 2]-V_
                        synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_HSV2RGB).astype(np.int16)
                        synthesis_perturbed_img_clip_HSV[np.sum(self.synthesis_perturbed_img, 2) == 771] = perturbed_bg_img[np.sum(self.synthesis_perturbed_img, 2) == 771]
                        self.synthesis_perturbed_img = synthesis_perturbed_img_clip_HSV
                    else:
                        perturbed_bg_img_HSV = perturbed_bg_img.astype(np.float32)
                        perturbed_bg_img_HSV = cv2.cvtColor(perturbed_bg_img_HSV, cv2.COLOR_RGB2HSV)
                        H_, S_, V_ = (random.random()-0.5)*20, (random.random()-0.5)/8, (random.random()-0.2)*20
                        perturbed_bg_img_HSV[:, :, 0], perturbed_bg_img_HSV[:, :, 1], perturbed_bg_img_HSV[:, :, 2] = perturbed_bg_img_HSV[:, :, 0]-H_, perturbed_bg_img_HSV[:, :, 1]-S_, perturbed_bg_img_HSV[:, :, 2]-V_
                        perturbed_bg_img_HSV = cv2.cvtColor(perturbed_bg_img_HSV, cv2.COLOR_HSV2RGB).astype(np.int16)
                        self.synthesis_perturbed_img[np.sum(self.synthesis_perturbed_img, 2) == 771] = perturbed_bg_img_HSV[np.sum(self.synthesis_perturbed_img, 2) == 771]

                else:
                    synthesis_perturbed_img_clip_HSV = self.synthesis_perturbed_img.copy().astype(np.float32)
                    synthesis_perturbed_img_clip_HSV[np.sum(self.synthesis_perturbed_img, 2) == 771] = perturbed_bg_img[np.sum(self.synthesis_perturbed_img, 2) == 771]
                    synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_RGB2HSV)
                    H_, S_, V_ = (random.random()-0.5)*20, (random.random()-0.5)/10, (random.random()-0.4)*20
                    synthesis_perturbed_img_clip_HSV[:, :, 0], synthesis_perturbed_img_clip_HSV[:, :, 1], synthesis_perturbed_img_clip_HSV[:, :, 2] = synthesis_perturbed_img_clip_HSV[:, :, 0]-H_, synthesis_perturbed_img_clip_HSV[:, :, 1]-S_, synthesis_perturbed_img_clip_HSV[:, :, 2]-V_
                    synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_HSV2RGB).astype(np.int16)
                    self.synthesis_perturbed_img = synthesis_perturbed_img_clip_HSV

                synthesis_perturbed_img = np.zeros_like(self.synthesis_perturbed_img, dtype=np.int16)
                if im_lr >= im_ud:
                    synthesis_perturbed_img[:, perturbed_y_min:perturbed_y_max, :] = self.synthesis_perturbed_img[:, perturbed_y_min:perturbed_y_max, :]
                else:
                    synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, :, :] = self.synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, :, :]
                self.synthesis_perturbed_img = synthesis_perturbed_img

                self.synthesis_perturbed_img[self.synthesis_perturbed_img > 255] = 255
                self.synthesis_perturbed_img[self.synthesis_perturbed_img < 0] = 0


                cv2.imwrite(self.save_path + 'png/' + perfix_ + '_' + fold_curve + '.png', self.synthesis_perturbed_img)
                synthesis_perturbed_color = np.concatenate((self.synthesis_perturbed_img, label), axis=2)
                with open(self.save_path+'color/'+perfix_+'_'+fold_curve+'.gw', 'wb') as f:
                    pickle_perturbed_data = pickle.dumps(synthesis_perturbed_color)
                    f.write(pickle_perturbed_data)


        if not is_save_perturbed:
            print('save error') #Consistent error
        else:
            cv2.imwrite(self.save_path + 'scan/' + self.save_suffix + '_' + str(m) + '.png', self.origin_img)
            end_time = time.time()
            elapsed_time = end_time - begin_time
            mm, ss = divmod(elapsed_time, 60)
            hh, mm = divmod(mm, 60)
            print(f"{m}_{n}_{fold_curve} Time : {hh:02.0f}:{mm:02.0f}:{ss:02.0f}")


def xgw(args):
    """
    Main function to process images and generate perturbations.
    """
    path = args.path
    bg_path = args.bg_path
    if not os.path.exists(path):
        raise Exception('-- No path')
    if not os.path.exists(bg_path):
        raise Exception('-- No bg_path')

    if args.output_path is None:
        save_path = '/lustre/home/gwxie/data/unwarp_new/train/data1024_greyV2/'  # Default path
    else:
        save_path = args.output_path

    # Create necessary output directories
    for dir_name in ['color', 'png', 'scan', 'outputs']:
        dir_path = os.path.join(save_path, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    save_suffix = str.split(args.path, '/')[-2]

    all_img_path = getDatasets(path)
    all_bgImg_path = getDatasets(bg_path)
    global begin_train
    begin_train = time.time()


    process_pool = Pool(2)  # Using 2 processes as specified
    for m, img_path in enumerate(all_img_path):
        for n in range(args.sys_num):
            img_path_ = os.path.join(path, img_path)
            bg_path_ = os.path.join(bg_path, random.choice(all_bgImg_path)) + '/'

            for _ in range(10): # Retry loop
                try:
                    saveFold = perturbed(img_path_, bg_path_, save_path, save_suffix)
                    saveCurve = perturbed(img_path_, bg_path_, save_path, save_suffix)

                    repeat_time_fold = min(max(round(np.random.normal(8, 4)), 1), 12)
                    repeat_time_curve = min(max(round(np.random.normal(6, 4)), 1), 10)

                    process_pool.apply_async(func=saveFold.save_img, args=(m, n, 'fold', repeat_time_fold, 'relativeShift_v2'))
                    process_pool.apply_async(func=saveCurve.save_img, args=(m, n, 'curve', repeat_time_curve, 'relativeShift_v2'))

                except BaseException as err: #Catch BaseException to avoid un-managed crash
                    print(err)
                    continue
                break


    process_pool.close()
    process_pool.join()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--path', default='validate', type=str,
                        help='the path of origin img.')
    parser.add_argument('--bg_path', default='validate', type=str,
                        help='the path of bg img.')
    parser.add_argument('--output_path', default=None, type=str,
                        help='the path of output img.')
    parser.add_argument('--count_from', '-p', default=0, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--repeat_T', default=0, type=int)
    parser.add_argument('--sys_num', default=7, type=int)

    args = parser.parse_args()
    xgw(args)
