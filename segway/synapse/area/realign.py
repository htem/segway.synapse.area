import cv2
import numpy as np
import os
import logging
import sys
import json
import datetime
from PIL import Image, ImageOps

import daisy
from daisy import Roi

logger = logging.getLogger(__name__)

def euclidean_len(a):
    return np.linalg.norm(a)

def to_ng_coord(zyx):
    return (int(zyx[2]/4),
            int(zyx[1]/4),
            int(zyx[0]/40))

class Realigner:
# modified by Jeff Rhoades (github.com/rhoadesScholar, Harvard 2021)
    def __init__(
            self,
            raw_file,
            raw_ds,
            xy_context_nm=0,
            xy_stride_nm=256,
            local_roi_offset=None,
            ):
        self.ds = daisy.open_ds(raw_file, raw_ds, 'r')
        self.local_roi_offset = local_roi_offset
        self.xy_context_nm = xy_context_nm
        self.xy_stride_nm = xy_stride_nm
        assert self.ds.voxel_size[1] == self.ds.voxel_size[2]
        self.xy_stride_pix = int(xy_stride_nm / self.ds.voxel_size[1] + .5)
        self.xy_stride_nm = self.xy_stride_pix * self.ds.voxel_size[1]

    def set_local_offset(self, local_roi_offset):
        self.local_roi_offset = local_roi_offset

    def add_padding_to_roi(self, roi, xy_padding_nm):
        roi = roi.grow((0, xy_padding_nm, xy_padding_nm), (0, xy_padding_nm, xy_padding_nm))
        return roi

    def calc_shift(#returns [nx2]
            self,
            stride_list,
            pattern_list,
            stride_r,
            stride_c,
            shift_limit=None,
            method=cv2.TM_CCOEFF,
            accumulate=True,
            ):
        """
        stride_list: [list of np.array] input image with stride (larger)
        pattern_list: [list of np.array] input image (smaller)
        stride_w: stride in of w in pixel.
        stride_h: stride in of h in pixel.

        return: [list of tuple] shift of each picture in pixel.
        """
        # checking img length
        stride_length = len(stride_list)
        pattern_length = len(pattern_list)
        if stride_length != pattern_length:
            raise ValueError(
                f'stride image length {stride_length} not equal to pattern image length {pattern_length}')

        # shift_x, shift_y: width and height
        offset = np.array([0, 0])
        xy_offset_list = [(0, 0)]
        stride_padding_offset = np.array([stride_r, stride_c])

        for i in range(len(stride_list) - 1):
            img_base = stride_list[i]
            img_pattern = pattern_list[i + 1]

            # pattern matching
            res = cv2.matchTemplate(img_base, img_pattern, method)
            _, _, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            top_left = top_left - stride_padding_offset

            if shift_limit is not None:
                shift_y = min(max(top_left[0], -shift_limit), shift_limit)
                shift_x = min(max(top_left[1], -shift_limit), shift_limit)
                top_left = np.array([shift_y, shift_x])

            if accumulate:
                offset += top_left
            else:
                offset = top_left
            xy_offset_list.append((offset[0], offset[1]))
            # calculate img
        return xy_offset_list

    def calc_shift_pair(#returns [nx2]
            self,
            img_ref,
            img_pattern,
            stride_r,
            stride_c,
            method=cv2.TM_CCOEFF,
            ):
        """
        stride_w: stride in of w in pixel.
        stride_h: stride in of h in pixel.

        return: [list of tuple] shift of each picture in pixel.
        """
        stride_padding_offset = np.array([stride_r, stride_c])
        res = cv2.matchTemplate(img_ref, img_pattern, method)
        _, _, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        offset = top_left - stride_padding_offset
        return offset

    def get_stride_roi(self, roi:daisy.Roi):

        if self.local_roi_offset:
            roi = Roi(roi.get_offset()+self.local_roi_offset, roi.get_shape())

        if not self.ds.roi.contains(roi):
            raise RuntimeError(f"Requested ROI {roi} not entirely within dataset {self.ds.roi}")

        roi = self.add_padding_to_roi(roi, self.xy_context_nm)
        roi = self.add_padding_to_roi(roi, self.xy_stride_nm)
        roi = self.ds.roi.intersect(roi)
        assert not roi.empty()

        pattern_roi = roi.copy()
        stride_roi = roi
        stride_roi = self.add_padding_to_roi(stride_roi, -self.xy_stride_nm)
        assert not stride_roi.empty()

        return stride_roi

    def get_data(self, stride_roi):

        stride_sections = self.ds.to_ndarray(roi=stride_roi)
        pattern_sections = stride_sections[:, self.xy_stride_pix:-self.xy_stride_pix, self.xy_stride_pix:-self.xy_stride_pix]
        return pattern_sections, stride_sections

    def convert_to_nm(self, xy_offset_list):
        new_list = []
        for xy in xy_offset_list:
            new_list.append((xy[0]*self.ds.voxel_size[2],
                             xy[1]*self.ds.voxel_size[1]))
        return new_list

    def realign(self,
                roi: daisy.Roi,
                ):

        stride_roi = self.get_stride_roi(roi)
        pattern_sections, stride_sections  = self.get_data(stride_roi)

        # logger.info('Realigning ...')
        xy_offset_list = self.calc_shift(
            stride_list=stride_sections,
            pattern_list=pattern_sections,
            stride_r=self.xy_stride_pix,
            stride_c=self.xy_stride_pix,
            )

        xy_offset_list = self.convert_to_nm(xy_offset_list)
        print(xy_offset_list)
        return xy_offset_list

    def get_previous_offset(self, corrected_offsets, i):
        if i == 0:
            return np.array((0, 0))
        else:
            return corrected_offsets[i-1]

    def multipass_realign(self, roi: daisy.Roi, factors: list):
        '''
        factors: list of -i previous sections to compare each section against for each i in factors
            true offset is the one with the least amount of offsets in the resulting list of offsets
        '''
        assert len(factors)

        stride_roi = self.get_stride_roi(roi)
        pattern_sections, stride_sections  = self.get_data(stride_roi)
        abs_z_offset = int(stride_roi.get_offset()[0]/self.ds.voxel_size[0])

        # calculate blank sections
        blanks = set()
        blanks_abs = []
        for i, pattern in enumerate(pattern_sections):
            if np.sum(pattern) == 0:
                blanks.add(i)
                blanks_abs.append(i+abs_z_offset)
        if len(blanks_abs):
            logger.warning(f'{blanks_abs} are blank!')
            logger.warning(f'At {to_ng_coord(stride_roi.get_offset())}!')

        corrected_offsets = {}

        for i, pattern in enumerate(pattern_sections):
            if i in blanks:
                corrected_offsets[i] = self.get_previous_offset(corrected_offsets, i)
                continue

            rel_offsets = []
            for j in [i-f for f in factors]:
                if j >= 0 and j not in blanks:
                    xy_offset = self.calc_shift_pair(
                        img_ref=stride_sections[j],
                        img_pattern=pattern,
                        stride_r=self.xy_stride_pix,
                        stride_c=self.xy_stride_pix,
                        )
                    rel_offsets.append((j, xy_offset))

            # get the smallest rel offset
            rel_offsets.sort(key=lambda x: euclidean_len(x[1]))
            if len(rel_offsets) == 0:
                corrected_offsets[i] = self.get_previous_offset(corrected_offsets, i)
                continue

            rel_offset = rel_offsets[0]
            corrected_offsets[i] = corrected_offsets[rel_offset[0]] + rel_offset[1]

        # postprocessing to detect big misaligned sections so to not make big changes
        hi_threshold = int(250/self.ds.voxel_size[1]+.5)  # in nm
        lo_threshold = int(200/self.ds.voxel_size[1]+.5)  # in nm
        section_n = 2
        for i in range(len(pattern_sections)):
            if i in blanks:
                continue
            prev_section = i-section_n
            while prev_section in blanks:
                prev_section -= 1
            next_section = i+section_n
            while next_section in blanks:
                next_section += 1
            if prev_section < 0 or next_section >= len(pattern_sections):
                continue
            # print(f'{i}: {prev_section}, {next_section}')
            diff0 = euclidean_len(corrected_offsets[i]-corrected_offsets[prev_section])
            diff1 = euclidean_len(corrected_offsets[i]-corrected_offsets[next_section])
            diff2 = euclidean_len(corrected_offsets[prev_section]-corrected_offsets[next_section])
            if diff0 > hi_threshold and diff1 > hi_threshold and diff2 < lo_threshold:
                logger.warning(f"Section {i+abs_z_offset} is detected to be misplaced ({to_ng_coord(stride_roi.get_offset())})")
                corrected_offsets[i] = (corrected_offsets[prev_section]+corrected_offsets[next_section])/2

        # recorrect blanks
        for i in range(len(pattern_sections)):
            if i in blanks:
                corrected_offsets[i] = self.get_previous_offset(corrected_offsets, i)

        ret = []
        for i in sorted(corrected_offsets.keys()):
            ret.append(corrected_offsets[i])
            # print(f'{i}: {corrected_offsets[i]}')

        ret = self.convert_to_nm(ret)

        return ret
