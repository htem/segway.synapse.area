import copy
import logging
import os
import time
from itertools import product
from os import path
from collections import defaultdict
import math

import daisy
import numpy as np
import trimesh
from daisy import Coordinate
from skimage import measure
from scipy import ndimage

from plantcv import plantcv as pcv

from .downscale import downscale_block

logger = logging.getLogger(__name__)

class SynapseArea():

    def __init__(self,
            synapse_labels,
            regions,
            voxel_size,
            affs_ndarray=None,
            affs_threshold=50,
            true_voxel_size=None,
            super_resolution_factor=(1, 2, 2),
            local_realignment=False,
            local_realigner=None,
            local_alignment_offsets_xy=None,
            binary_erosion=0,
            ):
        pass

        voxel_size = daisy.Coordinate(tuple(voxel_size))

        self.synapse_labels = synapse_labels
        self.affs_threshold = affs_threshold
        self.super_resolution_factor = super_resolution_factor
        self.binary_erosion = binary_erosion

        self.regions = dict()
        self.cropped_mask = dict()
        self.area_props = defaultdict(dict)
        self.skeletons = dict()
        self.skeleton_segments = dict()
        self.synapse_drifts = dict()
        self.pruned_skeletons = dict()
        self.longest_skeleton_segments_len = dict()
        self.xy_section_offsets = dict()
        self.synapse_centroids = dict()

        self.prune_spurious_branches = True
        self.local_realignment = local_realignment
        # self.local_realignment = False
        self.local_realigner = local_realigner
        self.local_alignment_offsets_xy = local_alignment_offsets_xy
        if local_realignment:
            assert local_realigner is not None or local_alignment_offsets_xy is not None

        for reg in regions:
            self.regions[reg['label']] = reg

        if true_voxel_size is None:
            true_voxel_size = voxel_size
        if super_resolution_factor is not None:
            true_voxel_size = [int(k/v) for k, v in zip(true_voxel_size, super_resolution_factor)]
            voxel_size = [int(k/v) for k, v in zip(voxel_size, super_resolution_factor)]
            voxel_size = daisy.Coordinate(tuple(voxel_size))
        self.true_voxel_size = true_voxel_size
        self.voxel_size = voxel_size
        assert self.true_voxel_size[1] == self.true_voxel_size[2]
        self.pixel_len = (self.true_voxel_size[1]+self.true_voxel_size[2])/2
        self.pixel_depth = self.true_voxel_size[0]

        pcv.params.line_thickness = 2

        self.affs = affs_ndarray
        dims = len(synapse_labels.shape)
        assert affs_ndarray.shape[-dims:] == synapse_labels.shape[-dims:]


    def line_area(self, line_len, drift=None):
        depth = self.pixel_depth
        if drift is not None:
            depth = math.sqrt(depth*depth+drift*drift)
        return line_len * depth * self.pixel_len

    def diameter_equiv(self, area):
        return math.sqrt(area/math.pi)*2

    def get_area(self, label, methods,
                ):
        self.area_props[label] = {}
        if 'pixel' in methods:
            self.get_pixel_area(label)
        if 'pixel_with_drift' in methods:
            self.get_pixel_area(label, with_drift=True)
        if 'skeleton' in methods:
            self.get_skeleton_area(label)
        if 'skeleton_with_drift' in methods:
            self.get_skeleton_area(label, with_drift=True)
        if 'mesh' in methods:
            self.get_mesh_area(label)
        if 'ellipsoid' in methods:
            self.get_ellipsoid_area(label)
        if 'ellipsoid_with_drift' in methods:
            self.get_ellipsoid_area(label, with_drift=True)
        return self.area_props[label]

    def get_pixel_area(self, label, with_drift=False):
        if self.prune_spurious_branches:
            self.get_longest_skeleton_segments_len(label)
        skeleton = self.get_skeleton(label)
        if with_drift:
            synapse_drifts = self.get_synapse_drifts(label)

        area_sum = 0
        area_sum_drift = 0
        count_sum = 0
        for i, s in enumerate(skeleton):
            # count # of pixels in skeleton
            count = s.sum()
            count = int(count/255)  # skeleton mask is either 0 or 255
            count_sum += count
            if with_drift:
                area_sum_drift += self.line_area(count, synapse_drifts[i])
            area_sum += self.line_area(count)

        area_sum /= 1000000
        area_sum_drift /= 1000000
        self.area_props[label]['pixel_count'] = count_sum
        self.area_props[label]['pixel_area'] = area_sum
        self.area_props[label]['pixel_diameter'] = self.diameter_equiv(area_sum)
        if with_drift:
            self.area_props[label]['pixel_area_drift'] = area_sum_drift

    def get_skeleton_area(self, label, with_drift=False):
        skeleton_segments_lens = self.get_longest_skeleton_segments_len(label)
        if with_drift:
            synapse_drifts = self.get_synapse_drifts(label)

        area_sum = 0
        area_sum_drift = 0
        longest_diameter = 0
        for i, path_length in enumerate(skeleton_segments_lens):
            longest_diameter = max(longest_diameter, path_length)
            area = self.line_area(path_length)
            area_sum += area
            if with_drift:
                area_sum_drift += self.line_area(path_length, synapse_drifts[i])

        area_sum /= 1000000
        area_sum_drift /= 1000000
        self.area_props[label]['skeleton_area'] = area_sum
        self.area_props[label]['skeleton_diameter'] = self.diameter_equiv(area_sum)
        self.area_props[label]['skeleton_longest_diameter'] = int(longest_diameter * self.true_voxel_size[1])
        if with_drift:
            self.area_props[label]['skeleton_area_drift'] = area_sum_drift

    def get_ellipsoid_area(self, label, with_drift=False):
        skeleton_segments_lens = self.get_longest_skeleton_segments_len(label)
        if len(skeleton_segments_lens) == 0:
            self.area_props[label]['ellipsoid_area'] = 0
            self.area_props[label]['ellipsoid_diameter'] = 0
            if with_drift:
                self.area_props[label]['ellipsoid_area_drift'] = 0
                self.area_props[label]['ellipsoid_diameter_drift'] = 0
            return

        diameter = max(skeleton_segments_lens) * self.true_voxel_size[1]
        depth = len(skeleton_segments_lens) * self.true_voxel_size[0]

        area = (diameter/2)*(depth/2)*math.pi / 1000000
        self.area_props[label]['ellipsoid_area'] = area
        self.area_props[label]['ellipsoid_diameter'] = (diameter+depth)/1000/2

        # debug = False
        if with_drift:
            centroids = self.get_synapse_centroids(label)
            if self.local_realignment:
                xy_corrections = self.get_xy_pix_realignment_offsets(label)
                for i in range(len(centroids)):
                    centroids[i][0] += xy_corrections[i][1]
                    centroids[i][1] += xy_corrections[i][0]

            top = centroids[0]
            bottom = centroids[-1]
            total_xy_drift = np.linalg.norm((top[0]-bottom[0], top[1]-bottom[1])) * self.true_voxel_size[1]
            depth = np.linalg.norm((depth, total_xy_drift))

            area = (diameter/2)*(depth/2)*math.pi / 1000000
            self.area_props[label]['ellipsoid_area_drift'] = area
            self.area_props[label]['ellipsoid_diameter_drift'] = (diameter+depth)/1000/2

    def pad_vertices(self, verts, offset=None):
        '''Replicate top and bottom sections by half Z step to compensate for mesh areas'''
        assert len(verts)
        if offset is not None:
            verts +=  offset # get z,y,x coordinates of border pixels
        first_border = verts[verts[:,0] == verts[:,0].min(), :]
        first_border -= (int(self.true_voxel_size[0] / 2), 0, 0)
        last_border = verts[verts[:,0] == verts[:,0].max(), :]
        last_border += (int(self.true_voxel_size[0] / 2), 0, 0)
        verts = np.append(verts, first_border, axis=0)
        verts = np.append(verts, last_border, axis=0)
        return verts

    def skeleton_to_verts(self, skeleton, xy_corrections_pix=None):

        verts = []
        for i, s in enumerate(skeleton):
            z = i*self.true_voxel_size[0]
            tmp = []
            for p in np.argwhere(s):
                y = float(p[0])
                x = float(p[1])
                if xy_corrections_pix:
                    y += xy_corrections_pix[i][1]
                    x += xy_corrections_pix[i][0]
                y *= self.true_voxel_size[1]
                x *= self.true_voxel_size[2]
                tmp.append((z, y, x))
            verts.extend(tmp)
        verts = np.array(verts)
        if len(verts) == 0:
            print(skeleton)
            raise RuntimeError("Skeleton is empty")
        return verts

    def get_mesh_area(self, label):
        xy_corrections = None
        if self.prune_spurious_branches:
            self.get_longest_skeleton_segments_len(label)
        skeleton = self.get_skeleton(label)

        if len(skeleton) == 0:
            return 0  # empty prediction

        if self.local_realignment:
            xy_corrections = self.get_xy_pix_realignment_offsets(label)

        verts = self.skeleton_to_verts(skeleton, xy_corrections)
        verts = self.pad_vertices(verts)
        cloud = trimesh.PointCloud(verts)
        area = 0
        try:
            # can fail with very small geometries
            mesh = cloud.convex_hull
            mesh.process(validate=True)
            area = np.sum(mesh.area)
        except Exception as e:
            pass
        # mesh area double counts the surface area to needed to div by 2
        area /= 2
        area /= 1000000
        self.area_props[label]['mesh_area'] = area
        self.area_props[label]['mesh_diameter'] = self.diameter_equiv(area)

    def get_cropped_mask(self, label):
        if label in self.cropped_mask:
            return self.cropped_mask[label]
        reg = self.regions[label]
        z1, y1, x1, z2, y2, x2 = reg['bbox']
        crop = self.synapse_labels[z1:z2, y1:y2, x1:x2]
        reg_mask = (crop == label).astype(np.uint8)

        if self.binary_erosion:
            for i in range(len(reg_mask)):
                if np.sum(reg_mask[i]) <= 10:
                    continue  # don't shrink very small predictions
                shrunk_mask = ndimage.binary_erosion(reg_mask[i], iterations=self.binary_erosion).astype(np.uint8)
                if np.sum(reg_mask[i]) == 0:
                    continue  # don't save gone objects
                reg_mask[i] = shrunk_mask

        if self.affs is not None:
            cropped_aff = self.affs[:, z1:z2, y1:y2, x1:x2]
            cropped_aff = cropped_aff[1:, :, :, :].mean(axis=0) <= self.affs_threshold
            reg_mask &= cropped_aff

        if self.super_resolution_factor is not None:
            for i, n in enumerate(self.super_resolution_factor):
                reg_mask = np.repeat(reg_mask, n, axis=i)

        self.cropped_mask[label] = reg_mask
        return reg_mask

    def get_skeleton(self, label):
        if label in self.skeletons:
            return self.skeletons[label]
        mask = self.get_cropped_mask(label)
        skeleton = []
        # lengths = []
        for plane in mask:
            assert plane.dtype == np.uint8
            if np.count_nonzero(plane) == 0:
                continue
            temp = pcv.morphology.skeletonize(mask=plane)
            skeleton.append(temp)

        self.skeletons[label] = np.array(skeleton, dtype=np.uint8)
        return self.skeletons[label]

    def get_longest_skeleton_segments_len(self, label):
        if label in self.longest_skeleton_segments_len:
            return self.longest_skeleton_segments_len[label]
        lengths = []
        skeleton = self.get_skeleton(label)
        skeleton_segments = self.get_skeleton_segments(label)
        for i, section_segments in enumerate(skeleton_segments):
            segmented_img, objs = section_segments
            labeled_img = pcv.morphology.segment_path_length(segmented_img=segmented_img, 
                                                     objects=objs, label="default")
            path_lengths = pcv.outputs.observations['default']['segment_path_length']['value']
            longest = max(path_lengths)
            longest_index = path_lengths.index(longest)
            if len(path_lengths) > 1:
                if self.prune_spurious_branches:
                    skel = skeleton[i]
                    skel[:, :] = 0
                    for i, obj in enumerate(objs):
                        if i == longest_index:
                            for xy in obj:
                                xy = xy[0]
                                skel[xy[1]][xy[0]] = 255
                            continue
            length = longest
            length += 1.5
            if len(path_lengths) > 1:
                length += 1.5
            lengths.append(length)
        self.longest_skeleton_segments_len[label] = lengths
        return lengths

    def get_skeleton_segments(self, label):
        if label in self.skeleton_segments:
            return self.skeleton_segments[label]
        skeleton = self.get_skeleton(label)
        self.skeleton_segments[label] = []
        for section in skeleton:
            segmented_img, obj = pcv.morphology.segment_skeleton(skel_img=section)
            self.skeleton_segments[label].append((segmented_img, obj))
        return self.skeleton_segments[label]

    def get_synapse_drifts(self, label):
        if label in self.synapse_drifts:
            return self.synapse_drifts[label]

        centroids = self.get_synapse_centroids(label)
        for i, c in enumerate(centroids):
            prev = max(0, i-1)
            next = min(len(centroids)-1, i+1)
            n_sections = (next-prev)
            if n_sections:
                drift = np.linalg.norm([a-b for a, b in zip(centroids[prev], centroids[next])])
                assert n_sections <= 2 and n_sections >= 1
                drift *= self.true_voxel_size[1] / n_sections
                drifts.append(drift)
            else:
                drifts.append(0)
        self.synapse_drifts[label] = drifts
        return drifts

    def get_synapse_centroids(self, label):
        if label in self.synapse_centroids:
            return copy.deepcopy(self.synapse_centroids[label])

        mask = self.get_skeleton(label)
        centroids = []
        for s in mask:
            props = measure.regionprops(s)
            centroids.append([k for k in props[0].centroid])
        self.synapse_centroids[label] = centroids
        return copy.deepcopy(centroids)

    def convert_pix_coords_to_roi(self, z1, y1, x1, z2, y2, x2):
        zyx1 = daisy.Coordinate((z1, y1, x1))
        zyx2 = daisy.Coordinate((z2, y2, x2))
        zyx1 *= self.voxel_size
        zyx2 *= self.voxel_size
        roi = daisy.Roi(zyx1, zyx2-zyx1)
        return roi

    def convert_nm_to_pix(self, xy_section_offsets_nm):
        new_list = []
        for xy in xy_section_offsets_nm:
            new_list.append((xy[0]/self.voxel_size[2],
                             xy[1]/self.voxel_size[1]))
        return new_list

    def convert_pix_to_nm(self, xy_section_offsets_pix):
        new_list = []
        for xy in xy_section_offsets_pix:
            new_list.append((xy[0]*self.true_voxel_size[2],
                             xy[1]*self.true_voxel_size[1]))
        return new_list

    def get_xy_pix_realignment_offsets(self, label):
        if label in self.xy_section_offsets:
            return self.xy_section_offsets[label]
        reg = self.regions[label]
        z1, y1, x1, z2, y2, x2 = reg['bbox']
        if z2 - z1 <= 1:
            self.xy_section_offsets[label] = [(0, 0)]
            return self.xy_section_offsets[label]

        if self.local_alignment_offsets_xy is not None:
            ret = []
            for i in range(z1, z2):
                # print(i)
                ret.append(self.local_alignment_offsets_xy[i])
        else:
            assert False
            assert self.local_realigner is not None
            ds_roi = self.convert_pix_coords_to_roi(z1, y1, x1, z2, y2, x2)
            ret = self.local_realigner.realign(roi=ds_roi)

        ret = self.convert_nm_to_pix(ret)
        self.xy_section_offsets[label] = ret
        return ret
