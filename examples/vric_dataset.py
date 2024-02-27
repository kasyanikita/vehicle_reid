from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import torchreid
from torchreid.data import ImageDataset


class VRICDataset(ImageDataset):
    dataset_dir = 'data/VRIC'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).

        # Train
        i = 0
        train_annot_path = osp.join(self.dataset_dir, "vric_train.txt")
        train_images_dir = osp.join(self.dataset_dir, "train_images")
        train = []
        map_train_id = {}
        with open(train_annot_path) as f:
            for line in f.readlines():
                filename, id, cam_id = line.split(" ")

                filepath = osp.join(train_images_dir, filename)
                if id not in map_train_id:
                    map_train_id[id] = i
                    i += 1
                id = map_train_id[id]
                
                train.append((filepath, id, cam_id))

        i = 0
        query_annot_path = osp.join(self.dataset_dir, "vric_probe.txt")
        query_images_dir = osp.join(self.dataset_dir, "probe_images")
        query = []
        map_test_id = {}
        with open(query_annot_path) as f:
            for line in f.readlines():
                filename, id, cam_id = line.split(" ")

                filepath = osp.join(query_images_dir, filename)
                if id not in map_test_id:
                    map_test_id[id] = i
                    i += 1
                id = map_test_id[id]
                
                query.append((filepath, id, cam_id))

        gallery_annot_path = osp.join(self.dataset_dir, "vric_gallery.txt")
        gallery_images_dir = osp.join(self.dataset_dir, "gallery_images")
        gallery = []
        with open(gallery_annot_path) as f:
            for line in f.readlines():
                filename, id, cam_id = line.split(" ")

                filepath = osp.join(gallery_images_dir, filename)
                id = map_test_id[id]
                
                gallery.append((filepath, id, cam_id))

        super(VRICDataset, self).__init__(train, query, gallery, **kwargs)


torchreid.data.register_image_dataset('vric', VRICDataset)