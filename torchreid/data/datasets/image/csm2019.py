from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import copy
import json

from torchreid.data.datasets import ImageDataset

class CastSearchMovie2019(ImageDataset):
    dataset_dir = 'csm2019'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)

        self.query_dir = osp.join(self.dataset_dir, 'dummy', 'dummy')
        self.gallery_dir = osp.join(self.dataset_dir, 'dummy', 'dummy')

        required_files = [
            self.dataset_dir,
            self.query_dir,
            self.gallery_dir
        ]
        #self.check_before_run(required_files)
        real_data_dir = '/media/chundi/Data/WIDER_FACE/'

        with open(real_data_dir + 'train.json') as f:
            train_data = json.load(f)

        key_list = sorted(list(train_data.keys()))

        train_label_dict = {}

        for k in key_list:
            candidates = train_data[k]['candidates']
            for candidate in candidates:
                if candidate['label'] == 'others':
                    train_label_dict[candidate['id']] = 'others'
                else:
                    train_label_dict[candidate['id']] = candidate['img'].split('/')[0] + '_' + candidate['label']

        labels_set_list = sorted(list(set(train_label_dict.values())))
        labels2lId = {}

        for i, label in enumerate(labels_set_list):
            labels2lId[label] = i

        with open(real_data_dir + 'train_paths.txt') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        train = [(img_path, labels2lId[train_label_dict[img_path.split('/')[-1].split('.')[0]]], -1) for img_path in lines]
        with open(real_data_dir + 'val_cast_paths.txt') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        query = [(img_path, -1, -1) for img_path in lines]

        with open(real_data_dir + 'val_paths.txt') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        gallery = [(img_path, -1, -1) for img_path in lines]


        super(CastSearchMovie2019, self).__init__(train, query, gallery, **kwargs)