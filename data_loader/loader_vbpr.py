import torch
import numpy as np
import pandas as pd

from data_loader.loader_base import DataLoaderBase


class DataLoaderVBPR(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.item_pre_visual_feature = None
        self.load_pretrained_visual_data()
        self.print_info(logging)

    def load_pretrained_visual_data(self):
        pre_model = "item_visual_feat"
        pretrain_path = '%s/%s/%s.npy' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        self.item_pre_visual_feature = np.load(pretrain_path)

    def print_info(self, logging):
        logging.info('n_users:     %d' % self.n_users)
        logging.info('n_items:     %d' % self.n_items)
        logging.info('n_cf_train:  %d' % self.n_cf_train)
        logging.info('n_cf_test:   %d' % self.n_cf_test)


