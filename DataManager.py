import cv2
import random
import numpy as np
import os.path as osp
import logging as logger
import imgaug.augmenters as iaa


logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class DataManager():
    def __init__(self):
        self.verify_index = 0
        self.seq = iaa.Sequential([
            iaa.MultiplySaturation((0.5, 1.4)),
            iaa.MultiplyBrightness((0.5, 1.4)),
        ])

    def set_enroll_data(self, data_root, enroll_file):
        self.enroll_id = []
        self.enroll_img_path = []
        train_file_buf = open(enroll_file)
        line = train_file_buf.readline().strip()

        while line:
            path, id = line.split(' ')
            fullpath = osp.join(data_root, path)
            if osp.isfile(fullpath):
                self.enroll_id.append(id)
                self.enroll_img_path.append(fullpath)
            else:
                logger.error(f'{fullpath} does not exist.')
            line = train_file_buf.readline().strip()
        
        logger.info(f'Set enroll data success, ID: {len(set(self.enroll_id))} Image: {len(self.enroll_img_path)}')

    def set_verify_data(self, data_root, verify_file):
        self.verify_id = []
        self.verify_img_path = []
        train_file_buf = open(verify_file)
        line = train_file_buf.readline().strip()
        while line:
            path, id = line.split(' ')
            fullpath = osp.join(data_root, path)
            if osp.isfile(fullpath):
                self.verify_id.append(id)
                self.verify_img_path.append(fullpath)
            else:
                logger.error(f'{fullpath} does not exist.')
            line = train_file_buf.readline().strip()
        
        logger.info(f'Set verify data success, ID: {len(set(self.verify_id))} Image: {len(self.verify_img_path)}')

    def get_enroll_data(self):
        enroll_id = []
        enroll_img = []
        for id, imgpath in zip(self.enroll_id, self.enroll_img_path):
            enroll_id.append(id)
            enroll_img.append(cv2.imread(imgpath))
        return enroll_id, enroll_img
        
    def get_one_testdata(self):
        id = self.verify_id[self.verify_index]
        img = cv2.imread(self.verify_img_path[self.verify_index])
        trans_img = self.seq.augment_images([img])[0]
        self.verify_index += 1
        if self.verify_index >= len(self.verify_img_path) - 1:
            self.verify_index = 0
        return id, trans_img
