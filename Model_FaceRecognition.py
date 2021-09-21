import cv2
import torch
import numpy as np
import configparser
from ModelHandler import ModelHandler
from sklearn.metrics.pairwise import cosine_similarity

from backbone.iresnet import iresnet50

class FaceRecognition(ModelHandler):
    def __init__(self, configfile):
        super().__init__()
        self.enroll_id = []
        self.enroll_feature = []
        self.read_config(configfile)
        self.load_model()

    def read_config(self, configfile):
        config = configparser.ConfigParser()
        config.read(configfile)
        input_size = config['model'].getint('input_size')
        self.input_size = (input_size, input_size)
        self.model_path = config['model']['model_path']
        self.use_cuda = config['model'].getboolean('use_cuda')
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.thresh = config['model'].getfloat('thresh')

    def load_model(self):
        self.model = iresnet50()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device).eval()

    def preprocessing(self, npimg):
        if npimg.shape[:2] != self.input_size:
            npimg = cv2.resize(npimg, self.input_size).astype(np.float32)
        npimg = (npimg / 255. - 0.5) / 0.5
        npimg = npimg.transpose((2, 0, 1))
        tensor = torch.from_numpy(npimg).unsqueeze(0).float().to(self.device)
        return tensor

    def enrollment(self, id, npimg):
        id = str(id)
        if id == '-1': raise ValueError
        input = self.preprocessing(npimg)
        output = self.model(input).cpu().detach().numpy()[0]
        self.enroll_id.append(id)
        self.enroll_feature.append(output)

    def verification(self, npimg):
        input = self.preprocessing(npimg)
        output = self.model(input).cpu().detach().numpy()
        scores = cosine_similarity(output, self.enroll_feature)[0]
        highest_score_index = np.argmax(scores)
        highest_score = scores[highest_score_index]
        verified_id = self.enroll_id[highest_score_index]
        if highest_score < self.thresh:
            verified_id = '-1'
        return verified_id, highest_score
