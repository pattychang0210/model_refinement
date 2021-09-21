import os
import cv2
import os.path as osp
from Model_FaceRecognition import FaceRecognition
from DataManager import DataManager

recognizer = FaceRecognition('config.ini')
data_manager = DataManager()

data_manager.set_enroll_data(
    '/data_2t/sam/fr-repos/testing/non_mask_all',
    '/data_2t/sam/fr-repos/testing/enroll.txt')

data_manager.set_verify_data(
    '/data_2t/sam/fr-repos/testing/non_mask_all',
    '/data_2t/sam/fr-repos/testing/verify.txt')

enroll_id, enroll_img = data_manager.get_enroll_data()
for i, (id, img) in enumerate(zip(enroll_id, enroll_img)):
    print(i)
    recognizer.enrollment(id, img)

outputdir = 'hard_example'
os.makedirs(outputdir, exist_ok=True)
while True:
    id, trans_img = data_manager.get_one_testdata()
    verified_id, score = recognizer.verification(trans_img)
    if verified_id != id:
        filename = f'{id}_{verified_id}_{score:.4f}.jpg'
        outputpath = osp.join(outputdir, filename)
        cv2.imwrite(outputpath, trans_img)
        print(f'Failed, score: {score:.4f}')
    else:
        print(f'Successful, score: {score:.4f}')

