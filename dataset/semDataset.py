import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2

category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"


def place_seed_points(mask, down_stride=8, max_num_sp=5, avg_sp_area=100):
	'''
	:param mask: the RoI region to do clustering, torch tensor: H x W
	:param down_stride: downsampled stride for RoI region
	:param max_num_sp: the maximum number of superpixels
	:return: segments: the coordinates of the initial seed, max_num_sp x 2
	'''

	segments_x = np.zeros(max_num_sp, dtype=np.int64)
	segments_y = np.zeros(max_num_sp, dtype=np.int64)

	m_np = mask.cpu().numpy()
	down_h = int((m_np.shape[0] - 1) / down_stride + 1)
	down_w = int((m_np.shape[1] - 1) / down_stride + 1)
	down_size = (down_h, down_w)
	m_np_down = cv2.resize(m_np, dsize=down_size, interpolation=cv2.INTER_NEAREST)

	nz = np.nonzero(m_np_down)
	# After transform, there may be no nonzero in the label
	if len(nz[0]) != 0:

		p = [np.min(nz[0]), np.min(nz[1])]
		pend = [np.max(nz[0]), np.max(nz[1])]

		# cropping to bounding box around ROI
		m_np_roi = np.copy(m_np_down)[p[0]:pend[0] + 1, p[1]:pend[1] + 1]

		# num_sp is adaptive, based on the area of support mask
		mask_area = (m_np_roi == 1).sum()
		num_sp = int(min((np.array(mask_area) / avg_sp_area).round(), max_num_sp))

	else:
		num_sp = 0

	if (num_sp != 0) and (num_sp != 1):
		for i in range(num_sp):

			# n seeds are placed as far as possible from every other seed and the edge.

			# STEP 1:  conduct Distance Transform and choose the maximum point
			dtrans = distance_transform_edt(m_np_roi)
			dtrans = gaussian_filter(dtrans, sigma=0.1)

			coords1 = np.nonzero(dtrans == np.max(dtrans))
			segments_x[i] = coords1[0][0]
			segments_y[i] = coords1[1][0]

			# STEP 2:  set the point to False and repeat Step 1
			m_np_roi[segments_x[i], segments_y[i]] = False
			segments_x[i] += p[0]
			segments_y[i] += p[1]
	
	segments = np.concatenate([segments_x[..., np.newaxis], segments_y[..., np.newaxis]], axis=1)  # max_num_sp x 2
	segments = torch.from_numpy(segments)

	return segments

class SemDataset(Dataset):
    def __init__(self, data_dir_path, coco_list_path, 
                       data_type="train",
                       data_category = 0,
                       main_idx_list = None,
                       sub_idx_list = None,
                       sub_img_coco_list_path=None, 
                       transform=None, 
                       sub_img_list_path=None):
        # datatype in [train, val]이면 해당 coco list에서 반을 나누어 sub_img_list를 만든다.
        # datatype == test이면 train_data의 sub_img_list에서 test set의 크기만큼 임으로 sub_img_list를 만든어 낸다.
        self.data_type = data_type 
        self.data_dir_path = data_dir_path
        self.transform = transform
        self.cat = data_category
        self.max_sp = 5
        if data_type in ["train", "val"]:
            if main_idx_list is None or sub_idx_list is None:
                self.coco = COCO(coco_list_path)
                self.main_img_list, self.sub_img_list = self.seperate_data()
            else:
                self.main_img_list, self.sub_img_list = main_idx_list, sub_idx_list
        else:
            self.test_img_coco = COCO(coco_list_path)
            self.coco = COCO(sub_img_coco_list_path)
            if sub_idx_list is None:
                self.main_img_list, self.sub_img_list = self.seperate_data()
            else:
                self.main_img_list, _ = self.seperate_data()
                self.sub_img_list = sub_idx_list

    def get_idx_list(self):
        return self.main_idx_list, self.sub_img_list

    def seperate_data(self):
        if self.data_type in ["train", "val"]:
            img_list = list(self.coco.getImgIds())
            main_img_idx = random.choices(img_list, k=(len(img_list)//2))
            sub_img_idx = list(set(range(0, len(img_list))) -  set(main_img_idx))
            
            return main_img_idx, sub_img_idx
        else:
            main_img_idx = list(self.coco.getCatIds())
            sub_img_idx = random.choices(list(self.sub_img_coco.getImgIds()), k=len(main_img_idx))
            return main_img_idx, sub_img_idx
            
    def __len__(self):
        return len(self.main_img_list)

    def get_mask_img(self, img_info):
        ann_id = self.coco.getAnnIds(imgIds=img_info["id"])
        anns = self.coco.loadAnns(ann_id)

        cat_id = self.coco.getCatIds()
        cat = self.coco.loadCats(cat_id)

        # origin mask, 1: unknown.............
        masks = np.zeros((img_info['height'], img_info['width'])) 
        
        for i in range(len(anns)):
            class_name = get_classname(anns[i]['category_id'], cat)
            pixel_value = category_names.index(class_name)
            if self.cat == 0:
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            elif self.cat == pixel_value :
                masks = np.maximum(self.coco.annToMask(anns[i]), masks)
        masks = masks.astype(np.float32)
        
        return masks

    def get_seed(self, sub_img, sub_label):
        seed_list = []
        mask = (sub_label[0, :, :] == 1).float()
        seed = place_seed_points(mask, down_stride=8, max_num_sp=self.max_sp)
        seed_list.append(seed.unsqueeze(0))

        seed_list = torch.cat(seed_list, 0)
        
        return seed_list


    def __getitem__(self, idx):
        if self.data_type in ['train', 'val']:
            main_img_idx = self.coco.getImgIds(imgIds=self.main_img_list[idx])
            main_img_info = self.coco.loadImgs(main_img_idx)[0]

            sub_img_idx = self.coco.getImgIds(imgIds=self.sub_img_list[idx])
            sub_img_info = self.coco.loadImgs(sub_img_idx)[0]

            main_img = cv2.imread(os.path.join(self.data_dir_path, main_img_info['file_name']))
            main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            main_img /= 255.0

            sub_img = cv2.imread(os.path.join(self.data_dir_path, sub_img_info['file_name']))
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            sub_img /= 255.0

            main_mask = self.get_mask_img(main_img_info)
            sub_mask = self.get_mask_img(sub_img_info)

            if self.transform is not None:
                trans = self.transform(image=main_img, mask=main_mask)
                main_img = trans['image']
                main_mask = trans['mask']
                trans = self.transform(image=sub_img, mask=sub_mask)
                sub_img = trans['image']
                sub_mask = trans['mask']
            
            seed = self.get_seed(sub_img, sub_mask.unsqueeze(0))
            
            return main_img, main_mask, sub_img.unsqueeze(0), sub_mask.unsqueeze(0), seed
        else:
            main_img_idx = self.test_img_coco.getImgIds(imgIds=self.main_img_list[idx])
            main_img_info = self.test_img_coco.loadImgs(main_img_idx)[0]

            sub_img_idx = self.coco.getImgIds(imgIds=self.sub_img_list[idx])
            sub_img_info = self.coco.loadImgs(sub_img_idx)[0]

            main_img = cv2.imread(os.path.join(self.data_dir_path, main_img_info['file_name']))
            main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            main_img /= 255.0

            sub_img = cv2.imread(os.path.join(self.data_dir_path, sub_img_info['file_name']))
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            sub_img /= 255.0

            sub_mask = self.get_mask_img(sub_img_info)

            if self.transform is not None:
                trans = self.transform(image=main_img)
                main_img = trans['image']
                trans = self.transform(image=sub_img, mask=sub_masks)
                sub_img = trans['image']
                sub_mask = trans['mask']
            
            seed = self.get_seed(sub_img, sub_mask.unsqueeze(0))
            
            return main_img, None, sub_img, sub_mask, seed


if __name__=="__main__":    
    train_transform = A.Compose([
                            ToTensorV2()
                            ])

    train_dataset = SemDataset(data_dir_path="/opt/ml/input/data",
                           coco_list_path="/opt/ml/input/data/train.json",
                           data_type="train",
                           data_category=0,
                           transform=train_transform)

    main_img, main_mask, sub_img, sub_mask, seed = train_dataset[0]