# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Data handling imports
import pandas as pd
pd.reset_option('max_colwidth')
import numpy as np

# Read DICOM format
import pydicom

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# General purpose imports
import os
from glob import glob

# Benchmark \ Baseline U-net
# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp
import os

from skimage.measure import label

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# general folder path
user = 'or' # 'or' \ 'roy'
if user == 'roy':
    # I downloaded the data from: 'https://www.kaggle.com/datasets/seesee/siim-train-test'
    folder_path = '/Users/royrubin/Downloads/siim/' 
else:
    folder_path = r'C:\MSC\HC\HW2\siim'
    
# paths for files
train_rle_encodings_file_path = os.path.join(folder_path, 'train-rle.csv')
train_files_path = os.path.join(folder_path, 'dicom-images-train/')
test_files_path = os.path.join(folder_path, 'dicom-images-test/')

print(f'train_rle_encodings_file_path: {train_rle_encodings_file_path}')
print(f'train_files_path: {train_files_path}')
print(f'test_files_path: {test_files_path}')


#reading all dcm files into train and text
if user in  'roy':
    # I downloaded the data from: 'https://www.kaggle.com/datasets/seesee/siim-train-test'
    train_file_names = sorted(glob(train_files_path + "*/*/*.dcm"))
    test_file_names = sorted(glob(test_files_path + "*/*/*.dcm")) # sorted([f for f in listdir(test_files_path) if '.dcm' in f])
else:
    # I downloaded the data from: 'https://www.kaggle.com/datasets/seesee/siim-train-test'
    train_file_names = sorted(glob(train_files_path + "*/*/*.dcm"))
    test_file_names = sorted(glob(test_files_path + "*/*/*.dcm")) # sorted([f for f in listdir(test_files_path) if '.dcm' in f])
    
debug_true = True
if debug_true:
    train_file_names = train_file_names[0:100]

print(f"\ntrain files: amount {len(train_file_names)}, examples: \n{train_file_names[0]}\n{train_file_names[1]}")
print(f"\ntest files: amount {len(test_file_names)}, examples: \n{test_file_names[0]}\n{test_file_names[1]}")

# read train-rle.csv
train_rle_encodings_df = pd.read_csv(train_rle_encodings_file_path, delimiter=",")
train_rle_encodings_df.rename(columns={" EncodedPixels": "EncodedPixels"}, inplace=True)



print(f'size {train_rle_encodings_df.shape}')
train_rle_encodings_df.head(3)

# reading images

# temp = [{'UID': pydicom.read_file(file_name).SOPInstanceUID, 'Image': pydicom.read_file(file_name).pixel_array} 
#                 for file_name in train_file_names]

# only rad DCM files
DCM_files = [pydicom.read_file(file_name) for file_name in train_file_names]
train_metadata = [pydicom.dcmread(file_name) for file_name in train_file_names]

# train_images = [dcm_file.pixel_array for dcm_file in DCM_files]
# train_bla = [dcm_file.SOPInstanceUID for dcm_file in DCM_files]

train_data_list =  [[dcm_file.SOPInstanceUID, cv2.cvtColor(dcm_file.pixel_array, cv2.COLOR_GRAY2RGB)] for dcm_file in DCM_files]
 
# temp = [{'UID': pydicom.read_file(file_name).SOPInstanceUID, 'Image': pydicom.read_file(file_name).pixel_array} 
#                 for file_name in train_file_names]

# get train data and meta data
train_images_df = pd.DataFrame(train_data_list, columns = ['UID', 'Image'])


def show_patient_data(patient_file_name: str):
    #displaying metadata
    data = pydicom.dcmread(patient_file_name)
    print(data)
    
    #displaying the image
    img = pydicom.read_file(patient_file_name).pixel_array
    plt.figure(0)
    plt.grid(False)
    plt.imshow(img, cmap='bone')
    
def generate_metadata_dataframe(train_metadata: list, masks: pd.DataFrame):
    
    patients = pd.DataFrame()
    # todo remove unwanted feature
    for data in train_metadata:
        patient = dict()
        # patient = data.__dict__
        # save the wanted features from the dicom foramt
        patient["UID"] = data.SOPInstanceUID
        patient["PatientID"] = data.PatientID        
        patient["Age"] = data.PatientAge
        patient["Sex"] = data.PatientSex
        patient["Modality"] = data.Modality
        patient["BodyPart"] = data.BodyPartExamined
        patient["ViewPosition"] = data.ViewPosition
        patient["Columns"] = data.Columns
        patient["Rows"] = data.Rows
        patient["PatientOrientation"] = data.PatientOrientation
        patient["PhotometricInterpretation"] = data.PhotometricInterpretation
        patient["PixelSpacing"] = data.PixelSpacing
        patient["SamplesPerPixel"] = data.SamplesPerPixel
        patient["PixelSpacing"] = data.PixelSpacing

        # add a label to the data - if the patient has the disease or not
        try:
            encoded_pixels = masks[masks["ImageId"] == patient["UID"]].values[0][1]
            # patient["EncodedPixels"] = encoded_pixels
            patient["Label"] = 'Healthy' if encoded_pixels == '-1' else 'Pneumothorax'
        except:
            patient["Label"] = 'NoLabel'
            

        patients = patients.append(patient, ignore_index=True)
    
    # return the dataframe as output
    return patients

if 0:
    train_metadata_df = generate_metadata_dataframe(train_metadata, train_rle_encodings_df)
    
    # show_patient_data(train_file_names[0])
    
    
    list_of_columns_to_drop = ['Modality','BodyPart','PatientOrientation','PhotometricInterpretation','SamplesPerPixel', 'Columns', 'Rows']
    train_metadata_df.drop(columns=list_of_columns_to_drop, inplace=True, errors='ignore')
    train_metadata_df['Age'] = train_metadata_df['Age'].apply(pd.to_numeric, errors='coerce')
    train_metadata_df.drop(train_metadata_df[train_metadata_df.Age> 120].index, inplace=True)



    for column in list(train_metadata_df.columns):
        if 'ID' not in column and 'EncodedPixels' not in column:
            counts = train_metadata_df[column].value_counts()
            print(f'\nNumber of unique values in column [{column}]: {len(counts)}, Value counts:\n{counts}\n--------------')
            
def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def get_image_by_uid(images_df, uid):
    image = images_df[images_df.UID == uid].Image.item() # item is added because the result is a series object with 1 element
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image




def get_mask_by_uid(rle_encodings_df, im_width, img_hieght , uid):
    # create the mask 
    # note: there can be more than one RLE encoding per image
    rle_encodings = rle_encodings_df[rle_encodings_df.ImageId == uid].EncodedPixels   
    final_mask = None
    for rle_encoding in rle_encodings.to_list():
        current_mask = rle2mask(rle=rle_encoding, width = im_width, height = img_hieght)
        if final_mask is None:
            final_mask = current_mask
        else:
            # print(f'another mask is added')
            final_mask += current_mask  # Important logic

    # mask needs to be rotated to fit the original image
    
    # todo seprate mask
    final_mask[final_mask>0] = 255 # all diceese the same
    mask = final_mask
    mask = np.rot90(mask, 3) #rotating three times 90 to the right place
    mask = np.flip(mask, axis=1)
    
    return mask



def get_total_area_of_and_metadata_of_masks(metadata_df, images_df, rle_encodings_df):
    """
    important note: if there are multiple notes for each image, we use the sum of these masks
    
    note: the result does not only give the area, but also some metadata for plotting later on
    """
    result = []
    for _, row in metadata_df.iterrows():
        # prepare data for plots
        try:
            uid = row['UID']
            image = get_image_by_uid(images_df, uid)
            img_hieght,im_width  = image.shape
            mask = get_mask_by_uid(rle_encodings_df, im_width, img_hieght, uid)
            pixels = np.count_nonzero(mask)
            
            # save results with additional metadata on the mask and image
            result.append(
                {
                    'TotalArea': pixels, 
                    'ViewPosition': row['ViewPosition'], 
                    'Sex': row['Sex'], 
                    'Age': row['Age'],
                    'Mask': mask,  # get the mask itself for later on when we will create heatmaps
                })
        except Exception as e:
            raise e
            print(f'could not process image with uid {uid}.\nreason: {e}')
            continue
            
    return result

if 0:
    train_metadata_df_ill = train_metadata_df[train_metadata_df['Label'] == 'Pneumothorax'].reset_index(drop=True)

# results = get_total_area_of_and_metadata_of_masks(
#     metadata_df=train_metadata_df_ill, 
#     images_df=train_images_df, 
#     rle_encodings_df=train_rle_encodings_df)

# list_of_areas = [list_item['TotalArea'] for list_item in results] 
# plt.figure(figsize=(15, 5))
# plt.title(f'Histogram of total area of pneumathorax in pixels')
# plt.hist(list_of_areas, bins = 100)
# plt.xlabel("Area in pixels")
# plt.ylabel("Count")
# plt.show()



def get_bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return rmin, cmin, rmax, cmax #ymin,xmin,ymax,cmax


def plot_imgs(uids_list, images_df, rle_encodings_df):
    for uid in uids_list:
        # prepare data for plots
        try:
            image = get_image_by_uid(images_df, uid)
            height_size, width_size = image.shape[0],image.shape[1] 
            mask = get_mask_by_uid(rle_encodings_df, height_size, width_size, uid)
            # Create the figure
            fig, axes = plt.subplots(1,4, figsize=(20,15))
            blob, blob_number = label(mask, return_num=True) # to talk with roi about this par
            axes[0].imshow(image, cmap='bone')
            axes[1].imshow(mask, cmap='gray')
            axes[2].imshow(image, cmap='bone')

            for i_blob_idx in range(1, blob_number+1):
                curr_blob = blob.copy()
                curr_blob[curr_blob!= (i_blob_idx)] = 0
                rmin, cmin, rmax, cmax = get_bounding_box(curr_blob)
              
                # Create the plot for original image +mask+ mask bounding box
                cv2.rectangle(image, (cmin,rmin),(cmax,rmax), (255,255,0), 5)
                
        
       
        except Exception as e:
                raise e
                print(f'could not process image with uid {uid}.\nreason: {e}')
                continue
        
        # Finally, show image
       
        
        # Create the plot for the Pneumathorax mask
        axes[1].set_title('Mask Only')
        
        # Create the plot for the original image with the mask on top of it
        axes[2].imshow(mask, alpha=0.3,cmap='Reds')
        axes[2].set_title('Image + mask')
        
        axes[3].imshow(image)
        axes[3].imshow(mask,alpha=0.3,cmap='Reds')
        axes[3].set_title('Image + Mask + Bounding Box')
        plt.show()
       
if 0:
    number_of_images_to_plot = 7
    
    train_metadata_df_ill = train_metadata_df[train_metadata_df['Label'] == 'Pneumothorax'].reset_index(drop=True)
    partial_ill_uids_list = train_metadata_df_ill.head(number_of_images_to_plot)['UID'].to_list()
"""
plot_imgs(
    uids_list=partial_ill_uids_list, 
    images_df=train_images_df, 
    rle_encodings_df=train_rle_encodings_df)

"""


import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings("ignore")
import segmentation_models_pytorch as smp

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component

def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle



class SIIMDataset(Dataset):
    def __init__(self, df, fnames, data_folder, size, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)
        self.gb = self.df.groupby('ImageId')
        self.fnames = fnames

    def __getitem__(self, idx):
        # image_id = self.fnames[idx]
        # df = self.gb.get_group(image_id)
        
        # annotations = df[' EncodedPixels'].tolist()
        # image_path = os.path.join(self.root, image_id + ".png")
        # image = cv2.imread(image_path)
        ########
        df_row =  self.root.iloc[idx]
        image = df_row['Image']
        uid = df_row['UID']
        height, width = image.shape
        mask = get_mask_by_uid(self.df, width, height, uid)
    
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
#                 HorizontalFlip(),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10, # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
#                 GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(
    fold,
    total_folds,
    data_folder,
    df_path,
    phase,
    size,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
):
    # df_all = pd.read_csv(df_path) # rle
    # df = df_all.drop_duplicates('ImageId')
    # df_with_mask = df[df[" EncodedPixels"] != " -1"]
    # df_with_mask['has_mask'] = 1
    # df_without_mask = df[df[" EncodedPixels"] == " -1"]
    # df_without_mask['has_mask'] = 0
    # df_without_mask_sampled = df_without_mask.sample(len(df_with_mask), random_state=69) # random state is imp
    # df = pd.concat([df_with_mask, df_without_mask_sampled])
    
    #NOTE: equal number of positive and negative cases are chosen.
    
    # kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    # train_idx, val_idx = list(kfold.split(df["ImageId"], df["has_mask"]))[fold]
    # train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    # df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%
    
    fnames = data_folder['UID'].values
    
    image_dataset = SIIMDataset(df_path, fnames, data_folder, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader





#sample_submission_path = '../input/siim-stage1/sample_submission.csv'
train_rle_path = r"C:\MSC\HC\HW2\siim\train-rle.csv"
# data_folder = "../input/siim-png-images/input/train_png"
# test_data_folder = "../input/siim-png-images/input/test_png"



# train_images_df
# train_rle_encodings_df


a=5

dataloader = provider(
    fold=0,
    total_folds=5,
    data_folder=train_images_df,
    df_path=train_rle_encodings_df,
    phase="train",
    size=512,
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    batch_size=16,
    num_workers=4,
)


batch = next(iter(dataloader)) # get a batch from the dataloader
images, masks = batch


# plot some random images in the `batch`
idx = random.choice(range(16))
plt.imshow(images[idx][0], cmap='bone')
plt.imshow(masks[idx][0], alpha=0.2, cmap='Reds')
plt.show()
if len(np.unique(masks[idx][0])) == 1: # only zeros
    print('Chosen image has no ground truth mask, rerun the cell')
    
    
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()
    
    
    
def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

#         dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
#         dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
#         dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)
        self.dice_pos_scores.extend(dice_pos)
        self.dice_neg_scores.extend(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f" % (epoch_loss, dice, dice_neg, dice_pos, iou))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou




model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)




model # a *deeper* look



class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model):
        self.fold = 1
        self.total_folds = 5
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 40
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = MixedLoss(10.0, 2.0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                fold=1,
                total_folds=5,
                data_folder=data_folder,
                df_path=train_rle_path,
                phase=phase,
                size=512,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
#         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()
            
            
            
model_trainer = Trainer(model)
model_trainer.start()

# PLOT TRAINING
losses = model_trainer.losses
dice_scores = model_trainer.dice_scores # overall dice
iou_scores = model_trainer.iou_scores

def plot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');
    plt.legend(); 
    plt.show()

plot(losses, "BCE loss")
plot(dice_scores, "Dice score")
plot(iou_scores, "IoU score")



class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                Resize(size, size),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples

def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

    
size = 512






mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 8
batch_size = 16
best_threshold = 0.5
min_size = 3500
device = torch.device("cuda:0")
train_rle_encodings_df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, size, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
model = model_trainer.net # get the model from model_trainer object
model.eval()
state = torch.load('./model.pth', map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])
encoded_pixels = []
for i, batch in enumerate(tqdm(testset)):
    preds = torch.sigmoid(model(batch.to(device)))
    preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
    for probability in preds:
        if probability.shape != (1024, 1024):
            probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        predict, num_predict = post_process(probability, best_threshold, min_size)
        if num_predict == 0:
            encoded_pixels.append('-1')
        else:
            r = run_length_encode(predict)
            encoded_pixels.append(r)
df['EncodedPixels'] = encoded_pixels
df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)



df.head()




            
            
















    
