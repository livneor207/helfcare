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



train_data_list =  [[dcm_file.SOPInstanceUID, dcm_file.pixel_array] for dcm_file in DCM_files]
 
 
 
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
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def get_mask_by_uid(rle_encodings_df, image, uid):
    # create the mask 
    # note: there can be more than one RLE encoding per image
    rle_encodings = rle_encodings_df[rle_encodings_df.ImageId == uid].EncodedPixels   
    final_mask = None
    for rle_encoding in rle_encodings.to_list():
        current_mask = rle2mask(rle=rle_encoding, width=image.shape[0], height=image.shape[1])
        if final_mask is None:
            final_mask = current_mask
        else:
            # print(f'another mask is added')
            final_mask += current_mask  # Important logic

    # mask needs to be rotated to fit the original image
    mask = final_mask
    mask = np.rot90(mask, 3) #rotating three times 90 to the right place
    mask = np.flip(mask, axis=1)
    
    return mask


def get_mask_by_uid(rle_encodings_df, image, uid):
    # create the mask 
    # note: there can be more than one RLE encoding per image
    rle_encodings = rle_encodings_df[rle_encodings_df.ImageId == uid].EncodedPixels   
    final_mask = None
    for rle_encoding in rle_encodings.to_list():
        current_mask = rle2mask(rle=rle_encoding, width=image.shape[0], height=image.shape[1])
        if final_mask is None:
            final_mask = current_mask
        else:
            # print(f'another mask is added')
            final_mask += current_mask  # Important logic

    # mask needs to be rotated to fit the original image
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
            mask = get_mask_by_uid(rle_encodings_df, image, uid)
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


train_metadata_df_ill = train_metadata_df[train_metadata_df['Label'] == 'Pneumothorax'].reset_index(drop=True)

results = get_total_area_of_and_metadata_of_masks(
    metadata_df=train_metadata_df_ill, 
    images_df=train_images_df, 
    rle_encodings_df=train_rle_encodings_df)

list_of_areas = [list_item['TotalArea'] for list_item in results] 
plt.figure(figsize=(15, 5))
plt.title(f'Histogram of total area of pneumathorax in pixels')
plt.hist(list_of_areas, bins = 100)
plt.xlabel("Area in pixels")
plt.ylabel("Count")
plt.show()



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
            mask = get_mask_by_uid(rle_encodings_df, image, uid)
            # blob, blob_number = label(mask, return_num=True) # to talk with roi about this part
            rmin, cmin, rmax, cmax = get_bounding_box(mask)
        except Exception as e:
            raise e
            print(f'could not process image with uid {uid}.\nreason: {e}')
            continue
        
        # Create the figure
        fig, axes = plt.subplots(1,4, figsize=(20,15))
        
        # Create the plot for the original image
        axes[0].imshow(image, cmap='bone')
        axes[0].set_title('Original Image')
        
        # Create the plot for the Pneumathorax mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask Only')
        
        # Create the plot for the original image with the mask on top of it
        axes[2].imshow(image, cmap='bone')
        axes[2].imshow(mask, alpha=0.3,cmap='Reds')
        axes[2].set_title('Image + mask')
        
        # Create the plot for original image +mask+ mask bounding box
        cv2.rectangle(image, (cmin,rmin),(cmax,rmax), (255,255,0), 5)
        axes[3].imshow(image)
        axes[3].imshow(mask,alpha=0.3,cmap='Reds')
        axes[3].set_title('Image + Mask + Bounding Box')
        
        # Finally, show image
        plt.show()
        
        
number_of_images_to_plot = 7

train_metadata_df_ill = train_metadata_df[train_metadata_df['Label'] == 'Pneumothorax'].reset_index(drop=True)
partial_ill_uids_list = train_metadata_df_ill.head(number_of_images_to_plot)['UID'].to_list()

plot_imgs(
    uids_list=partial_ill_uids_list, 
    images_df=train_images_df, 
    rle_encodings_df=train_rle_encodings_df)


        






        





    
