from glob import glob
import cv2
import albumentations
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from albumentations import Resize
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from HW2.PneumothoraxSegmentationProject import FAST_LOAD_MODE, WANTED_IMAGE_SIZE
from HW2.PneumothoraxSegmentationProject.Utilities.MaskUtilities import get_mask_from_rle_encodings


class SIIMDataset(Dataset):
    def __init__(self, dcm_files_path: str, rle_encodings_filepath: str):
        print(f'entered __init__ of SIIMDataset')
        self._dcm_files_path = dcm_files_path
        self._rle_encodings_filepath = rle_encodings_filepath
        self._dataframe = self._get_dataframe()

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self, index):
        # TODO: add transformations
        temp = self._dataframe.iloc[index]
        temp = temp.to_dict()
        return temp

    def _get_dcm_filenames_from_dcm_files_path(self, filepath: str):
        print(f'entered `_get_dcm_filenames_from_dcm_files_path`')
        self._dcm_file_names = sorted(glob(filepath + "*/*/*.dcm"))
        return self._dcm_file_names

    def _get_rle_encodings_df(self, filepath):
        print(f'entered `_get_rle_encodings_df`')
        # read train-rle.csv
        rle_encodings_df = pd.read_csv(filepath, delimiter=",")
        rle_encodings_df.rename(columns={" EncodedPixels": "EncodedPixels", "ImageId": "UID"}, inplace=True)
        return rle_encodings_df

    def _get_dataframe(self):
        rle_encodings_df = self._get_rle_encodings_df(filepath=self._rle_encodings_filepath)
        file_names = self._get_dcm_filenames_from_dcm_files_path(filepath=self._dcm_files_path)
        patients = pd.DataFrame()

        for index, file_name in enumerate(file_names):
            try:
                data = pydicom.dcmread(file_name)
            except Exception as e:
                print(f'{type(e)} : could not read {file_name}. problem was {e}')
                continue

            # temp TODO delete
            if FAST_LOAD_MODE is True and index > 500:
                break

            # create a new empty patient
            patient = dict()

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
            patient["OriginalImage"] = data.pixel_array

            # add a label to the data - if the patient has the disease or not
            try:
                matching_records = rle_encodings_df[rle_encodings_df["UID"] == patient["UID"]]
                rle_encodings = matching_records['EncodedPixels'].to_list()
                patient["Label"] = 'Healthy' if rle_encodings == ['-1'] else 'Pneumothorax'
                patient["OriginalMask"] = get_mask_from_rle_encodings(
                    rle_encodings=rle_encodings,
                    img_width=patient["OriginalImage"].shape[1],
                    img_height=patient["OriginalImage"].shape[0])
                patient["NumOfEncodings"] = 0 if rle_encodings == ['-1'] else len(rle_encodings)

                # plt.imshow(patient['OriginalMask'], cmap='gray')


            except:
                patient["Label"] = 'NoLabel'
                patient["OriginalMask"] = np.zeros([WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE])
                patient["NumOfEncodings"] = 0

            # Resize image
            try:
                # to fit the baseline model we need the image size to be smaller.
                # since we are doing segmentation, we need to resize the image and the mask together

                # # albumentations will help with that.
                # transform = albumentations.Compose([
                #     Resize(height=WANTED_IMAGE_SIZE,
                #            width=WANTED_IMAGE_SIZE,
                #            interpolation=cv2.INTER_LANCZOS4)  # please see interpolation method !!!
                # ])
                # transformed = transform(image=patient["OriginalImage"], mask=patient["OriginalMask"])
                # patient["Image"] = transformed['image']
                # patient["Mask"] = transformed['mask']

                # transform for square resize
                # transform = T.ToPILImage()
                # transform = T.Resize((WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE))

                transform = T.Compose([
                    T.Resize(WANTED_IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ])

                # convert to tensors
                # patient["OriginalImage"] = torch.from_numpy(patient["OriginalImage"])
                # patient["OriginalMask"] = torch.from_numpy(patient["OriginalMask"])

                # convert images to PIL format so that they can be transformed using torchbvision
                patient["OriginalImage"] = Image.fromarray(patient["OriginalImage"])
                patient["OriginalMask"] = Image.fromarray(patient["OriginalMask"])

                # transform
                patient["Image"] = transform(patient["OriginalImage"])
                patient["Mask"] = transform(patient["OriginalMask"])

                # plt.figure()
                # plt.imshow(patient['OriginalMask'], cmap='gray')
                # plt.imshow(torch.squeeze(patient['Mask'], dim=0).numpy(), cmap='gray')
                # plt.show()
                # # plt.figure()
                # # plt.imshow(torch.squeeze(patient['Image'], dim=0).numpy(), cmap='gray')
                # # plt.show()


            except Exception as e:
                print(f'problem with image resize ! ', patient['UID'], f'problem was {type(e), e}')
                patient["OriginalImage"] = np.zeros([WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE])
                patient["Mask"] = np.zeros([WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE])
                raise e  # could not process

            # finally
            patient.pop("OriginalImage", None)  # not needed anymore
            patient.pop("OriginalMask", None)  # not needed anymore
            patients = patients.append(patient, ignore_index=True)

        # return the dataframe as output
        return patients
