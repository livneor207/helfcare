import cv2
from matplotlib import pyplot as plt

from HW2.PneumothoraxSegmentationProject.Utilities.MaskUtilities import get_bounding_box


def show_patient_data(patient_file_name: str):
    pass
    # # displaying metadata
    # data = pydicom.dcmread(patient_file_name)
    # print(data)
    #
    # # displaying the image
    # img = pydicom.read_file(patient_file_name).pixel_array
    # plt.figure(0)
    # plt.grid(False)
    # plt.imshow(img, cmap='bone')


# def plot_imgs(uids_list, images_df, rle_encodings_df):
#     for uid in uids_list:
#         # prepare data for plots
#         try:
#             image = get_image_by_uid(images_df, uid)
#             mask = get_mask_by_uid(rle_encodings_df, image, uid)
#             # blob, blob_number = label(mask, return_num=True) # to talk with roi about this part
#             rmin, cmin, rmax, cmax = get_bounding_box(mask)
#         except Exception as e:
#             raise e
#             print(f'could not process image with uid {uid}.\nreason: {e}')
#             continue
#
#         # Create the figure
#         fig, axes = plt.subplots(1, 4, figsize=(20, 15))
#
#         # Create the plot for the original image
#         axes[0].imshow(image, cmap='bone')
#         axes[0].set_title('Original Image')
#
#         # Create the plot for the Pneumathorax mask
#         axes[1].imshow(mask, cmap='gray')
#         axes[1].set_title('Mask Only')
#
#         # Create the plot for the original image with the mask on top of it
#         axes[2].imshow(image, cmap='bone')
#         axes[2].imshow(mask, alpha=0.3, cmap='Reds')
#         axes[2].set_title('Image + mask')
#
#         # Create the plot for original image +mask+ mask bounding box
#         cv2.rectangle(image, (cmin, rmin), (cmax, rmax), (255, 255, 0), 5)
#         axes[3].imshow(image)
#         axes[3].imshow(mask, alpha=0.3, cmap='Reds')
#         axes[3].set_title('Image + Mask + Bounding Box')
#
#         # Finally, show image
#         plt.show()


def plot_images(dataset, indices: list):
    for index in indices:
        # prepare data for plots
        try:
            record = dataset[index]
            image = record['Image']
            mask = record['Mask']
            top_left, bottom_right = get_bounding_box(mask)
        except Exception as e:
            raise e

        # Create the figure
        fig, axes = plt.subplots(1, 3, figsize=(10, 8))

        # Create the plot for the original image
        axes[0].imshow(image, cmap='bone')
        axes[0].set_title('Original Image')

        # Create the plot for the Pneumathorax mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask Only')

        # Create the plot for original image +mask+ mask bounding box
        if top_left is not None and bottom_right is not None:
            cv2.rectangle(image, top_left, bottom_right, (255, 255, 0), 5)
        axes[2].imshow(image)
        axes[2].imshow(mask, alpha=0.3, cmap='Reds')
        axes[2].set_title('Image + Mask + Bounding Box')

        # Finally, show image
        plt.show()
