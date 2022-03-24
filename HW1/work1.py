import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale
import time
#__________________________________________________________________________
# Step 1: 
    
# Step 1.1  -- generate head phantom 

svg_ending = '.svg'
png_ending = '.png'
path  = r'C:\MSC\HC\HW1\Shepp_logan.png'

# get image 
image = shepp_logan_phantom()


# now, let's make a circular mask with a radius of 100 pixels and
# apply the mask again
big_circle = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(big_circle, (200, 200), 50, 255, -1)
big_circle_redunction = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(big_circle_redunction, (200, 200), 40, 255, -1)
small_circle = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(small_circle, (200, 200), 10, 255, -1)
mask1 = big_circle-big_circle_redunction+small_circle

mask2 = np.zeros((image.shape[:2]))
mask2[185:215, :] = 1

 
# show the output images


#image = rescale(image, scale=0.4, mode='reflect')

    
# Step 1.2 -- synthetic projection using radon transform


def generate_image_sinogram_comparation_plot(image):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
               aspect='auto')
    fig.tight_layout()
    plt.show()
    return


generate_image_sinogram_comparation_plot(image)
generate_image_sinogram_comparation_plot(mask1)
generate_image_sinogram_comparation_plot(mask2)

#______________________________________________________________________________
# Step 2: 
# Step 2.1  - perform a reconstruction for each angle:
    

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
theta_600 = np.linspace(0., 180., 600, endpoint=False)
theta_25 = np.linspace(0., 180., 25, endpoint=False)
sinogram_600 = radon(image, theta=theta_600)
sinogram_25 = radon(image, theta=theta_25)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram_25.shape[0]
ax1.set_title("Radon transform -# projection angels="+str(25)+"\n(Sinogram)")
ax1.set_xlabel("Projection angle (deg)")
ax1.set_ylabel("Projection position (pixels)")
ax1.imshow(sinogram_25, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram_600.shape[0]

ax2.set_title("Radon transform -# projection angels="+str(600)+"\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram_600, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram_600.shape[0] + dy),
           aspect='auto')

fig.tight_layout()
plt.show()
    
fig, ax = plt.subplots(8, 1, figsize=(20, 20))
time_list = [] 
amount_of_projection_angles_list = []
for figure_idx in range(0,8):
    # define amount_of_projection_angles
    if figure_idx<4:
        amount_of_projection_angles =  int(0.5*(figure_idx+1)*(max(image.shape)//ax.shape[0]))
    else:
        amount_of_projection_angles =  int(1.5*(figure_idx+1)*(max(image.shape)//ax.shape[0]))
    amount_of_projection_angles_list.append(amount_of_projection_angles)
    
    # create the sinogram (document how long it took)
    theta = np.linspace(0., 180., amount_of_projection_angles, endpoint=False)
    t1 = time.time()
    sinogram = radon(image, theta=theta)
    radon_time  =  np.round(time.time()-t1,2)
    time_list.append(radon_time)
    
    # create the plot
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    plt.imshow(sinogram, cmap=plt.cm.Greys_r,
                extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
                aspect='auto')
    ax[figure_idx].set_title("Radon transform -# projection angles="+str(amount_of_projection_angles)+"\n(Sinogram)")
    ax[figure_idx].set_xlabel("Projection angle (deg)")
    ax[figure_idx].set_ylabel("Projection position (pixels)")
    ax[figure_idx].imshow(sinogram, cmap=plt.cm.Greys_r,
                extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
                aspect='auto')

fig.tight_layout()
plt.show()

# Discuess change in run time according to number of projection angles
plt.figure()
plt.plot(amount_of_projection_angles_list, time_list)
plt.xlabel('# projection angles')
plt.ylabel('time [Sec]')
plt.grid()
plt.title('time as function of projection angles')
plt.show()
# Step 2.2  - •	Explain the effect of varying the angles:
"""
bla bla bla 
"""
    

#______________________________________________________________________________
# Step 3 - difference of performing back projection and filtered back projection:
    
# Generate the reconstructions
reconstruction_fbp_no_filter = iradon(sinogram, theta=theta, filter_name=None)
reconstruction_fbp_with_filter = iradon(sinogram, theta=theta, filter_name='ramp')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 14))

# Generate a plot for the original
ax1.set_title("Shepp-Logan head phantom")
ax1.imshow(image, cmap=plt.cm.Greys_r)

# Generate the sinogram
ax2.set_title("reconstruction_fbp_no_filter")
ax2.imshow(reconstruction_fbp_no_filter, cmap=plt.cm.Greys_r)

ax3.set_title("reconstruction_fbp_with_filter")
ax3.imshow(reconstruction_fbp_with_filter, cmap=plt.cm.Greys_r)

fig.tight_layout()
plt.show()

#______________________________________________________________________________
# Step 4 - Apply & describe an algebraic iterative reconstruction technique (CGLS\SIRT):
    
    
    
    
    
    
    
    
    