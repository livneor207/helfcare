
import os 
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
import time
#__________________________________________________________________________
# Step 1: 
    
# Step 1.1  -- generate head phantom 

svg_ending = '.svg'
png_ending = '.png'
path  = r'C:\MSC\HC\HW1\Shepp_logan.png'

# get image 
image = shepp_logan_phantom()
#image = rescale(image, scale=0.4, mode='reflect')

    
# Step 1.2 -- synthetic projection using radon transform



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
    
    
fig, ax = plt.subplots(8, 1, figsize=(400, 100))
time_list = [] 
amount_of_projection_angels_list = []
for figure_idx in range(0,8):
    if figure_idx<4:
        amount_of_projection_angels =  int(0.5*(figure_idx+1)*(max(image.shape)//ax.shape[0]))
    else:
        amount_of_projection_angels =  int(1.5*(figure_idx+1)*(max(image.shape)//ax.shape[0]))
    amount_of_projection_angels_list.append(amount_of_projection_angels_list)
    theta = np.linspace(0., 180., amount_of_projection_angels, endpoint=False)
    t1 = time.time()
    sinogram = radon(image, theta=theta)
    randon_time  =  np.round(time.time()-t1,2)
    print(randon_time)
    time_list.append(randon_time)
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    # plt.figure()
    # plt.imshow(sinogram, cmap=plt.cm.Greys_r,
    #             extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    #             aspect='auto')
   
    # ax[figure_idx].set_title("Radon transform -# projection angels="+str(amount_of_projection_angels)+"\n(Sinogram)")
    # ax[figure_idx].set_xlabel("Projection angle (deg)")
    # ax[figure_idx].set_ylabel("Projection position (pixels)")
    # ax[figure_idx].imshow(sinogram, cmap=plt.cm.Greys_r,
    #             extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    #             aspect='auto')

#plt.show()

plt.figure()
plt.plot(amount_of_projection_angels_list, time_list)
plt.xlabel('# projection angels')
plt.ylabel('time [Sec]')
plt.grid()
plt.title('time as function of projection angels')
plt.show()
# Step 2.2  - •	Explain the effect of varying the angles:
"""
bla bla bla 
"""
    

#______________________________________________________________________________
# Step 3 - difference of performing back projection and filtered back projection:
    
    


#______________________________________________________________________________
# Step 4 - Apply & describe an algebraic iterative reconstruction technique (CGLS\SIRT):
    
    
    
    
    
    
    
    
    