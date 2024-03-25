# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:42:10 2023

@author: pymte
"""

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmath
from time import time

plt.close('all')

PDB_ID = '6j6j' #input the Protein Data Bank ID of chosen molecule
pdb_file = f'pdb_files/{PDB_ID}.pdb' #Note: ensure PDB file is downloaded in .pdb format 

#Settings........................................................................
r =1               #Tip radius (nm)
angle = 6          #Cone angle (o)
pix_per_nm = 6.25  #Sampling (pix/nm)
noise = 0.1          #Noise (rms nm)

#rotation about different axis:
theta_x = 90.0    
theta_y = 0.0
theta_z = 0.0

z_thresh = 0.5  #Set the fraction (0-1) of coordinates to exclude (e.g. membrane embedded fraction)

#Calculations....................................................................
start1 = time()
protein = md.load(pdb_file) # Loads the PDB file using mdtraj

# Get the coordinates of all atoms in the PDB file
coords = protein.xyz
coords = coords[0,:,:] #Given initially as a 1xNx3 matrix. Only want Nx3

end1= time()
time_taken1 = end1 - start1
print(f'Time to collect atomic coords: {time_taken1}s')

start2= time()
# rotations
if theta_x > 0:
    radian_angle_x = np.deg2rad(theta_x)  # Convert degrees to radians
    coords_rx = coords.copy()  # Create a copy of 'coords' to store the rotated values

    # Perform the rotation for Y and Z coordinates
    coords_rx[:, 1] = coords[:, 1] * np.cos(radian_angle_x) - coords[:, 2] * np.sin(radian_angle_x)
    coords_rx[:, 2] = coords[:, 1] * np.sin(radian_angle_x) + coords[:, 2] * np.cos(radian_angle_x)

    # Overwrite the original Y and Z coordinates with the rotated values
    coords[:, 1:3] = coords_rx[:, 1:3]

   
if theta_y > 0:
    radian_angle_y = np.deg2rad(theta_y)
    coords_ry = coords.copy()  # Create a copy of 'coords' to store the rotated values

    # Perform the Y-axis rotation
    coords_ry[:, 0] = coords[:, 0] * np.cos(radian_angle_y) + coords[:, 2] * np.sin(radian_angle_y)
    coords_ry[:, 2] = coords[:, 2] * np.cos(radian_angle_y) - coords[:, 0] * np.sin(radian_angle_y)

    # Overwrite the original X and Z coordinates with the rotated values
    coords[:, 0] = coords_ry[:, 0]
    coords[:, 2] = coords_ry[:, 2]

# Perform rotation around the Z-axis (theta_z)
if theta_z > 0:
    radian_angle_z = np.deg2rad(theta_z)
  
    v = np.vstack((coords[:, 0], coords[:, 1]))
    R = np.array([[np.cos(radian_angle_z), np.sin(radian_angle_z)],
              [-np.sin(radian_angle_z), np.cos(radian_angle_z)]])
    so = np.dot(R, v)
    

    # Update the X and Y coordinates
    coords[:, 0] = so[0, :]
    coords[:, 1] = so[1, :]  

if z_thresh > 0:
    z_thresh = z_thresh * (np.max(coords[:, 2]) - np.min(coords[:, 2])) + np.min(coords[:, 2])
    mask = coords[:, 2] > z_thresh

# Use numpy.where to get the indices where the condition is met
    indices = np.where(mask)

# Extract the corresponding elements from coords
    coords = coords[indices]

# Shift the z-coordinates
coords[:, 2] = coords[:, 2] - np.min(coords[:, 2])

# Calculate pixel scaling for tip radius
rs = r * pix_per_nm

# Set image size
fspace = (np.max(coords[:, 2]) - r) * np.tan(angle * np.pi / 180) * pix_per_nm + 1


end_pos = [int(np.floor(min(coords[:, 0] * pix_per_nm) - rs - fspace)),
           int(np.ceil(max(coords[:, 0] * pix_per_nm) + rs + fspace)),
           int(np.floor(min(coords[:, 1] * pix_per_nm) - rs - fspace)),
           int(np.ceil(max(coords[:, 1] * pix_per_nm) + rs + fspace))]

# Create an image matrix
img = np.zeros((end_pos[1] - end_pos[0] + 1, end_pos[3] - end_pos[2] + 1))

# Initialize coords_s
coords_s = np.zeros((coords.shape[0], 3))

# Pixel scaling of coordinates
coords_s[:, 2] = coords[:, 2]
coords_s[:, 0] = coords[:, 0] * pix_per_nm
coords_s[:, 1] = coords[:, 1] * pix_per_nm

for i in range(len(coords)):
    offs_x = round(coords_s[i, 0]) - coords_s[i, 0]
    offs_y = round(coords_s[i, 1]) - coords_s[i, 1]
    
    dx = np.arange(-rs - fspace, rs + fspace) #+1) 
    dy = np.arange(-rs - fspace, rs + fspace) #+1)
    dx, dy = np.meshgrid(dx, dy, indexing='ij')
 
    
    dxl = dx.ravel('F') - offs_x
    dyl = dy.ravel('F') - offs_y
    
    
    dh = np.zeros(len(dxl))
    h = np.zeros(len(dxl))

    for j in range(len(dxl)):
        
        dh[j] = np.real(cmath.sqrt(-((dxl[j] / pix_per_nm) ** 2) - ((dyl[j] / pix_per_nm) ** 2) + r ** 2)) - r #tip interaction
        
        if dh[j] > -r:
            h[j] = coords_s[i, 2] + dh[j]
        else:
            di = np.real(cmath.sqrt(((dxl[j] / pix_per_nm) ** 2) + ((dyl[j] / pix_per_nm) ** 2))) - r
            h[j] = coords_s[i, 2] - r - (di / np.tan(angle * np.pi / 180))
           
        pos_x = round(coords_s[i, 0] + dxl[j] - end_pos[0] )
        pos_y = round(coords_s[i, 1] + dyl[j] - end_pos[2] )
        
        if img[pos_x, pos_y] < h[j]:
            img[pos_x, pos_y] = h[j]


img_n = img+ np.random.randn(*img.shape)*noise #Add noise

# Create a figure and display the image with the AFM colormap
AFM = np.load('AFM_cmap.npy')
AFM = ListedColormap(AFM) 

plt.imshow(img_n, cmap=AFM)
cbar = plt.colorbar() 
cbar.set_label('Heigt (nm)', fontsize=14) 
cbar.ax.tick_params(labelsize=12) 
plt.show()

x_tick_values = plt.xticks()[0][1:]/pix_per_nm
y_tick_values = plt.yticks()[0][1:]/pix_per_nm

x_tick_labels = [round(x, 2) for x in x_tick_values]
y_tick_labels = [round(y, 2) for y in y_tick_values]

plt.xticks(ticks=plt.xticks()[0][1:], labels=x_tick_labels, fontsize = 12)
plt.yticks(ticks=plt.yticks()[0][1:], labels=y_tick_labels, fontsize = 12)

plt.xlabel('nm', fontsize = 14)
plt.ylabel('nm', fontsize = 14)
plt.axis('equal')
plt.axis('tight')
plt.show()

end2= time()
time_taken2 = end2 - start2
print(f'Time to iterate through coords and calculate surface: {time_taken2}s')
