import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from scipy.ndimage import uniform_filter

# ===============================================================
#                     FILE PATHS
# ===============================================================
path_abs = r"/home/gabriel/Descargas/BIO_S1_SCS__1S_20251201T192339_20251201T192356_T_G01_M01_C04_T027_F144_01_DKQ4ID/measurement/bio_s1_scs__1s_20251201t192339_20251201t192356_t_g01_m01_c04_t027_f144_i_abs.tiff"
path_phase = r"/home/gabriel/Descargas/BIO_S1_SCS__1S_20251201T192339_20251201T192356_T_G01_M01_C04_T027_F144_01_DKQ4ID/measurement/bio_s1_scs__1s_20251201t192339_20251201t192356_t_g01_m01_c04_t027_f144_i_phase.tiff"

# ===============================================================
#                     READ AMPLITUDE BANDS
# ===============================================================
with rasterio.open(path_abs) as src:
    print("Amplitude bands:", src.count)
    hh_abs = src.read(1)
    hv_abs = src.read(2)
    vh_abs = src.read(3)
    vv_abs = src.read(4)
    profile = src.profile  # Save metadata for export if needed

print("Original HH shape:", hh_abs.shape)

with rasterio.open(path_phase) as src:
    print("Amplitude bands:", src.count)
    hh_phase = src.read(1)
    hv_phase = src.read(2)
    vh_phase = src.read(3)
    vv_phase = src.read(4)
    profile = src.profile  # Save metadata for export if needed

print("Original HH shape:", hh_abs.shape)

# here we combine the abs, and phase values to form a complex number 
HH = hh_abs * np.exp(1j*hh_phase)
HV = hv_abs * np.exp(1j*hv_phase)
VH = vh_abs * np.exp(1j*vh_phase)
VV = vv_abs * np.exp(1j*vv_phase)

from scipy import signal
win = [11, 3]

kernel  = np.ones((win[0],win[1]),np.float32)/(win[0]*win[1])


k1Full = ((HH+VV))
k2Full = ((HH-VV))
k3Full = ((HV+VH))

da_full = np.shape(HH)[0]  
dr_full = np.shape(HH)[0]
    
# del HHFull, VVFull, VHFull, HVFull

# Now we do average and decimation to get the elmemehts of the Coherency matrix.
# Filtering the images again
T11Full =  signal.convolve2d(np.abs(k1Full)**2, kernel, 
                              mode='same', boundary='fill', fillvalue=0) 
T11 = T11Full[::win[0],::win[1]]

# Now let's get the other diagonal elements
T22Full =  signal.convolve2d(np.abs(k2Full)**2, kernel, 
                              mode='same', boundary='fill', fillvalue=0) 
T22 = T22Full[::win[0],::win[1]]

T33Full =  signal.convolve2d(np.abs(k3Full)**2, kernel, 
                              mode='same', boundary='fill', fillvalue=0) 
T33 = T33Full[::win[0],::win[1]]

# Now let's get the off diagonal elements 
T12Full =  signal.convolve2d(k1Full*np.conj(k2Full), kernel, 
                              mode='same', boundary='fill', fillvalue=0) 
T12 = T12Full[::win[0],::win[1]]

T13Full =  signal.convolve2d(k1Full*np.conj(k3Full), kernel, 
                              mode='same', boundary='fill', fillvalue=0) 
T13 = T13Full[::win[0],::win[1]]

T23Full =  signal.convolve2d(k2Full*np.conj(k3Full), kernel, 
                              mode='same', boundary='fill', fillvalue=0) 
T23 = T23Full[::win[0],::win[1]]

# Once we have all the elements we can visualise the diagonal (powers) in an RGB
size = np.shape(T11)     # this function tell us the size
iRGBPauli = np.zeros([size[0],size[1],3])    # create the 3D container
iRGBPauli[:,:,0] = T22/(T22.mean()*1.5)
iRGBPauli[:,:,1] = T33/(T33.mean()*1.5)
iRGBPauli[:,:,2] = T11/(T11.mean()*1.5)
iRGBPauli[np.abs(iRGBPauli) > 1] = 1

# path_out = str(path_save) + '/RGB_Pauli_full_' + flag_image
# Example: take only every 10th pixel in each dimension
#iRGBPauli_small = iRGBPauli[::10, ::10, :]
fig, ax = plt.subplots(figsize=(12,12))
ax.imshow(iRGBPauli)

ax.set_title("Pauli RGB Composite (Multilooked)", fontsize=16)
ax.set_xlabel("Easting (m)", fontsize=12)
ax.set_ylabel("Northing (m)", fontsize=12)
ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)
plt.axis("oN")

# Legend for RGB channels
legend_elements = [
    Patch(facecolor='red', edgecolor='r', label='HH - VV (Double-bounce)'),
    Patch(facecolor='green', edgecolor='g', label='HV + VH (Volume)'),
    Patch(facecolor='blue', edgecolor='b', label='HH + VV (Surface)')
]
ax.legend(handles=legend_elements, 
          loc='upper left',        # anchor point of the legend box
          bbox_to_anchor=(1.05, 1), # move outside the axes (right side)
          fontsize=10, 
          framealpha=0.7)

ax.tick_params(axis='both', which='major', labelsize=10)


plt.show()

# fig.savefig(path_out, bbox_inches='tight', pad_inches=0)

### ---------------PART 2 --------------
looksa = 19
looksr = 3

pauli1 = np.sqrt(uniform_filter(np.abs((HH+VV) / np.sqrt(2))**2, [looksa, looksr]))
pauli2 = np.sqrt(uniform_filter(np.abs((HH-VV) / np.sqrt(2))**2, [looksa, looksr]))
pauli3 = np.sqrt(uniform_filter(np.abs((HV+VH) / np.sqrt(2))**2, [looksa, looksr]))

# initialize array
#rgb_pauli = np.zeros((dimrg, dimaz, 3), 'float32')
"""
rgb_pauli[:,:,0] = np.clip(np.transpose(pauli2), 0, 2.5*np.mean(pauli2)) #red
rgb_pauli[:,:,1] = np.clip(np.transpose(pauli3), 0, 2.5*np.mean(pauli3)) #green
rgb_pauli[:,:,2] = np.clip(np.transpose(pauli1), 0, 2.5*np.mean(pauli1)) #blue

rgb_pauli[:,:,0] = rgb_pauli[:,:,0] / np.max(rgb_pauli[:,:,0])
rgb_pauli[:,:,1] = rgb_pauli[:,:,1] / np.max(rgb_pauli[:,:,1])
rgb_pauli[:,:,2] = rgb_pauli[:,:,2] / np.max(rgb_pauli[:,:,2])
"""
# plots

"""
plt.figure(figsize = (10,14))

## lexicographic
ax = plt.subplot(2,1,1)
plt.imshow(iRGBPauli, aspect = 'auto')
ax.set_title('RGB lexicographic basis')
plt.tight_layout()

## Pauli
ax = plt.subplot(2,1,2)
plt.imshow(rgb_pauli, aspect = 'auto')
ax.set_title('RGB Pauli basis')
plt.tight_layout()
"""

# Compute the alpha angle

# --- calculate the length of the Pauli vector
pauli_l = np.sqrt(np.abs(pauli1)**2 + np.abs(pauli2)**2 + np.abs(pauli3)**2)

# --- compute alpha angle [rad]
# TODO: Complete
alpha = np.arccos(np.abs(pauli1)/pauli_l)

# plots

plt.figure(figsize=(10,12))
plt.subplot(2,1,1)
plt.imshow(iRGBPauli, aspect='auto')
plt.colorbar() # dummy colorbar to align images
plt.subplot(2,1,2)
plt.imshow((alpha) * 180/np.pi , cmap = 'jet', vmin = 0, vmax = 90, aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
plt.tight_layout()
plt.show()

# -------- PART 3 -----------


# Join Pauli elements into Pauli scattering vector w_p
wp = np.dstack((pauli1, pauli2, pauli3))

# w_p is a 3 element vector for each pixel
wp.shape




# Compute coherency matrix T3 as the outer product of w_p with w_p.conj()
T3 = np.einsum('...i,...j->...ij', wp, wp.conj())

# Delete wp to save memory
del wp

# T3 is a 3 by 3 matrix for each pixel
T3.shape

rows, cols = T3.shape[:2]
W = np.zeros((rows, cols, 3), dtype=np.float32)
V = np.zeros((rows, cols, 3, 3), dtype=np.complex64)

for i in range(rows):
    for j in range(cols):
        w, v = np.linalg.eigh(T3[i, j, :, :])
        W[i, j, :] = w      # eigenvalues
        V[i, j, :, :] = v   # eigenvectors

# Number of pixels to average in azimuth and range for the Multilook
looksa = 19
looksr = 3

# Perform the multilook over the T3 matrix
T3 = uniform_filter(T3, (looksa, looksr, 1, 1))

T3.shape

# Check that T3 matrix is Hermitian
np.set_printoptions(precision=3)
T3[0,0,:,:]

# Perform eigendecomposition of T3 (hermitian) matrix
W, V = np.linalg.eigh(T3)

# W contains the eigenvalues and V the eigenvectors
W.shape
V.shape

# Note that np.linalg.eigh returns the eigenvalues & eigenvectors ordered in ascending order --> lambda_1 (larger) is the last one
# TODO: Complete
lambda1 = W[:, :, 2]
lambda2 = W[:, :, 1] 
lambda3 = W[:, :, 0]

# --- Compute probabilities
# TODO: Complete
pr1 = lambda1 / (lambda1 + lambda2 + lambda3)
pr2 = lambda2 / (lambda1 + lambda2 + lambda3)
pr3 = lambda3 / (lambda1 + lambda2 + lambda3)


# --- Compute entropy
entropy = - ( pr1*np.log10(pr1)/np.log10(3) + pr2*np.log10(pr2)/np.log10(3) + pr3*np.log10(pr3)/np.log10(3) )
"""

# TODO: Complete
anisotropy = (lambda2-lambda3) / (lambda2+lambda3)

# Note that np.linalg.eigh returns the eigenvectors ordered in ascending order --> U1 corresponding to lambda_1 is the last one
U1 = V[:, :, :, 2]
U2 = V[:, :, :, 1]
U3 = V[:, :, :, 0]

U1.shape
# extract alpha angles

alpha1 = np.arccos(np.abs(U1[:,:,0]))
alpha2 = np.arccos(np.abs(U2[:,:,0]))
alpha3 = np.arccos(np.abs(U3[:,:,0]))

# calculate the mean alpha angle
# TODO: Complete
alpha = (pr1*alpha1 + pr2*alpha2 + pr3*alpha3) 
alpha = alpha * 180/np.pi   # [deg]

# -- Generate Pauli RGB from coherency matrix diagonal elements
# NOTE: square root applied to convert intensities to amplitudes for visualization
# NOTE: the ordering of channels R, G, B --> Pauli 2, Pauli 3, Pauli 1
naz = pauli1.shape[0]
nrg = pauli1.shape[1]
pauli_rgb = np.zeros((nrg, naz, 3), 'float32')

T11 = T3[:, :, 0, 0]
T22 = T3[:, :, 1, 1]
T33 = T3[:, :, 2, 2]
pauli_rgb[:,:,0] = np.clip(np.transpose(np.sqrt(np.abs(T22))), 0, 2.5*np.mean(np.sqrt(np.abs(T22)))) 
pauli_rgb[:,:,1] = np.clip(np.transpose(np.sqrt(np.abs(T33))), 0, 2.5*np.mean(np.sqrt(np.abs(T33)))) 
pauli_rgb[:,:,2] = np.clip(np.transpose(np.sqrt(np.abs(T11))), 0, 2.5*np.mean(np.sqrt(np.abs(T11)))) 

pauli_rgb[:,:,0] = pauli_rgb[:,:,0] / np.max(pauli_rgb[:,:,0]) 
pauli_rgb[:,:,1] = pauli_rgb[:,:,1] / np.max(pauli_rgb[:,:,1]) 
pauli_rgb[:,:,2] = pauli_rgb[:,:,2] / np.max(pauli_rgb[:,:,2]) 

# Plot of entropy / mean alpha / anistropy

plt.figure(figsize = (10, 6*3))
ax = plt.subplot(3,1,1)
plt.imshow(np.transpose(entropy), vmin = 0, vmax = 1, cmap = 'jet', aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title('Entropy H')

ax = plt.subplot(3,1,2)
plt.imshow(np.transpose(alpha), vmin = 0, vmax = 90, cmap = 'jet', aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title('Mean alpha angle')

ax = plt.subplot(3,1,3)
plt.imshow(np.transpose(anisotropy), vmin = 0, vmax = 1, cmap = 'jet', aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title('Anisotropy')

plt.tight_layout()
"""

plt.figure(figsize=(10,12))
plt.subplot(2,1,1)
plt.imshow(iRGBPauli, aspect='auto')
plt.colorbar() # dummy colorbar to align images
plt.subplot(2,1,2)
plt.imshow((entropy) * 180/np.pi , cmap = 'jet', vmin = 0, vmax = 90, aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
plt.tight_layout()
plt.show()