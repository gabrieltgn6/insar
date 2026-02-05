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

# Crop images in azimuth
crop_az = (6000, 12000)
HH = HH[crop_az[0]:crop_az[1], :]
HV = HV[crop_az[0]:crop_az[1], :]
VH = VH[crop_az[0]:crop_az[1], :]
VV = VV[crop_az[0]:crop_az[1], :]

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
"""
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
"""
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


# plots
"""
plt.figure(figsize=(10,12))
plt.subplot(2,1,1)
plt.imshow(iRGBPauli, aspect='auto')
plt.colorbar() # dummy colorbar to align images
plt.subplot(2,1,2)
plt.imshow((alpha) * 180/np.pi , cmap = 'jet', vmin = 0, vmax = 90, aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
plt.tight_layout()
plt.show()
"""

# -------- PART 3 -----------
"""

# Crop images in azimuth
crop_az = (6000, 12000)
HH = HH[crop_az[0]:crop_az[1], :]
HV = HV[crop_az[0]:crop_az[1], :]
VH = VH[crop_az[0]:crop_az[1], :]
VV = VV[crop_az[0]:crop_az[1], :]

"""
def HSV_colormap_to_rgb(colormap, h, s, v):
    """
    Converts H, S, V to RGB using a colormap.
    Automatically aligns dimensions if arrays are multilooked or cropped.
    """
    # Ensure h is in [0,1]
    h = np.clip(h, 0, 1)

    # Get RGB from colormap
    base_rgb = colormap(h)[..., :3]

    # Align dimensions
    if base_rgb.shape[:2] != v.shape[:2]:
        base_rgb = base_rgb[:v.shape[0], :v.shape[1], :]
        s_fixed = s[:v.shape[0], :v.shape[1]]
        h = h[:v.shape[0], :v.shape[1]]
    else:
        s_fixed = s

    # Linear interpolation HSV -> RGB
    tmp = (1 - s_fixed)[..., np.newaxis] * np.ones(3) + s_fixed[..., np.newaxis] * base_rgb

    # Ensure v has last axis
    if v.ndim == 2:
        v = v[..., np.newaxis]

    return v * tmp


#HH.shape

# --- calculate Pauli elements
# TODO: Complete
pauli1 = (HH+VV)/np.sqrt(2)
pauli2 = (HH-VV)/np.sqrt(2)
pauli3 = (VH+HV)/np.sqrt(2)
# Delete original SLCs to save memory
#del HH, VV, HV, VH
# Join Pauli elements into Pauli scattering vector w_p
wp = np.dstack((pauli1, pauli2, pauli3))

# w_p is a 3 element vector for each pixel
wp.shape

# Compute coherency matrix T3 as the outer product of w_p with w_p.conj()
T3 = np.einsum('...i,...j->...ij', wp, wp.conj())

# Delete wp to save memory
#del wp

# T3 is a 3 by 3 matrix for each pixel
T3.shape

# Number of pixels to average in azimuth and range for the Multilook
looksa = 19
looksr = 3

# Perform the multilook over the T3 matrix
T3 = uniform_filter(T3, (looksa, looksr, 1, 1))

T3.shape

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

lambda1.shape


# --- Compute probabilities
# TODO: Complete
pr1 = lambda1 / (lambda1 + lambda2 + lambda3)
pr2 = lambda2 / (lambda1 + lambda2 + lambda3)
pr3 = lambda3 / (lambda1 + lambda2 + lambda3)


# --- Compute entropy
entropy = - ( pr1*np.log10(pr1)/np.log10(3) + pr2*np.log10(pr2)/np.log10(3) + pr3*np.log10(pr3)/np.log10(3) )


# TODO: Complete
anisotropy = (lambda2-lambda3) / (lambda2+lambda3)

# Note that np.linalg.eigh returns the eigenvectors ordered in ascending order --> U1 corresponding to lambda_1 is the last one
U1 = V[:, :, :, 2]
U2 = V[:, :, :, 1]
U3 = V[:, :, :, 0]

alpha1 = np.arccos(np.abs(U1[:,:,0]))
alpha2 = np.arccos(np.abs(U2[:,:,0]))
alpha3 = np.arccos(np.abs(U3[:,:,0]))


# calculate the mean alpha angle
# TODO: Complete
alpha = (pr1*alpha1 + pr2*alpha2 + pr3*alpha3) 
alpha = alpha * 180/np.pi   # [deg]
#alpha = np.arccos(np.abs(pauli1)/pauli_l)

alpha1= (alpha1*180)/np.pi
alpha2 = (alpha2*180)/np.pi
alpha3 = (alpha3*180)/np.pi

### HSV representation of H/alpha

colormap = plt.colormaps.get('jet')

# Normalize the alpha into 0 to 1
alpha = alpha / 90

# Intensity : amp
amp = np.sqrt(np.abs(T11) + np.abs(T22) + np.abs(T33))
amp = np.clip(amp, 0, 2.5*np.mean(amp))
amp = amp /np.max(amp)

# First case: take saturation = 1
saturation = np.ones_like(alpha)
rgb_alpha = HSV_colormap_to_rgb(colormap, alpha, saturation, amp)

# Second case: take saturation = 1 - entropy 
saturation = 1 - entropy
rgb_Halpha = HSV_colormap_to_rgb(colormap, alpha, saturation, amp)

# Plots
#hsv
plt.figure(figsize = (8, 12))
ax=plt.subplot(2,1,1)
ax.set_title("RGB Alpha")
plt.imshow((rgb_alpha), aspect = 'auto', interpolation = 'nearest')
ax=plt.subplot(2,1,2)
ax.set_title("RGB Halpha")
plt.imshow((rgb_Halpha), aspect = 'auto', interpolation = 'nearest')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,12))
ax = plt.subplot(1,1,1)
ax.set_title("RGB Pauli")
plt.imshow(iRGBPauli, aspect='auto')
plt.colorbar() # dummy colorbar to align images
plt.tight_layout()
plt.show()

# -- plot alpha1, alpha2, alpha3

plt.figure(figsize = (10, 12))

ax = plt.subplot(3,1,1)
plt.imshow((alpha1) , vmin = 0 , vmax = 90, cmap = 'jet', aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title("alpha1")

ax = plt.subplot(3,1,2)
plt.imshow((alpha2) , vmin = 0 , vmax = 90, cmap = 'jet', aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title("alpha2")

ax = plt.subplot(3,1,3)
plt.imshow((alpha3) , vmin = 0 , vmax = 90, cmap = 'jet', aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title("alpha3")

plt.tight_layout()

pauli_l = np.sqrt(np.abs(pauli1)**2 + np.abs(pauli2)**2 + np.abs(pauli3)**2)

alpha = np.arccos(np.abs(pauli1)/pauli_l)
plt.figure(figsize=(10,12))
ax = plt.subplot(3,1,1)
plt.imshow((entropy) , cmap = 'jet', vmin = 0, vmax = 1, aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title("Entropy H")
ax = plt.subplot(3,1,2)
plt.imshow((alpha) * 180/np.pi , cmap = 'jet', vmin = 0, vmax = 90, aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title("Mean Alpha angle")
ax = plt.subplot(3,1,3)
plt.imshow((anisotropy)  , cmap = 'jet', vmin = 0, vmax = 1, aspect = 'auto', interpolation = 'nearest')
plt.colorbar()
ax.set_title("Anisotropy")
plt.tight_layout()
plt.show()


"""
plt.figure(figsize=(12, 16))

# Imagen 1: Pauli (Ocupa toda la fila 1)
ax1 = plt.subplot(4, 2, (1, 2)) # El rango (1, 2) une las dos columnas
plt.imshow(iRGBPauli, aspect='equal') 
plt.title("Pauli RGB")
# Para que el colorbar no mueva el tamaño de la imagen, usamos un ScalarMappable
plt.colorbar(plt.cm.ScalarMappable(), ax=ax1, label='RGB Composite')

# Imagen 2: Alpha (Ocupa toda la fila 2)
ax2 = plt.subplot(4, 2, (3, 4))
im2 = plt.imshow(alpha * 180/np.pi, cmap='jet', vmin=0, vmax=90, aspect='equal', interpolation='nearest')
plt.title(r"$\alpha$ (Alpha Angle)")
plt.colorbar(im2, ax=ax2, label='Degrees')

# Imagen 3: Entropy (Fila 3, Columna 1)
ax3 = plt.subplot(4, 2, 5)
im3 = plt.imshow(entropy, cmap='jet', vmin=0, vmax=1, aspect='equal', interpolation='nearest')
plt.title("Entropy")
plt.colorbar(im3, ax=ax3)

# Imagen 4: Anisotropy (Fila 3, Columna 2)
ax4 = plt.subplot(4, 2, 6)
im4 = plt.imshow(anisotropy, cmap='jet', vmin=0, vmax=1, aspect='equal', interpolation='nearest')
plt.title("Anisotropy")
plt.colorbar(im4, ax=ax4)

plt.tight_layout()
plt.show()
"""


# ===============================================================
#                    HISTOGRAMS OF H, A, α
# ===============================================================

# Flatten valid pixels (exclude NaN or masked)
entropy_flat = entropy[np.isfinite(entropy)].ravel()
anisotropy_flat = anisotropy[np.isfinite(anisotropy)].ravel()
alpha_flat = alpha[np.isfinite(alpha)].ravel()

# Convert alpha to degrees if not already
if alpha_flat.max() <= 1.5:  # if normalized between 0–1
    alpha_flat = alpha_flat * 90

# Define histogram style
plt.figure(figsize=(15, 5))

# ------------------ Entropy Histogram ------------------
plt.subplot(1, 3, 1)
plt.hist(entropy_flat, bins=50, range=(0, 1), color='royalblue', edgecolor='black', alpha=0.7)
plt.title('Entropy (H)', fontsize=14)
plt.xlabel('Entropy H', fontsize=12)
plt.ylabel('Normalized Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().set_ylim(bottom=0)

# ------------------ Anisotropy Histogram ------------------
plt.subplot(1, 3, 2)
plt.hist(anisotropy_flat, bins=50, range=(0, 1), color='orange', edgecolor='black', alpha=0.7)
plt.title('Anisotropy (A)', fontsize=14)
plt.xlabel('Anisotropy A', fontsize=12)
plt.ylabel('Normalized Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().set_ylim(bottom=0)

# ------------------ Mean Alpha Histogram ------------------
plt.subplot(1, 3, 3)
plt.hist(alpha_flat, bins=60, range=(0, 90), color='seagreen', edgecolor='black', alpha=0.7)
plt.title('Mean Alpha Angle (°)', fontsize=14)
plt.xlabel('Alpha (degrees)', fontsize=12)
plt.ylabel('Normalized Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().set_ylim(bottom=0)

plt.tight_layout()
plt.show()


# Filtrar valores válidos para evitar errores en el plot
mask = np.isfinite(entropy) & np.isfinite(alpha)
h_vals = entropy[mask].ravel()
a_vals = alpha[mask].ravel() * 90  # Asegurar que esté en grados 0-90

plt.figure(figsize=(8, 6))
# Usamos hexbin para ver la densidad de puntos sin saturar el gráfico
plt.hexbin(h_vals, a_vals, gridsize=100, cmap='viridis', mincnt=1)

plt.title('Plano de Clasificación $H - \\alpha$', fontsize=14)
plt.xlabel('Entropía (Despolarización $\\leftarrow$)', fontsize=12)
plt.ylabel('Ángulo Alpha (Mecanismo de dispersión)', fontsize=12)
plt.colorbar(label='Densidad de píxeles')
plt.grid(True, linestyle='--', alpha=0.5)

# Límites teóricos del plano H-alpha
plt.xlim(0, 1)
plt.ylim(0, 90)
plt.show()