import os.path as op 
import toblerone as tob 
from toblerone import estimators, projection 
import numpy as np 
import nibabel 
from scipy import sparse
from scipy.sparse import linalg
from tob_projectors import voxel2nodes_weights, node2voxel_weights
import matplotlib.pyplot as plt 
import pyvista
pyvista.set_plot_theme('document')
import itertools
from scipy.interpolate import interpn
plt.rcParams.update({'font.size': 14})

# %qtconsole
loadnii = lambda f: nibabel.load(f).get_fdata().reshape(-1)
loadfunc = lambda f: nibabel.load(f).darrays[0].data
MASKED_RMS = lambda a,b,mask: np.sqrt(((a[mask] - b[mask])**2).mean())

ins = tob.Surface('data/103818.L.white.32k_fs_LR.surf.gii')
mids = tob.Surface('data/103818.L.midthickness.32k_fs_LR.surf.gii')
outs = tob.Surface('data/103818.L.pial.32k_fs_LR.surf.gii')
plots = tob.Surface('data/103818.L.very_inflated.32k_fs_LR.surf.gii')
crop = ins.points.shape[0]
thickness = np.linalg.norm((outs.points - ins.points), ord=2, axis=1)
depth_mask = (thickness > 0.2)
hemi = tob.classes.Hemisphere('data/103818.L.white.32k_fs_LR.surf.gii',
                              'data/103818.L.pial.32k_fs_LR.surf.gii', 'L')
spc = tob.ImageSpace('data/tob_L_pvs.nii.gz')
pvs = loadnii('data/tob_L_pvs.nii.gz').reshape(-1,3)
hemi.apply_transform(spc.world2vox)
hemi.PVs = pvs
mask =  (pvs[:,0] > 0)
vflat = loadnii('data/flat.nii.gz')
vrand = loadnii('data/rand.nii.gz')
vsine = loadnii('data/sine.nii.gz')

i,j,k = [ np.arange(0, spc.size[dim]) for dim in range(3) ]
i,j,k = np.meshgrid(j,i,k)
i,j,k = np.vstack((i.flatten(), j.flatten(), k.flatten()))

FLAT_FIELD = 60 * np.ones(spc.size.prod())
WM_FLAT_FIELD = 20 * np.ones_like(FLAT_FIELD)

SINE_FIELD = FLAT_FIELD + 6 * (np.sin(i/3) + np.sin(j/3) + np.sin(k/3))
WM_SINE_FIELD = WM_FLAT_FIELD + (2 * (np.sin(i/3) + np.sin(j/3) + np.sin(k/3)))

RAND_FIELD = FLAT_FIELD + 10 * np.random.normal(0, 0.4, FLAT_FIELD.shape)
WM_RAND_FIELD = WM_FLAT_FIELD + 5 * np.random.normal(0, 0.2, FLAT_FIELD.shape)

FLAT = FLAT_FIELD * pvs[:,0]
SINE = SINE_FIELD * pvs[:,0]
RAND = RAND_FIELD * pvs[:,0]

WM_FLAT = WM_FLAT_FIELD * pvs[:,1]
WM_SINE = WM_SINE_FIELD * pvs[:,1]
WM_RAND = WM_RAND_FIELD * pvs[:,1]

i,j,k = [ np.arange(0, spc.size[dim]) for dim in range(3) ]
interp_points = tob.utils.affineTransformPoints(mids.points, spc.world2vox)
ii,jj,kk = np.meshgrid(j,i,k)
ijk = np.vstack((ii.flatten(), jj.flatten(), kk.flatten()))
SSINE = interpn((i,j,k), SINE_FIELD.reshape(spc.size), interp_points)
SFLAT = interpn((i,j,k), FLAT_FIELD.reshape(spc.size), interp_points)
SRAND = interpn((i,j,k), RAND_FIELD.reshape(spc.size), interp_points)

SHAPE = (1,3)

plotter = pyvista.Plotter()
mesh = plots.to_polydata()
# plotter.add_mesh(mesh, scalars=SSINE, cmap='cwr', clim=[50,70], scalar_bar_args={'height':0.25})
plotter.add_mesh(mesh, scalars=SSINE)
plotter.add_scalar_bar()
plotter.show()
