import os.path as op 
import toblerone as tob 
import numpy as np 
import nibabel 
import pickle 
from scipy import sparse
from toblerone import projection


data_path = 'svb_data'
spc_path = op.join(data_path, 'oxasl_run1_scale1', 'calib.nii.gz')
s2r_path = op.join(data_path, 'oxasl_run1_scale1', 's2asl_world.mat')
t1_path = op.join(data_path, 'oxasl_run1_scale1', 'brain.nii.gz')
pv_path = op.join(data_path, 'pvs.nii.gz')

def index_surfs(): 
    spc = tob.ImageSpace(spc_path)
    s2r = np.loadtxt(s2r_path)

    for surf in ['white', 'pial']:

        s = tob.Surface(op.join(data_path, 'fs_30k', 'lh.%s.surf.gii' % surf))
        s.index_on(spc, s2r)
        with open(op.join(data_path, 'fs_30k', 'lh.%s_idx.pkl' % surf), 'wb') as f: 
            pickle.dump(s, f)

def make_pvs():
    spc = tob.ImageSpace(spc_path)
    s2r = np.loadtxt(s2r_path)

    surfs = [] 
    for surf in ['white', 'pial']:
        s = op.join(data_path, 'fs_30k', 'lh.%s.surf.gii' % surf)
        surfs.append(s)

    hemi = tob.classes.Hemisphere(*surfs, 'L')
    pvs, _ = tob.estimators._cortex(hemi, spc, s2r, np.ceil(spc.vox_size).astype(np.int8), 7, False)
    spc.save_image(pvs, pv_path)

def param2vox_projection(params, insurf, outsurf, pvs, subcortmask, spc):
    
    assert pvs.shape[0] == spc.size.prod()

    # Matrix is sized voxels x parameters
    p2v = sparse.dok_matrix((spc.size.prod, params.shape[0]), dtype=np.float32)

    # For each voxel, what parameters contribute, and in what weight?
    s2v_mat = projection.surf2vol_weights(insurf, outsurf, spc)

    to_process = np.union1d(np.flatnonzero((s2v_mat.sum(1) > 0).A), subcortmask)
    shift = insurf.points.shape[0]
    pv_weights = pvs[:,0:2] / pvs[:,0:2].sum(1)

    # For each voxel, we want to take GM x s2v mat, and WM x the voxel vertex, then normalise
    for vidx in to_process:
        sweights = s2v_mat[vidx,:]
        p2v[vidx,sweights.inds] = pv_weights[vidx,0] * sweights.data 
        p2v[vidx,shift+vidx] = pv_weights[vidx,1]

    # Normalise the whole lot?
    return p2v

if __name__ == "__main__":

    # index_surfs()
    # make_pvs()


    spc = toblerone.ImageSpace(spc_path)
    pvs = nibabel.load(op.join(data_path, 'pvs.nii.gz')).get_fdata().reshape(-1,3)
    surfs = [] 
    for surf in ['white', 'pial']: 
        with open(op.join(data_path, 'fs_30k', 'lh.%s_idx.pkl' % surf), 'rb') as f: 
            surfs.append(pickle.load(f)) 

    insurf, outsurf = surfs

    # Assemble the vertices that estimation is performed on: all surface nodes, and wm voxel centres 
    wmmask = np.flatnonzero(pvs[:,1] > 0)
    sdata = np.ones(surf.points.shape[0])
    vdata = np.ones(wmmask.size)

    params = np.concatenate(sdata, vdata)

    x = param2vox_projection(params, insurf, outsurf, pvs, wmmask, spc)

