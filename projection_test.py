import toblerone as tob
import os.path as op 
import numpy as np
import nibabel
import os 
import glob
from toblerone import estimators, classes, projection
from tob_projectors import node2voxel_weights, voxel2nodes_weights
import pickle
from scipy import sparse
from scipy.sparse import linalg
from subprocess import run 
import tempfile
from wb_projection import * 
from pdb import set_trace

INROOT = '/mnt/hgfs/Data/toblerone_evaluation_data/HCP_retest/test'
OUTROOT = '/mnt/hgfs/Data/projection_test'
N_SUBS = 50
SIDES = ['L', 'R']
REF = op.join(OUTROOT, 'ref_2.2.nii.gz')
loadnii = lambda f: nibabel.load(f).get_fdata().reshape(-1)
loadfunc = lambda f: nibabel.load(f).darrays[0].data
FACTOR = 5

def SUBIDS():
    dirs = glob.glob(op.join(INROOT, '*'))
    if not dirs: 
        raise RuntimeError("did not find any subject directories at path")
    subdirs = [ d for d in dirs if (op.isdir(d) and ((op.split(d)[1] != 'refs') 
        and (op.splitext(d)[1] != '.zip') and (op.split(d)[1] != 'fast'))) ]

    subids = [ op.split(d)[1] for d in subdirs ] 
    return subids

def hemi_paths(subid, side):

    d = op.join(INROOT, subid, 'T1w/fsaverage_LR32k')
    paths = [ 
        op.join(d, '%s.%s.%s.32k_fs_LR.surf.gii') % (subid, side, surf)
        for surf in ['white', 'midthickness', 'pial']
    ]
    assert all([ op.exists(p) for p in paths ])
    return paths

def vsv_subject(data_dict, subdir, surf_paths, ref_path, factor):

    # Prepare objects for calls 
    spc = classes.ImageSpace(ref_path)
    ins, mids, outs = surf_paths
    savesurf = tob.Surface(mids)
    hemi = classes.Hemisphere(ins, outs, 'L')
    hemi.PVs = pvs 
    hemi.apply_transform(spc.world2vox)

    # Toblerone's projection matrices 
    n2v_path = op.join(subdir, 's2v_tob_matrix.npz')
    v2n_path = op.join(subdir, 'v2s_tob_matrix.npz')
    if not op.exists(n2v_path):
        n2v_tob_mat = node2voxel_weights(spc, hemi, None, factor)
        sparse.save_npz(n2v_path, n2v_tob_mat)
    if not op.exists(v2n_path):
        v2n_tob_mat = voxel2nodes_weights(spc, hemi, None, factor)
        sparse.save_npz(v2n_path, v2n_tob_mat)
    v2n_tob_mat = sparse.load_npz(v2n_path)
    n2v_tob_mat = sparse.load_npz(n2v_path)

    for key, vol in data_dict.items(): 
        print(key)
        pathroot = op.join(subdir, key)

        # Toblerone solution 
        v2s_tob = v2n_tob_mat.dot(vol)
        savesurf.save_metric(v2s_tob[:savesurf.points.shape[0]], pathroot + '_v2s_tob.func.gii')

        v2s2v_tob = n2v_tob_mat.dot(v2s_tob)
        spc.save_image(v2s2v_tob, pathroot + '_v2s2v_tob.nii.gz')

        path = pathroot + '_v2s2v_tob_sp.nii.gz'
        if not op.exists(path):
            v2s2v_tob_sp = linalg.lsqr(v2n_tob_mat, v2s_tob)[0]
            spc.save_image(v2s2v_tob_sp, path)

        # WB solution 
        v2n_wb_mat_path = op.join(outdir, 'v2s_wb_matrix.npz')
        path = op.join(outdir, pathroot + '_v2s_wb.func.gii')
        if (not op.exists(v2n_wb_mat_path)) or (not op.exists(path)):
            v2n_wb_mat, v2s_wb = wb_v2n_method(vol, ins, mids, outs, spc, factor)
            sparse.save_npz(v2n_wb_mat_path, v2n_wb_mat)
            savesurf.save_metric(v2s_wb, path)

        path = pathroot + '_v2s2v_wb.nii.gz'
        if not op.exists(path):
            v2s2v_wb = wb_n2v_method(v2s_wb, ins, mids, outs, ref_path, factor)    
            spc.save_image(v2s2v_wb, path)

        v2n_wb_mat = projection.sparse_normalise(v2n_wb_mat, 1)
        v2s2v_wb_sp = linalg.lsqr(v2n_wb_mat, v2s_wb)[0]
        spc.save_image(v2s2v_wb_sp, pathroot + '_v2s2v_wb_sp.nii.gz')




def make_ground_truths(pvs, spc):

    FLAT_FIELD = 60 * np.ones(spc.size.prod())
    i,j,k = [ np.arange(0, spc.size[dim]) for dim in range(3) ]
    i,j,k = np.meshgrid(j,i,k)
    ijk = np.vstack((i.flatten(), j.flatten(), k.flatten()))
    x,y,z = tob.utils.affineTransformPoints(ijk.T, spc.vox2world).T
    SINE_FIELD = 10 * np.sin(x/3) + np.cos(y/3) + np.sin(z/3)

    FLAT = FLAT_FIELD * pvs[:,0]
    SINE = FLAT + (SINE_FIELD * pvs[:,0])
    MASK = (FLAT > 0)
    RAND = 1 * FLAT
    RAND[MASK] *= np.random.normal(1, 0.1, MASK.sum())
    RAND[RAND < 0] = 0 
    return { 'flat': FLAT, 'sine': SINE, 'rand': RAND }

if __name__ == "__main__":

    ref_spc = classes.ImageSpace(REF)
    supersample = np.ceil(ref_spc.vox_size).astype(np.int8)

    all_results = []
    
    for idx, subid in enumerate(SUBIDS()[:N_SUBS]):
        print(idx, subid)

        # Prepare output directory
        outdir = op.join(OUTROOT, subid)
        os.makedirs(outdir, exist_ok=True)

        # Load the L/R hemispheres, estimate PVs for it and save 
        L_surfs = hemi_paths(subid, 'L')
        L_hemi = classes.Hemisphere(L_surfs[0], L_surfs[2], 'L')
        L_pvs_path = op.join(outdir, 'tob_L_pvs.nii.gz')
        if not op.exists(L_pvs_path):
            pvs, _ = estimators._cortex(L_hemi, ref_spc, np.eye(4), supersample, 16, False)
            ref_spc.save_image(pvs, L_pvs_path)
        pvs = nibabel.load(L_pvs_path).get_fdata().reshape(-1,3)

        # Make the ground truths
        VTRUTHS = make_ground_truths(pvs, ref_spc)
        for key, vol in VTRUTHS.items():
            ref_spc.save_image(vol, op.join(outdir, '%s.nii.gz' % key))

        # Run the test suite
        vsv_subject(VTRUTHS, outdir, L_surfs, REF, FACTOR)
