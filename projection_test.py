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
import copy 
from scipy.interpolate import interpn

INROOT = '/mnt/hgfs/Data/toblerone_evaluation_data/HCP_retest/test'
OUTROOT = '/mnt/hgfs/Data/projection_test'
N_SUBS = 25
SIDES = ['L', 'R']
REF = op.join(OUTROOT, 'ref_2.2.nii.gz')
loadnii = lambda f: nibabel.load(f).get_fdata().reshape(-1)
loadfunc = lambda f: nibabel.load(f).darrays[0].data
FACTOR = 5
EXPERIMENTS = ['naive', 'edge_corr', 'wm_cbf']
TESTS = ['flat', 'sine', 'rand']
REFRESH = True 

def savefunc(data, fname):
    g = nibabel.GiftiImage()
    g.add_gifti_data_array(nibabel.gifti.GiftiDataArray(data))
    nibabel.save(g, fname)

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

def test_subject(data_dict, surf_dict, subid, exp, surf_paths, ref_path, factor):

    subdir = op.join(OUTROOT, exp, subid)

    # Prepare objects for calls 
    spc = classes.ImageSpace(ref_path)
    ins, mids, outs = surf_paths
    savesurf = tob.Surface(mids)
    hemi = classes.Hemisphere(ins, outs, 'L')
    hemi.PVs = pvs 
    hemi.apply_transform(spc.world2vox)

    # Toblerone's projection matrices 
    edge_corr = (exp != 'naive')
    if exp == 'naive':
        s2v_path = op.join(OUTROOT, subdir, 's2v_tob_matrix.npz')
        v2s_path = op.join(OUTROOT, subdir, 'v2s_tob_matrix.npz')
    else: 
        s2v_path = op.join(OUTROOT, 'edge_corr', subid, 's2v_tob_matrix.npz')
        v2s_path = op.join(OUTROOT, 'edge_corr', subid, 'v2s_tob_matrix.npz')

    if not op.exists(s2v_path):
        s2v_tob_mat = node2voxel_weights(spc, hemi, None, factor)
        sparse.save_npz(s2v_path, s2v_tob_mat)
    if not op.exists(v2s_path):
        v2s_tob_mat = voxel2nodes_weights(spc, hemi, None, factor, edge_corr)
        sparse.save_npz(v2s_path, v2s_tob_mat)
    v2s_tob_mat = sparse.load_npz(v2s_path)
    s2v_tob_mat = sparse.load_npz(s2v_path)

    if exp == 'naive':
        v2s_tob_mat = v2s_tob_mat[:savesurf.points.shape[0],:] 
        s2v_tob_mat = s2v_tob_mat[:,:savesurf.points.shape[0]]
        

    # Prepare WB_command's matrices. It does not change according to the different experiments
    # so load it from the naive experiment directory. 
    naive_dir = op.join(OUTROOT, 'naive', subid)
    weights_path = op.join(naive_dir, 'wb_weights.txt')
    wb_mat_path = op.join(naive_dir, 'wb_mat.npz')
    if not op.exists(wb_mat_path):
        weights, truth = wb_v2n_method(data_dict['flat'], ins, mids, outs, spc, 5)
        with open(weights_path, 'w') as f: 
            f.writelines(weights)

        wb_mat = load_wb_v2n_matrix(weights_path, savesurf, spc)
        sparse.save_npz(wb_mat_path, wb_mat)

    wb_mat = sparse.load_npz(wb_mat_path)
    v2s_wb_mat = projection.sparse_normalise(wb_mat, 1)
    gm_pv = wb_mat.sum(0).A.flatten()
    gm_pv = gm_pv / gm_pv.max()
    s2v_wb_mat = projection.sparse_normalise(copy.deepcopy(wb_mat), 0)
    s2v_wb_mat.data *= np.take(gm_pv, s2v_wb_mat.indices)
    s2v_wb_mat = s2v_wb_mat.T

    for key in TESTS: 
        print(subid, exp, key)
        pathroot = op.join(subdir, key)

        metric = surf_dict[key]
        metric_padded = metric 
        if exp == 'wm_cbf':
            vol = data_dict[key] + data_dict['wm_'+key]
            metric_padded = np.concatenate((metric, data_dict['wm_'+key]))
        else: 
            vol = data_dict[key]
            if exp == 'edge_corr':
                metric_padded = np.concatenate((metric, np.zeros_like(vol)))

        # Toblerone outputs vol - surf - vol 
        path = pathroot + '_v2s_tob.func.gii'
        if not op.exists(path) or REFRESH:
            v2s_tob = v2s_tob_mat.dot(vol)
            savefunc(v2s_tob, path)
        v2s_tob = loadfunc(path)

        path = pathroot + '_v2s2v_tob.nii.gz'
        if not op.exists(path) or REFRESH:
            v2s2v_tob = s2v_tob_mat.dot(v2s_tob)
            spc.save_image(v2s2v_tob, path)

        path = pathroot + '_v2s2v_tob_sp.nii.gz'
        if not op.exists(path) or REFRESH:
            v2s2v_tob_sp = linalg.lsqr(v2s_tob_mat, v2s_tob)[0]
            spc.save_image(v2s2v_tob_sp, pathroot + '_v2s2v_tob_sp.nii.gz')

        # WB command outputs vol - surf - vol 
        path = pathroot + '_v2s_wb.func.gii'
        if not op.exists(path) or REFRESH:
            v2s_wb = v2s_wb_mat.dot(vol)
            savefunc(v2s_wb, path)
        v2s_wb = loadfunc(path)
        
        path = pathroot + '_v2s2v_wb.nii.gz'
        if not op.exists(path) or REFRESH:
            v2s2v_wb = s2v_wb_mat.dot(v2s_wb)
            spc.save_image(v2s2v_wb, path)

        path = pathroot + '_v2s2v_wb_sp.nii.gz'
        if not op.exists(path) or REFRESH:
            v2s2v_wb_sp = linalg.lsqr(v2s_wb_mat, v2s_wb)[0]
            spc.save_image(v2s2v_wb_sp, path)

        # Toblerone outputs surf - vol - surf 
        path = pathroot + '_s2v_tob.nii.gz'
        if not op.exists(path) or REFRESH:
            s2v_tob = s2v_tob_mat.dot(metric_padded)
            spc.save_image(s2v_tob, path)
        s2v_tob = loadnii(path)

        path = pathroot + '_s2v2s_tob'
        if not op.exists(path) or REFRESH:
            s2v2s_tob = v2s_tob_mat.dot(s2v_tob)
            savefunc(s2v2s_tob, path)

        path = pathroot + '_s2v2s_tob_sp'
        if not op.exists(path) or REFRESH:
            s2v2s_tob_sp = linalg.lsqr(s2v_tob_mat, s2v_tob)[0]
            savefunc(s2v2s_tob_sp, path)

        # WB_commnd outputs surf - vol - surf 
        path = pathroot + '_s2v_wb.nii.gz'
        if not op.exists(path) or REFRESH:
            s2v_wb = s2v_wb_mat.dot(metric)
            spc.save_image(s2v_wb, path)
        s2v_wb = loadnii(path)

        path = pathroot + '_s2v2s_wb'
        if not op.exists(path) or REFRESH:
            s2v2s_wb = v2s_wb_mat.dot(s2v_wb)
            savefunc(s2v2s_wb, path)

        path = pathroot + '_s2v2s_wb_sp'
        if not op.exists(path) or REFRESH:
            s2v2s_wb_sp = linalg.lsqr(s2v_wb_mat, s2v_wb)[0]
            savefunc(s2v2s_wb_sp, path)


def make_surf_truths(spc, surf):
    i,j,k = [ np.arange(0, spc.size[dim]) for dim in range(3) ]
    interp_points = tob.utils.affineTransformPoints(surf.points, spc.world2vox)
    ii,jj,kk = np.meshgrid(j,i,k)
    ijk = np.vstack((ii.flatten(), jj.flatten(), kk.flatten()))
    period = 3
    FLAT_FIELD = 60 * np.ones(spc.size)
    SINE_FIELD = FLAT_FIELD + (10 * np.sin(ijk.sum(0)/period)).reshape(spc.size)
    RAND_FIELD = FLAT_FIELD + np.random.normal(0, 5, spc.size)
    SINE = interpn((i,j,k), SINE_FIELD, interp_points)
    FLAT = interpn((i,j,k), FLAT_FIELD, interp_points)
    RAND = interpn((i,j,k), RAND_FIELD, interp_points)
    return { 'flat': FLAT, 'rand': RAND, 'sine': SINE }


def make_volume_truths(pvs, spc):

    FLAT_FIELD = 60 * np.ones(spc.size.prod())
    WM_FLAT_FIELD = 20 * np.ones_like(FLAT_FIELD)
    i,j,k = [ np.arange(0, spc.size[dim]) for dim in range(3) ]
    i,j,k = np.meshgrid(j,i,k)
    ijk = np.vstack((i.flatten(), j.flatten(), k.flatten()))
    SINE_FIELD = 10 * np.sin(ijk.sum(0)/3)
    WM_SINE_FIELD = SINE_FIELD/2

    FLAT = FLAT_FIELD * pvs[:,0]
    MASK = (FLAT > 0)
    SINE = (FLAT + SINE_FIELD) * pvs[:,0]
    RAND = 1 * FLAT
    RAND[MASK] *= np.random.normal(1, 0.1, MASK.sum())
    RAND[RAND < 0] = 0 

    pvs[~MASK,1] = 0 
    WM_FLAT = WM_FLAT_FIELD * pvs[:,1]
    WM_SINE = (WM_FLAT_FIELD + WM_SINE_FIELD) * pvs[:,1]
    WM_RAND = 1 * WM_FLAT
    WM_RAND[MASK] *= np.random.normal(1, 0.05, MASK.sum())
    RAND[RAND < 0] = 0 

    return { 'flat': FLAT, 'sine': SINE, 'rand': RAND, 
              'wm_flat': WM_FLAT, 'wm_sine': WM_SINE, 'wm_rand': WM_RAND }

if __name__ == "__main__":

    ref_spc = classes.ImageSpace(REF)
    supersample = np.ceil(ref_spc.vox_size).astype(np.int8)

    for exp in EXPERIMENTS:
    
        for idx, subid in enumerate(SUBIDS()[:N_SUBS]):

            # Prepare output directory
            outdir = op.join(OUTROOT, exp, subid)
            os.makedirs(outdir, exist_ok=True)

            # Load the L/R hemispheres, estimate PVs for it and save 
            L_surfs = hemi_paths(subid, 'L')
            L_hemi = classes.Hemisphere(L_surfs[0], L_surfs[2], 'L')
            L_pvs_path = op.join(OUTROOT, 'naive', subid, 'tob_L_pvs.nii.gz')
            if not op.exists(L_pvs_path):
                pvs, _ = estimators._cortex(L_hemi, ref_spc, np.eye(4), supersample, 16, False)
                ref_spc.save_image(pvs, L_pvs_path)
            pvs = nibabel.load(L_pvs_path).get_fdata().reshape(-1,3)

            # Make the ground truths
            midsurf = tob.Surface(L_surfs[1])
            vol_truths = make_volume_truths(pvs, ref_spc)
            surf_truths = make_surf_truths(ref_spc, midsurf)
            for key in TESTS:
                vol = vol_truths[key]
                path = op.join(outdir, '%s.nii.gz' % key)
                if not op.exists(path) or REFRESH:
                    ref_spc.save_image(vol, path)
                vol_truths[key] = loadnii(path)
                path = op.join(outdir, key+'.func.gii')
                if not op.exists(path):
                    midsurf.save_metric(surf_truths[key], path)
                surf_truths[key] = loadfunc(path)    

            # Run the test suite
            try: 
                test_subject(vol_truths, surf_truths, subid, exp, L_surfs, REF, FACTOR)
            except Exception as e: 
                print(e)
                continue 