import os.path as op 
import numpy as np
import nibabel
import os 
from toblerone import projection, classes, utils
import pickle
import gzip
from scipy import sparse
from scipy.sparse import linalg
from scipy.interpolate import interpn
import wb_projection
from pdb import set_trace

# Location where the 45 subject directories of the HCP test-retest dataset 
# are stored. Only the test component is used. Output will be stored in the 
# data subdirectory 
INROOT = '/mnt/hgfs/Data/toblerone_evaluation_data/HCP_retest/test'
OUTROOT = 'projected_output'
N_SUBS = 45
REF = op.join(OUTROOT, 'ref_2.2.nii.gz') # reference space for all tests 
REF_SPC = classes.ImageSpace(REF) # same again, as object 
FACTOR = 5  # voxel subdivision factor for both methods 
EXPERIMENTS = ['naive', 'edge_corr', 'wm_cbf'] # wb_cbf corresponds to 2-component signal 
SIGNALS = ['flat', 'sine', 'rand'] # signal distributions 
REFRESH = False # force recomputation of all data 

loadnii = lambda f: nibabel.load(f).get_fdata().reshape(-1)
loadfunc = lambda f: nibabel.load(f).darrays[0].data

def savefunc(data, fname):
    g = nibabel.GiftiImage()
    g.add_gifti_data_array(nibabel.gifti.GiftiDataArray(data.astype(np.float32)))
    nibabel.save(g, fname)

# extract all subject IDs from the HCP test directory 
def SUBIDS():
    dirs = os.listdir(INROOT)
    if not dirs: 
        raise RuntimeError("did not find any subject directories at path")

    return dirs

# Return the white, midthickness and pial surfaces for a HCP subject 
def hemi_paths(subid, side):
    d = op.join(INROOT, subid, 'T1w/fsaverage_LR32k')
    paths = [ 
        op.join(d, '%s.%s.%s.32k_fs_LR.surf.gii') % (subid, side, surf)
        for surf in ['white', 'midthickness', 'pial']
    ]
    assert all([ op.exists(p) for p in paths ])
    return paths

# Simulate a small area on a surface, within the mask (excludes areas of 
# low thickness)
def make_activation(for_surf, mask):
    at_idx = np.flatnonzero(mask)[np.random.randint(mask.sum())]
    metric = np.zeros(for_surf.points.shape[0])
    n1 = for_surf.tris[(for_surf.tris == at_idx).any(1),:].flatten()
    n2 = for_surf.tris[np.isin(for_surf.tris, n1).any(1),:].flatten()
    metric[n1] = 0.6
    metric[n2] = 0.8
    metric[at_idx] = 1 
    return metric 

# Produce the flat, sine and random signal fields from which ground
# truth is generated 
def make_fields(spc):        
    i,j,k = [ np.arange(0, spc.size[dim]) for dim in range(3) ]
    i,j,k = np.meshgrid(j,i,k)
    i,j,k = np.vstack((i.flatten(), j.flatten(), k.flatten()))
    FLAT_FIELD = 60 * np.ones(spc.size.prod())
    WM_FLAT_FIELD = 20 * np.ones_like(FLAT_FIELD)
    SINE_FIELD = FLAT_FIELD + 6 * (np.sin(i/3) + np.sin(j/3) + np.sin(k/3))
    WM_SINE_FIELD = WM_FLAT_FIELD + (2 * (np.sin(i/3) + np.sin(j/3) + np.sin(k/3)))
    RAND_FIELD = FLAT_FIELD + 10 * np.random.normal(0, 0.4, FLAT_FIELD.shape)
    WM_RAND_FIELD = WM_FLAT_FIELD + 4 * np.random.normal(0, 0.2, FLAT_FIELD.shape)    
    return { 'flat': FLAT_FIELD, 'wm_flat': WM_FLAT_FIELD, 'sine': SINE_FIELD, 
            'wm_sine': WM_SINE_FIELD, 'rand': RAND_FIELD, 'wm_rand': WM_RAND_FIELD } 

# Intersect a surface into the various fields and find vertex signal values
# via interpolation
def make_surf_truths(spc, surf):
    i,j,k = [ np.arange(0, spc.size[dim]) for dim in range(3) ]
    interp_points = utils.affineTransformPoints(surf.points, spc.world2vox)
    field_dict = make_fields(spc)
    SINE = interpn((i,j,k), field_dict['flat'].reshape(spc.size), interp_points)
    FLAT = interpn((i,j,k), field_dict['sine'].reshape(spc.size), interp_points)
    RAND = interpn((i,j,k), field_dict['rand'].reshape(spc.size), interp_points)
    return { 'flat': FLAT, 'rand': RAND, 'sine': SINE }

# Insert voxel-wise PVs into various fields and multiply corresponding values
# to get a volume truth with PVE 
def make_volume_truths(pvs, spc):
    field_dict = make_fields(spc)
    FLAT = field_dict['flat'] * pvs[:,0]
    SINE = field_dict['sine'] * pvs[:,0]
    RAND = field_dict['rand'] * pvs[:,0]
    WM_FLAT = field_dict['wm_flat'] * pvs[:,1]
    WM_SINE = field_dict['wm_sine'] * pvs[:,1]
    WM_RAND = field_dict['wm_rand'] * pvs[:,1]

    return { 'flat': FLAT, 'sine': SINE, 'rand': RAND, 
              'wm_flat': WM_FLAT, 'wm_sine': WM_SINE, 'wm_rand': WM_RAND }

# The main test function that is called on each subject. For a given experiment
# (eg, naive), it will run both methods on all three signal distributions (flat,
# sine etc). It also runs the local signal test (small activation peak)
def test_subject(data_dict, surf_dict, subid, exp, surf_paths):

    # Set up. Load reference space, midsurface (on which we save .func.gii
    # data) and subject output directory 
    ins, mids, outs = surf_paths
    savesurf = classes.Surface(mids)
    subdir = op.join(OUTROOT, exp, subid)

    # Load the toblerone projector (prepared before calling this function)
    tp_name = op.join(OUTROOT, 'projectors', '%s_tob.gz' % subid)
    with gzip.open(p_name, 'rb') as f: 
        projector = pickle.load(f)

    # Naive: volume-surface projection only, no edge correction
    if exp == 'naive':
        v2n_tob_mat = projector.vol2surf_matrix(False)
        n2v_tob_mat = projector.surf2vol_matrix()

    # Volume-surface, with edge correction
    elif exp == 'edge_corr':
        v2n_tob_mat = projector.vol2surf_matrix(True)
        n2v_tob_mat = projector.surf2vol_matrix()

    # Volume-node, with edge correction
    else:  
        v2n_tob_mat = projector.vol2node_matrix(True)
        n2v_tob_mat = projector.node2vol_matrix()
  
    # Load wb_command projection that we prepared earlier 
    wb_mat_path = op.join(OUTROOT, 'projectors', '%s_wb_mat.npz' % subid)
    wb_mat = sparse.load_npz(wb_mat_path)
    v2n_wb_mat = utils.sparse_normalise(wb_mat, 1)
    n2v_wb_mat = wb_projection.wb_n2v_method(wb_mat)
    
    # Local signal test (peak activation). This projection is performed using
    # the naive toblerone method (as this is directly comparable with existing
    # methods for projecting fMRI data)
    if exp == 'naive':
        print(subid, 'activation')
        smask = loadfunc(op.join(outdir, 'smask.func.gii'))
        metric = make_activation(savesurf, smask)
        savefunc(metric, op.join(subdir, 'activation.func.gii'))

        # Toblerone round trip
        tvol = n2v_tob_mat.dot(metric)
        tmetric = v2n_tob_mat.dot(tvol)
        REF_SPC.save_image(tvol, op.join(subdir, 'tob_activation.nii.gz'))
        savefunc(tmetric, op.join(subdir, 'tob_activation.func.gii'))

        # wb round trip 
        wbvol = n2v_wb_mat.dot(metric)
        wbmetric = v2n_wb_mat.dot(wbvol)
        REF_SPC.save_image(wbvol, op.join(subdir, 'wb_activation.nii.gz'))
        savefunc(wbmetric, op.join(subdir, 'wb_activation.func.gii'))

    # Global signal tests (flat, sine, rand)
    for key in SIGNALS: 
        print(subid, exp, key)
        pathroot = op.join(subdir, key)

        # Extract ground truths from the data dicts 
        metric = surf_dict[key]
        vol = data_dict[key]

        # If projecting to node space, then we need to pad out the metric with
        # data from the subcortex for toblerone. Use the WM CBF values from 
        # the ground truth (the subcortical nodes represent pure WM)
        metric_padded = metric 
        if exp == 'wm_cbf':
            vol += data_dict['wm_'+key]
            metric_padded = np.concatenate((metric, data_dict['wm_'+key]))


        # Toblerone outputs vol - surf - vol 
        path = pathroot + '_v2n_tob.func.gii'
        if not op.exists(path) or REFRESH:
            v2n_tob = v2n_tob_mat.dot(vol)
            savefunc(v2n_tob, path)
        v2n_tob = loadfunc(path)

        path = pathroot + '_v2n2v_tob.nii.gz'
        if not op.exists(path) or REFRESH:
            v2n2v_tob = n2v_tob_mat.dot(v2n_tob)
            REF_SPC.save_image(v2n2v_tob, path)

        path = pathroot + '_v2n2v_tob_sp.nii.gz'
        if not op.exists(path) or REFRESH:
            v2n2v_tob_sp = linalg.lsqr(v2n_tob_mat, v2n_tob)[0]
            REF_SPC.save_image(v2n2v_tob_sp, path)

        # WB command outputs vol - surf - vol 
        path = pathroot + '_v2n_wb.func.gii'
        if not op.exists(path) or REFRESH:
            v2n_wb = v2n_wb_mat.dot(vol)
            savefunc(v2n_wb, path)
        v2n_wb = loadfunc(path)
        
        path = pathroot + '_v2n2v_wb.nii.gz'
        if not op.exists(path) or REFRESH:
            v2n2v_wb = n2v_wb_mat.dot(v2n_wb)
            REF_SPC.save_image(v2n2v_wb, path)

        path = pathroot + '_v2n2v_wb_sp.nii.gz'
        if not op.exists(path) or REFRESH:
            v2n2v_wb_sp = linalg.lsqr(v2n_wb_mat, v2n_wb)[0]
            REF_SPC.save_image(v2n2v_wb_sp, path)

        # Toblerone outputs surf - vol - surf 
        path = pathroot + '_n2v_tob.nii.gz'
        if not op.exists(path) or REFRESH:
            n2v_tob = n2v_tob_mat.dot(metric_padded)
            REF_SPC.save_image(n2v_tob, path)
        n2v_tob = loadnii(path)

        path = pathroot + '_n2v2n_tob.func.gii'
        if not op.exists(path) or REFRESH:
            s2v2n_tob = v2n_tob_mat.dot(n2v_tob)
            savefunc(s2v2n_tob, path)

        path = pathroot + '_n2v2n_tob_sp.func.gii'
        if not op.exists(path) or REFRESH:
            s2v2n_tob_sp = linalg.lsqr(n2v_tob_mat, n2v_tob)[0]
            savefunc(s2v2n_tob_sp, path)

        # WB_commnd outputs surf - vol - surf 
        path = pathroot + '_n2v_wb.nii.gz'
        if not op.exists(path) or REFRESH:
            n2v_wb = n2v_wb_mat.dot(metric)
            REF_SPC.save_image(n2v_wb, path)
        n2v_wb = loadnii(path)

        path = pathroot + '_n2v2n_wb.func.gii'
        if not op.exists(path) or REFRESH:
            s2v2n_wb = v2n_wb_mat.dot(n2v_wb)
            savefunc(s2v2n_wb, path)

        path = pathroot + '_n2v2n_wb_sp.func.gii'
        if not op.exists(path) or REFRESH:
            s2v2n_wb_sp = linalg.lsqr(n2v_wb_mat, n2v_wb)[0]
            savefunc(s2v2n_wb_sp, path)



if __name__ == "__main__":
    os.makedirs(op.join(OUTROOT, 'projectors'), exist_ok=True)
    for exp in EXPERIMENTS:
        for idx, subid in enumerate(SUBIDS()[:N_SUBS]):

            # Prepare output directory
            outdir = op.join(OUTROOT, exp, subid)
            os.makedirs(outdir, exist_ok=True)
            L_surfs = hemi_paths(subid, 'L')

            # Prepare L hemisphere toblerone projection (this also incorporates)
            # the PV estimation step, which is needed to produce the ground truths
            p_name = op.join(OUTROOT, 'projectors', '%s_tob.gz' % subid)
            if not op.exists(p_name):
                L_hemi = classes.Hemisphere(L_surfs[0], L_surfs[2], 'L')
                proj = projection.Projector(L_hemi, REF_SPC, FACTOR, cores=14)

                with gzip.open(p_name, 'wb') as f: 
                    f.write(pickle.dumps(proj))
            else: 
                with gzip.open(p_name, 'rb') as f: 
                    proj = pickle.load(f)

            # Make the ground truths for the global signal tests. We only save them
            # if they do not already exist for this subject (don't want to change 
            # nature of some of them)
            midsurf = classes.Surface(L_surfs[1])
            vol_truths = make_volume_truths(proj.flat_pvs(), REF_SPC)
            surf_truths = make_surf_truths(REF_SPC, midsurf)

            # Saving the ground truths if they don't exist, otherwise load the existing ones
            for key in SIGNALS:
                vol = vol_truths[key]
                path = op.join(outdir, '%s.nii.gz' % key)
                if not op.exists(path) or REFRESH:
                    REF_SPC.save_image(vol, path)
                    if exp == 'wm_cbf':
                        REF_SPC.save_image(vol_truths['wm_'+key], op.join(outdir, 'wm_'+key+'.nii.gz'))
                else:
                    vol_truths[key] = loadnii(path)
                    if exp != 'wm_cbf': 
                        vol_truths.pop('wm_'+key)
                    else: 
                        vol_truths['wm_'+key] = loadnii(op.join(outdir, 'wm_'+key+'.nii.gz'))

                path = op.join(outdir, key+'.func.gii')
                if not op.exists(path) or REFRESH:
                    midsurf.save_metric(surf_truths[key], path)
                surf_truths[key] = loadfunc(path)    

            # Prepare WB_command's matrices. It does not change according to the different experiments
            # Do a dummy run on some volume data and save --output-weights as a text file 
            weights_path = op.join(OUTROOT, 'projectors', '%s_wb_weights.txt' % subid)
            wb_mat_path = op.join(OUTROOT, 'projectors', '%s_wb_mat.npz' % subid)
            if not op.exists(weights_path):
                weights, _ = wb_projection.wb_v2n_method(vol_truths['flat'], 
                    *L_surfs, REF_SPC, FACTOR)
                with open(weights_path, 'w') as f: 
                    f.writelines(weights)

            # Load in the --output-weights from above and read them into a sparse mat 
            if not op.exists(wb_mat_path):
                wb_mat = wb_projection.load_wb_v2n_matrix(weights_path, midsurf, REF_SPC)
                sparse.save_npz(wb_mat_path, wb_mat)

            # Prepare the surface mask which excludes areas with thickness < 1mm 
            smask_path = op.join(outdir, 'smask.func.gii')
            if not op.exists(smask_path):
                insurf = classes.Surface(L_surfs[0])
                outsurf = classes.Surface(L_surfs[2])
                thickness = np.linalg.norm(outsurf.points - insurf.points, ord=2, axis=1)
                smask = (thickness > 1)
                savefunc(smask, smask_path)

            # Run the test suite
            # try: 
            #     # pass
            test_subject(vol_truths, surf_truths, subid, exp, L_surfs)
            # except Exception as e: 
            #     print(e)
            #     continue 