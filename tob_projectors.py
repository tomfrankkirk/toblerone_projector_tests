import numpy as np 
from scipy import sparse
import multiprocessing as mp 
from toblerone import projection

# FIXME:add a warning if the surfaces are not enclosed in the FoV?

def node2voxel_weights(spc, L_hemi=None, R_hemi=None, 
                       factor=10, cores=mp.cpu_count()):
    """
    Projection weights from nodes to voxels, represented as a sparse matrix. 
    NB any transformation from the native space of the surfaces to the space
    of the voxel grid must be performed PRIOR to this function. 
    
    Args: 
        hemispheres: 
        factor: int, voxel subdivision factor (default 10)
        cores: number of CPU cores to use, default max 
    
    Returns: 
        sparse matrix of size (voxels x nodes), where the columns are
            arranged from L->R: L cortex, R cortex, subcortical nodes
    """
    
    if (L_hemi is None) or (R_hemi is None): 
        if (L_hemi is not None): hemi = L_hemi 
        if (R_hemi is not None): hemi = R_hemi 
        assert hemi is not None 
        
        pvs = hemi.PVs
        s2v_mat = projection.surf2vol_weights(*hemi.surfs, spc, factor, cores).tocsc()
        s2v_mat.data *= np.take(pvs[:,0], s2v_mat.indices)
   
    else: 

        # Combine PV estimates from each hemisphere into single map 
        pvs = np.zeros((spc.size.prod(), 3), dtype=np.float32)
        pvs[:,0] = np.minimum(1.0, L_hemi.PVs[:,0] + R_hemi.PVs[:,0])
        pvs[:,1] = np.minimum(1.0 - pvs[:,0], L_hemi.PVs[:,1] + R_hemi.PVs[:,1])
        pvs[:,2] = 1.0 - np.sum(pvs[:,0:2], axis=1)

        # Projection for surface nodes to voxels for each hemisphere
        L_s2v_mat = projection.surf2vol_weights(*L_hemi.surfs, spc, factor, cores).tocsc()
        R_s2v_mat = projection.surf2vol_weights(*R_hemi.surfs, spc, factor, cores).tocsc()

        # GM PV can be shared between both hemispheres, so rescale each row of
        # the s2v matrices by the proportion of all voxel-wise GM that belongs
        # to that hemisphere (eg, the GM could be shared 80:20 between the two)
        LR_GM_sum = L_hemi.PVs[:,0] + R_hemi.PVs[:,0]
        LR_GM_sum[LR_GM_sum == 0] = 1 
        L_GM_weights = L_hemi.PVs[:,0] / LR_GM_sum
        R_GM_weights = R_hemi.PVs[:,0] / LR_GM_sum
        L_s2v_mat.data *= np.take(L_GM_weights, L_s2v_mat.indices)
        R_s2v_mat.data *= np.take(R_GM_weights, R_s2v_mat.indices)

        # Stack the two matrices horizontally, and rescale each row again by the 
        # the total GM PV for that voxel (eg, could be 60% GM and 40% WM)
        s2v_mat = sparse.hstack((L_s2v_mat, R_s2v_mat), format="csc")
        s2v_mat.data *= np.take(pvs[:,0], s2v_mat.indices)
    
    # 1:1 mapping between nodes that represent voxels, and their respective voxels
    # BUT their weights are rescaled by their WM fractions. This ensures that voxels
    # that contain both WM and GM have total weights that sum to unity 
    v2v = sparse.dia_matrix((pvs[:,1], 0), 
                shape=(spc.size.prod(), spc.size.prod()), dtype=np.float32)
    
    # Stack into a matrix sized (voxels x nodes)
    n2v_mat = sparse.hstack((s2v_mat, v2v), format="csr")
    
    # Do some sanity checks here. Per-vox sum should equal the brain PV 
#     vox_weights = n2v_mat.sum(1).A.flatten()
#     assert (np.abs(vox_weights[vox_weights > 0] - 1) < 1e-6).all()
        
    return n2v_mat

def voxel2nodes_weights(spc, L_hemi=None, R_hemi=None, 
                        factor=10, edge_correct=True, cores=mp.cpu_count()):
    
    if (L_hemi is None) or (R_hemi is None): 
        if (L_hemi is not None): hemi = L_hemi 
        if (R_hemi is not None): hemi = R_hemi 
        assert hemi is not None 
        
        pvs = hemi.PVs
        v2s_mat = projection.vol2surf_weights(*hemi.surfs, spc, factor, cores).tocsr()
   
    else: 
        
        # Combine PV estimates from each hemisphere into single map 
        pvs = np.zeros((spc.size.prod(), 3), dtype=np.float32)
        pvs[:,0] = np.minimum(1.0, L_hemi.PVs[:,0] + R_hemi.PVs[:,0])
        pvs[:,1] = np.minimum(1.0 - pvs[:,0], L_hemi.PVs[:,1] + R_hemi.PVs[:,1])
        pvs[:,2] = 1.0 - np.sum(pvs[:,0:2], axis=1)
        
        # Projection from voxels to surface nodes 
        L_v2s_mat = projection.vol2surf_weights(*L_hemi.surfs, spc, factor, cores).tocsr()
        R_v2s_mat = projection.vol2surf_weights(*R_hemi.surfs, spc, factor, cores).tocsr()
        v2s_mat = sparse.vstack((L_v2s_mat, R_v2s_mat))
    
    # Correction for edge effects. For voxels that are partially CSF, upscale
    # their signal by 1 / (GM+WM PV) - this encodes the assumption that CSF
    # contributes no signal. Zero out the weight of voxels below a threshold
    if edge_correct: 
        brain_pv = pvs[:,:2].sum(1)
        brain = (brain_pv > 1e-6)
        upweight = np.zeros(brain_pv.shape, dtype=np.float32)
        upweight[brain] = 1 / brain_pv[brain]
        
        # Each column corresponds to a voxel. Multiply the values in each by 
        # their upweighting factor
        v2s_mat.data *= np.take(upweight, v2s_mat.indices)
    
    # Mapping of nodes that represent voxels is simply 1:1 
    v2v = sparse.eye(spc.size.prod(), dtype=np.float32)
    v2n_mat = sparse.vstack((v2s_mat, v2v), format="csr")
    
    # Sanity checks here 
    return v2n_mat


