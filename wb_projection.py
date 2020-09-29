import numpy as np
from scipy import sparse
import tempfile
from subprocess import run 
from toblerone.utils import sparse_normalise 
import os.path as op 
import nibabel

loadnii = lambda f: nibabel.load(f).get_fdata().reshape(-1)
loadfunc = lambda f: nibabel.load(f).darrays[0].data

def load_wb_v2n_matrix(fname, surf, spc):
    """
    Read in wb_command voxel to surface projection weights from text file. 

    Args: 
        fname: path to text-like file containing weights.
        surf: Surface object, for which weights were produced.
        spc: ImageSpace object, for which weights were produced. 

    Returns: 
        scipy sparse DoK matrix, sized (vertices x voxels)
    """
    v2n_wb = sparse.dok_matrix((surf.points.shape[0], spc.size.prod()), dtype=np.float32)
    with open(fname, 'r') as f: 
        for idx,line in enumerate(f.readlines()):
            chunks = line.split(', ')
            vtx = int(chunks[0])
            n_vox = int(chunks[1]) 
            if n_vox: 
                for vidx in range(n_vox):
                    a,b,c,w = chunks[2 + 4*vidx : 2 + 4*(vidx+1)]
                    vox = np.ravel_multi_index((int(a),int(b),int(c)), spc.size)
                    v2n_wb[vtx,vox] = float(w)
                assert len(chunks) == (2+4*(vidx+1))
         
    if not v2n_wb.nnz: 
        raise RuntimeError("Empty matrix returned")
    return v2n_wb.tocsr()

def wb_v2n_method(vdata, insurf, midsurf, outsurf, spc, factor):
    """
    Wrapper to run wb_command volume to surface projection. 

    Args: 
        vdata: volumetric data array.
        insurf: path to white surface.
        midsurf: path to midsurface.
        outsurf: path to pial surface.
        spc: ImageSpace object, space to project from.
        factor: voxel subdivision factor.

    Returns: 
        tuple (weights, func). weights is an array of strings 
            for the saved weights, func is an array of projected
            output.
    """
        
    with tempfile.TemporaryDirectory() as d: 
        nii = op.join(d, 'v.nii.gz')
        spc.save_image(vdata, nii)
        func = op.join(d, 's.func.gii')
        mat = op.join(d, 'weights.txt')
        
        cmd = ("wb_command -volume-to-surface-mapping %s " % (nii) 
        + "%s %s -ribbon-constrained %s %s " % (midsurf, func, insurf, outsurf)
        + "-voxel-subdiv %d -output-weights-text %s" % (factor, mat))
        run(cmd, shell=True)
        with open(mat, 'r') as f:
            weights = f.readlines()
        func = loadfunc(func)

    return weights, func 

def wb_n2v_method(wb_mat):
    """
    Produce the inverse of wb_commands volume to surface projection.

    Args:
        wb_mat: scipy sparse matrix for volume to surface. 

    Returns:
        scipy sparse matrix for surface to volume. 
    """
        
    gm_pv = wb_mat.sum(0).A.flatten()
    gm_pv = gm_pv / gm_pv.max()
    n2v_wb_mat = sparse_normalise(wb_mat, 0)
    n2v_wb_mat.data *= np.take(gm_pv, n2v_wb_mat.indices)
    n2v_wb_mat = n2v_wb_mat.T
    return n2v_wb_mat 