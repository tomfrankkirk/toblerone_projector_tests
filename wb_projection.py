import numpy as np
from scipy import sparse
import tempfile
from subprocess import run 
from toblerone.projection import sparse_normalise 
import os.path as op 
import nibabel


loadnii = lambda f: nibabel.load(f).get_fdata().reshape(-1)
loadfunc = lambda f: nibabel.load(f).darrays[0].data

def load_wb_v2n_matrix(fname, surf, spc):
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

    gm_pv = wb_mat.sum(0).A.flatten()
    gm_pv = gm_pv / gm_pv.max()
    n2v_wb_mat = sparse_normalise(wb_mat, 0)
    n2v_wb_mat.data *= np.take(gm_pv, n2v_wb_mat.indices)
    n2v_wb_mat = n2v_wb_mat.T
    return n2v_wb_mat 