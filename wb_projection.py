import numpy as np
from scipy import sparse
import tempfile
from subprocess import run 
import toblerone as tob 
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

# def wb_n2v_method(sdata, insurf, midsurf, outsurf, ref, factor):
    
#     with tempfile.TemporaryDirectory() as d: 
#         f = op.join(d, 'temp.nii.gz')   
#         gii = op.join(d, 's.func.gii')
#         g = nibabel.gifti.GiftiImage()
#         da = nibabel.gifti.GiftiDataArray(sdata.astype(np.float32))
#         g.add_gifti_data_array(da)
#         nibabel.save(g, gii)
        
#         cmd = ("wb_command -metric-to-volume-mapping %s " % gii 
#         + "%s %s %s " % (midsurf, ref, f)
#         + "-ribbon-constrained %s %s -voxel-subdiv %d" % (insurf, outsurf, factor))
#         run(cmd, shell=True)
#         data = loadnii(f)
        
#     return data 
    