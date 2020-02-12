import os.path 
import toblerone
import subprocess


inroot = op.join('svb_data', 'fs_native')
outroot = op.join('svb_data', 'fs_32k')
struct2asl = np.loadtxt('svb_data/oxasl_run1_scale1/struc2asl.mat')
brain = 'svb_data/oxasl_run1_scale1/brain.nii.gz'
calib = 'svb_data/oxasl_run1_scale1/calib.nii.gz'

s2r_world = toblerone.utils._FLIRT_to_world(brain, calib, struct2asl)

for side in ['lh', 'rh']:
    for surf in ['white', 'pial']:
        sph = op.join(inroot, '%s.sphere' % side)
        s1 = op.join(inroot, '%s.%s' % (side, surf))
        ref = op.join(outroot, '30ksphere.surf.gii')
        out = op.join(outroot, '%s.%s.surf.gii' % (side, surf))
        cmd = 'wb_command -surface-resample %s %s %s BARYCENTRIC %s' % (s1, sph, ref, out)
        subprocess.run(cmd, shell=True)

        out2 = op.
