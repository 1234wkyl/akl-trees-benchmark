"""Run inside the isolated SAT container; no benchmark metrics are produced."""
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

out=Path('/benchmark-output')
modules=['numpy','scipy','yaml','hydra','laspy','jaklas','pykdtree',
         'torch_scatter','torch_sparse','torch_cluster','torch_points_kernels',
         'MinkowskiEngine','torchsparse']
versions={name:str(getattr(importlib.import_module(name),'__version__','not exposed')) for name in modules}
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable inside SAT container')
x=torch.arange(16,dtype=torch.float32,device='cuda').reshape(4,4)
assert torch.isfinite(x @ x.T).all().item()
checkpoint=torch.load('/home/nibio/mutable-outside-world/model_file/PointGroup-PAPER.pt',map_location='cpu')
if not isinstance(checkpoint,dict) or 'models' not in checkpoint or 'run_config' not in checkpoint:
    raise RuntimeError('checkpoint lacks expected models/run_config')
data={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
      'cuda_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0),
      'modules':versions,'checkpoint_keys':sorted(map(str,checkpoint)),
      'checkpoint_weight_names':sorted(map(str,checkpoint['models'])),
      'cuda_tensor_check':True,'checkpoint_deserialised':True,
      'note':'Weight deserialisation and extension imports passed; architecture loading is verified only by successful inference.'}
(out/'container_preflight.json').write_text(json.dumps(data,indent=2)+'\n')
with (out/'container_pip_freeze.txt').open('w') as f:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
print(json.dumps(data,indent=2))
