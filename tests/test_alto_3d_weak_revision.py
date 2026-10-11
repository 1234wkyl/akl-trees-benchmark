import importlib.util
from pathlib import Path
import numpy as np

script=Path(__file__).resolve().parents[1]/'scripts/103_refine_weak_labels_with_alto_3d.py'
spec=importlib.util.spec_from_file_location('alto103',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_local_terrain_plane_on_slope():
    x,y=np.meshgrid(np.arange(-2,3),np.arange(-2,3));z=.3*x-.2*y+20
    ground=np.column_stack([x.ravel(),y.ravel(),z.ravel()]);point=np.array([[.3,.4,30.]])
    xyz=np.vstack([ground,point]);cls=np.r_[np.full(len(ground),2),5]
    terrain,supported,rmse=mod.terrain_height(xyz,cls)
    assert supported[-1] and abs(terrain[-1]-(20+.3*.3-.2*.4))<1e-8
    assert rmse[-1]<1e-8

def test_unsupported_ground_does_not_trigger_height_filter():
    xyz=np.array([[0.,0.,0.],[1.,1.,0.],[.5,.5,30.]])
    terrain,supported,_=mod.terrain_height(xyz,np.array([2,2,5]))
    assert not supported.any()
    assert not mod.height_conflict(xyz[:,2]-terrain,supported,5.).any()

def test_height_filter_keeps_lower_branches_and_abstains_only_on_gross_conflicts():
    h=np.array([.5,2.,15.,17.9,18.1,-1.1]);valid=np.ones(6,bool)
    assert mod.height_conflict(h,valid,15.).tolist()==[False,False,False,False,False,True]
    h=np.array([19.4,19.6]);assert mod.height_conflict(h,np.ones(2,bool),15.).tolist()==[False,True]
