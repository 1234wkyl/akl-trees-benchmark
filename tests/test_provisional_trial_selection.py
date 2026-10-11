import importlib.util,sys,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts/17_select_training_trial.py'
sys.path.insert(0,str(p.parent));spec=importlib.util.spec_from_file_location('trial_selection',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TrialGate(unittest.TestCase):
    def record(self):return {'status':'TRAINING-ONLY FRESH DETECTION AND CROWN COMPARISON','frozen_seed_coordinates_used_for_generation':False,'historical_reproduction_verified':False,'by_site':{'S001':{}},'crown_input_sha256':'evidence'}
    def test_provisional_mode_does_not_satisfy_historical_default_gate(self):
        with self.assertRaises(ValueError):m.validate_reproduction_record(self.record(),{'S001'})
        m.validate_reproduction_record(self.record(),{'S001'},True)
    def test_heldout_or_incomplete_receipt_cannot_enter_provisional_selection(self):
        with self.assertRaises(ValueError):m.validate_reproduction_record(self.record(),{'S001','S006'},True)
        r=self.record();r['frozen_seed_coordinates_used_for_generation']=True
        with self.assertRaises(ValueError):m.validate_reproduction_record(r,{'S001'},True)
