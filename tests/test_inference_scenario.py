import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark import inference_scenario as m
from benchmark.common import sha256
spec=importlib.util.spec_from_file_location('queue',Path(__file__).resolve().parents[1]/'scripts/48_run_training_tile_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
class ScenarioGate(unittest.TestCase):
    def test_inference_only_gate_binds_training_reference_and_rejects_freeze_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data/sites').mkdir(parents=True);(root/'config').mkdir()
            sites={'S001':{'split':'train'}}
            (root/'data/sites/reference_sites.json').write_text('{}')
            (root/'config/segmentanytree_source.json').write_text('{}')
            labels={'sites':{'S001':{'labels':[{'id':'a','cls':'tree','world':[[0,0],[1,0],[1,1]]}]}}}
            for n in ('current.json','old.json'):(root/n).write_text(json.dumps(labels))
            (root/'selection.json').write_text(json.dumps({'site_ids':['S001'],'selected_config_id':'convex_b100'}))
            with patch.object(m,'ROOT',root),patch.object(m,'site_definitions',return_value=sites):
                r={'mode':'ASSUMED_REFERENCE_PRETRAINED_INFERENCE_ONLY','formal_reference_frozen':False,
                    'independent_labels_received':False,'evaluation_permitted':False,'model_configuration_locked_for_this_scenario':True,
                    'site_definitions_sha256':sha256(root/'data/sites/reference_sites.json'),
                    'sat_config_sha256':sha256(root/'config/segmentanytree_source.json'),
                    'reference_json':'current.json','reference_json_sha256':sha256(root/'current.json'),
                    'original_selection_reference_json':'old.json','original_selection_reference_json_sha256':sha256(root/'old.json'),
                    'training_geometry_signature':m.training_signature(root/'old.json'),
                    'training_conversion_selection':'selection.json','training_conversion_selection_sha256':sha256(root/'selection.json'),
                    'conversion_config_id':'convex_b100'}
                record=root/'scenario.json';record.write_text(json.dumps(r));m.validate_inference_scenario(record)
                r['formal_reference_frozen']=True;record.write_text(json.dumps(r))
                with self.assertRaises(ValueError):m.validate_inference_scenario(record)
                r['formal_reference_frozen']=False;record.write_text(json.dumps(r))
                labels['sites']['S001']['labels'][0]['cls']='tree_group';(root/'current.json').write_text(json.dumps(labels))
                r['reference_json_sha256']=sha256(root/'current.json');record.write_text(json.dumps(r))
                with self.assertRaises(ValueError):m.validate_inference_scenario(record)
    def test_partial_queue_never_claims_completion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';rows=[{'tile':'a','status':'AUDITED'}]
            queue.update_receipt(p,'image',rows,['a','b'],'scenario')
            self.assertEqual(json.loads(p.read_text())['status'],'RUNNING')
            rows.append({'tile':'b','status':'AUDITED'});queue.update_receipt(p,'image',rows,['a','b'],'scenario')
            r=json.loads(p.read_text());self.assertEqual(r['status'],'COMPLETED');self.assertFalse(r['evaluation_permitted'])
