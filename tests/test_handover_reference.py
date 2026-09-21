import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
from shapely.geometry import Polygon,box

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('handover',ROOT/'docs/handover_2026-09-18/assemble_adjudicated_reference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class HandoverAssemblyCases(unittest.TestCase):
    def fixtures(self):
        sites={sid:{'bbox_epsg2193':[i*100,0,i*100+100,100],'split':split,'stratum':'fixture'} for i,(sid,split) in enumerate([('S001','train'),('S003','val'),('S016','val'),('S020','test')])}
        base=gpd.GeoDataFrame([{'label_id':sid+'-old','site_id':sid,'class':'tree','geometry':box(s['bbox_epsg2193'][0]+10,10,s['bbox_epsg2193'][0]+20,20)} for sid,s in sites.items()],crs=2193)
        curated=base[base.site_id.isin(m.CURATED_SITES)].copy();curated['label_id']=curated.site_id+'-final'
        return sites,base,curated
    def test_whole_site_replacement_preserves_other_sites(self):
        sites,base,curated=self.fixtures();curated.loc[curated.site_id=='S016','class']='tree_group'
        final,audit=m.prepare_frames(base,curated,sites)
        self.assertEqual(set(final.label_id),{'S001-old','S003-old','S016-final','S020-final'})
        for sid in ['S001','S003']:self.assertTrue(final[final.site_id==sid].geometry.iloc[0].equals_exact(base[base.site_id==sid].geometry.iloc[0],0))
        self.assertEqual(len(audit),4);self.assertEqual(sum(final['class']=='tree'),3)
    def test_invalid_curated_geometry_is_not_silently_repaired(self):
        sites,base,curated=self.fixtures();curated.at[curated.index[0],'geometry']=Polygon([(210,10),(220,20),(210,20),(220,10)])
        with self.assertRaisesRegex(ValueError,'geometry'):m.prepare_frames(base,curated,sites)
    def test_curated_crown_hole_is_preserved(self):
        sites,base,curated=self.fixtures()
        outer=[(210,10),(220,10),(220,20),(210,20)]
        hole=[(212,12),(214,12),(214,14),(212,14)]
        curated.at[curated.index[0],'geometry']=Polygon(outer,[hole])
        final,_=m.prepare_frames(base,curated,sites)
        p=final[final.site_id=='S016'].geometry.iloc[0]
        self.assertEqual(p.area,96);self.assertEqual(len(p.interiors),1)
    def test_synthetic_agreement_cannot_assemble_real_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'receipt.json';p.write_text(json.dumps({'synthetic_only':True,'status':'SYNTHETIC_TEST_ONLY'}))
            with self.assertRaisesRegex(ValueError,'Real original'):m.validate_records(SimpleNamespace(agreement_receipt=p))


if __name__=='__main__':unittest.main()
