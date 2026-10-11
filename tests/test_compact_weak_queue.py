import importlib.util,json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'scripts/104_run_compact_weak_checkpoint_queue.py'
s=importlib.util.spec_from_file_location('compact104',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_compaction_preserves_raw_and_reconciled(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    dest=tmp_path/'outputs/nectar/new_experiment/tile';pred=dest/'prediction';(pred/'final_results').mkdir(parents=True)
    raw=pred/'final_results/result.laz';raw.write_bytes(b'raw');reconciled=dest/'reconciled.laz';reconciled.write_bytes(b'checked')
    waste=pred/'voxel_intermediate.pt';waste.write_bytes(b'regenerate')
    m.compact_prediction(pred,raw)
    assert raw.read_bytes()==b'raw' and reconciled.read_bytes()==b'checked' and not waste.exists()
    assert json.loads((dest/'intermediate_cleanup.json').read_text())['removed_bytes']==10

def test_compaction_refuses_symlinks(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    pred=tmp_path/'outputs/nectar/new_experiment/tile/prediction';pred.mkdir(parents=True)
    raw=pred/'raw.laz';raw.write_bytes(b'raw');outside=tmp_path/'input.laz';outside.write_bytes(b'original')
    (pred/'unsafe').symlink_to(outside)
    with pytest.raises(ValueError,match='symbolic links'):m.compact_prediction(pred,raw)
    assert outside.read_bytes()==b'original'
