import json
from pathlib import Path

import pytest

from deepforge.advanced import compare_runs, read_binary_scores, reproducibility_manifest
from deepforge.audit import audit_groups, audit_images, audit_yolo
from deepforge.cli import main
from deepforge.reporting import export_report, table_rows


def file(root, name, contents):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(contents, bytes): p.write_bytes(contents)
    else: p.write_text(contents, encoding='utf-8')
    return p


def test_group_leakage_redacted(tmp_path):
    p = file(tmp_path, 'groups.csv', 'group_id,split\npatient-secret,train\npatient-secret,validation\nother,test\n')
    report = audit_groups(p)
    assert report['overlapping_groups'] == 1
    assert report['overlaps'][0]['splits'] == ['train', 'val']
    assert 'patient-secret' not in json.dumps(report)
    assert audit_groups(p, redact=False)['overlaps'][0]['group'] == 'patient-secret'


@pytest.mark.parametrize('row', ['x,unknown', ',train', 'x,', 'x'])
def test_invalid_group_row(tmp_path, row):
    p = file(tmp_path, 'groups.csv', 'group_id,split\n'+row+'\n')
    with pytest.raises(ValueError): audit_groups(p)


def test_group_cli_exit(tmp_path, capsys):
    p = file(tmp_path, 'groups.csv', 'group_id,split\na,train\na,test\n')
    assert main(['groups', str(p), '--fail-on-leakage']) == 1
    assert json.loads(capsys.readouterr().out)['overlapping_groups'] == 1


def test_yolo_audit(tmp_path):
    for name in ['images/train/a.jpg', 'images/train/b.jpg', 'images/val/c.jpg', 'images/val/c.png', 'images/test/bg.jpg']:
        file(tmp_path,name,b'fixture')
    file(tmp_path, 'labels/train/a.txt','0 .5 .5 .2 .2\n0 .5 .5 .2 .2\n2 .5 .5 .2 .2\n1 .1 .1 .5 .5\n1 .5 .5 nan .2\n1 .5 .5 .2\n')
    file(tmp_path, 'labels/val/c.txt','1 .5 .5 .3 .3\n')
    file(tmp_path, 'labels/test/bg.txt','')
    file(tmp_path, 'labels/train/orphan.txt','0 .5 .5 .2 .2')
    r = audit_yolo(tmp_path, classes=2)
    assert r['images'] == 5
    assert r['valid_boxes'] == 2
    assert r['invalid_rows'] == 5
    assert r['empty_label_files'] == 1
    assert r['missing_labels'] == ['train/b.txt']
    assert r['orphan_labels'] == ['train/orphan.txt']
    assert r['ambiguous_image_stems'] == ['val/c.txt']
    assert r['class_counts'] == {'0':1, '1':1}


@pytest.mark.parametrize('row', ['-1 .5 .5 .2 .2', '0 .5 .5 0 .2', '0 inf .5 .2 .2', '0.5 .5 .5 .2 .2', '0 .5 .5 .2 .2 .1'])
def test_yolo_bad_rows(tmp_path, row):
    file(tmp_path,'images/train/a.jpg',b'fixture'); file(tmp_path,'labels/train/a.txt',row)
    assert audit_yolo(tmp_path)['invalid_rows'] == 1


def test_image_quality_and_similarity(tmp_path):
    from PIL import Image
    im=Image.new('RGB',(30,20),(30,90,50))
    for name in ('train/a.png','val/b.png'):
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); im.save(p)
    file(tmp_path,'test/bad.png',b'broken')
    file(tmp_path,'test/unknown.dcm',b'dicom')
    r=audit_images(tmp_path,similarity=True,threshold=0)
    assert r['scanned']==4 and r['decoded']==2 and r['invalid_images']==1 and r['unsupported_images']==1
    assert r['candidate_pair_count']==1 and r['candidate_pairs'][0]['distance']==0
    assert r['dimensions']['width']['mean']==30
    assert audit_images(tmp_path,max_images=1)['scan_truncated']


def test_image_pair_cap(tmp_path):
    from PIL import Image
    for i in range(3):
        for split in ('train','test'):
            p=tmp_path/split/f'{i}.png'; p.parent.mkdir(parents=True,exist_ok=True); Image.new('L',(10,10),i).save(p)
    r=audit_images(tmp_path,similarity=True,threshold=0,max_pairs=2)
    assert r['cross_split_comparisons']==9 and r['candidate_pair_count']==9
    assert r['pair_list_truncated'] and len(r['candidate_pairs'])==2


def test_perfect_binary_scores(tmp_path):
    p=file(tmp_path,'scores.csv','y_true,score\n1,1\n0,0\n1,0.9\n0,0.1\n')
    r=read_binary_scores(p)
    assert r['roc_auc']==r['average_precision']==1
    assert r['brier_score']==pytest.approx(.005)
    assert r['threshold_metrics']['accuracy']==1
    assert sum(b['count'] for b in r['calibration'])==4
    assert r['calibration'][-1]['count']==2


def test_binary_ties(tmp_path):
    p=file(tmp_path,'scores.csv','y_true,score\n1,.5\n0,.5\n1,.5\n0,.5\n')
    r=read_binary_scores(p)
    assert r['roc_auc']==r['average_precision']==.5
    assert r['threshold_metrics']['accuracy']==.5
    assert len(r['roc_curve'])==2
    assert r['brier_score']==.25
    assert r['ece']==0


@pytest.mark.parametrize('score', ['nan','inf','-0.1','1.1',''])
def test_binary_invalid(tmp_path,score):
    p=file(tmp_path,'scores.csv','y_true,score\n1,'+score+'\n0,0\n')
    with pytest.raises(ValueError): read_binary_scores(p)


def test_binary_single_class(tmp_path):
    p=file(tmp_path,'scores.csv','y_true,score\n1,.5\n')
    with pytest.raises(ValueError): read_binary_scores(p)


def test_compare_rank_and_reject(tmp_path):
    file(tmp_path,'a.csv','epoch,val_loss\n1,.5\n2,.4\n'); file(tmp_path,'b.csv','epoch,val_loss\n1,.6\n2,.3\n'); file(tmp_path,'bad.csv','bad\n0\n')
    r=compare_runs(tmp_path)
    assert r['best_run']=='b.csv' and r['valid_runs']==2 and r['rejected_runs']==1
    assert compare_runs(tmp_path,mode='max')['best_run']=='b.csv'
    assert compare_runs(tmp_path,max_runs=1)['scan_truncated']


def test_manifest(tmp_path):
    p=file(tmp_path,'config.txt',b'abc')
    r=reproducibility_manifest(p,include_runtime=False)
    assert r['file']['sha256']=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert r['file']['size_bytes']==3
    assert str(tmp_path) not in json.dumps(r)


def test_export_escape(tmp_path):
    report={'<script>':'<img src=x onerror=alert(1)>','items':[1,2]}
    p=tmp_path/'report.html'; export_report(p,'<Title>',report,cards=[('A','<bad>')])
    result=p.read_text(encoding='utf-8'); assert '<script>' not in result and '&lt;script&gt;' in result
    assert '<img src=x' not in result and '&lt;Title&gt;' in result
    p=tmp_path/'report.json'; export_report(p,'Tool',report); assert json.loads(p.read_text())==report
    assert ('items','2 entries (see JSON)') in table_rows(report)


@pytest.mark.parametrize('command,example', [('scores','scores.csv'),('groups','groups.csv'),('compare','runs'),('manifest','history.csv')])
def test_cli_new_commands(command,example,capsys):
    path=Path(__file__).resolve().parents[1]/'examples'/example
    assert main([command,str(path)])==0
    assert isinstance(json.loads(capsys.readouterr().out),dict)


@pytest.mark.parametrize('tool,sample', [('Dataset','dataset'),('Leakage','dataset'),('Images','dataset'),('Groups','groups.csv'),('YOLO','yolo'),('Training','history.csv'),('Compare','runs'),('Metrics','predictions.csv'),('Scores','scores.csv')])
def test_bundled_demo(tool,sample):
    from deepforge.gui_data import analyze
    from deepforge import __file__ as package_path
    opts={'path':str(Path(package_path).parent/'samples'/sample), 'metric':'val_loss', 'mode':'min', 'epoch_column':'epoch', 'true_column':'y_true', 'pred_column':'y_pred'}
    report,chart=analyze(tool,opts)
    assert isinstance(report,dict) and chart is not None


@pytest.mark.parametrize('chart', [
    {'type':'bars','values':[('<script>',2),('b',-1)]},
    {'type':'lines','x':[1,2],'series':{'<script>':[.5,.4]}},
    {'type':'matrix','labels':['<script>','b'],'values':[[2,1],[0,3]]},
    {'type':'roc','x':[0,0,1],'y':[0,1,1]},
])
def test_html_visualization(tmp_path,chart):
    from deepforge.reporting import chart_svg
    svg=chart_svg(chart)
    assert svg.startswith('<svg') and svg.endswith('</svg>') and '<script>' not in svg
    p=tmp_path/'out.html'; export_report(p,'Tool',{'value':1},chart=chart)
    assert '<svg' in p.read_text(encoding='utf-8')


def test_findings_preview_cap():
    from deepforge.reporting import finding_rows
    rows=finding_rows({'issues':[{'file':'a','error':'bad'}]*5},limit=2)
    assert len(rows)==3 and rows[-1][0]=='Additional findings'
