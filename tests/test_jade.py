import time
from pathlib import Path
import pytest
from deepforge.advanced import read_binary_scores
from deepforge.gui_data import preview_threshold
from deepforge.reporting import chart_svg

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'

@pytest.mark.parametrize('threshold', [0, .1, .5, .9, 1])
def test_threshold_preview_matches_recompute(threshold):
    source = EXAMPLES / 'scores.csv'
    base = read_binary_scores(source)
    actual = read_binary_scores(source, threshold=threshold)['threshold_metrics']
    p = preview_threshold(base, threshold)
    labels = actual['labels']; matrix = actual['confusion_matrix']; i = labels.index('1'); j = labels.index('0')
    assert (p['tp'],p['fp'],p['fn'],p['tn']) == (matrix[i][i],matrix[j][i],matrix[i][j],matrix[j][j])
    assert p['f1'] == pytest.approx(2*p['tp']/(2*p['tp']+p['fp']+p['fn']) if p['tp'] else 0)

@pytest.mark.parametrize('threshold', [-.1, 1.1, float('nan')])
def test_threshold_preview_invalid(threshold):
    with pytest.raises(ValueError): preview_threshold(read_binary_scores(EXAMPLES/'scores.csv'),threshold)

def test_probability_report():
    result=chart_svg({'type':'probability','title':'Precision–recall','x':[0,1],'y':[1,.5],'xlabel':'Recall','baseline':.5})
    assert 'Precision–recall' in result and 'Recall' in result and 'stroke-dasharray' in result

def test_jade_interaction():
    tk=pytest.importorskip('tkinter')
    from deepforge.gui import Studio
    try: root=tk.Tk()
    except tk.TclError: pytest.skip('A graphical display is required')
    try:
        app=Studio(root);root.geometry('1000x720');root.update()
        app.nav_query.set('binary');root.update()
        assert app.nav['Scores'].winfo_manager()=='pack'
        assert not app.nav['Dataset'].winfo_manager()
        app.open_first_match();assert app.tool=='Scores'
        app.nav_query.set('');app.toggle_sidebar();assert not app.side.winfo_manager()
        app.toggle_sidebar();assert app.side.winfo_manager()=='pack'
        app.load_demo();app.fields['threshold'].set('0.7');app.select('Training');app.select('Scores');assert app.fields['threshold'].get()=='0.7'
        app.run();deadline=time.monotonic()+20
        while app.busy and time.monotonic()<deadline: root.update();time.sleep(.02)
        assert app.report is not None and app.report['threshold']==.7
        app.threshold_var.set(.2);app.threshold_changed(.2);assert '0.200' in app.threshold_info.cget('text')
        assert app.report['threshold']==.7 # Preview does not silently change the report.
        app.apply_threshold();deadline=time.monotonic()+20
        while app.busy and time.monotonic()<deadline: root.update();time.sleep(.02)
        assert app.report['threshold']==.2
        app.find_query.set('nonexistent_text');assert not app.table.get_children()
        app.find_query.set('');assert app.table.get_children()
        for mode in ('ROC','Precision–recall','Calibration'):
            app.chart_mode.set(mode);app.draw();root.update();assert app.active_chart() is not None
        dialog=app.help();root.update();assert dialog.winfo_exists();dialog.destroy()
    finally:root.destroy()
