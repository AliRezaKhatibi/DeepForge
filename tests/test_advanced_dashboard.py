from deepforge.dashboard import dashboard_html, TABS
from deepforge.monitor import TrainingMonitor


def test_advanced_dashboard_real_data(tmp_path):
    path = tmp_path / 'live.json'
    monitor = TrainingMonitor(path, epochs=5, batches_per_epoch=2)
    monitor.batch(epoch=1, batch=1, metrics={'loss': .6})
    monitor.batch(epoch=1, batch=2, metrics={'loss': .5})
    monitor.epoch_end(1, {'loss': .4, 'accuracy': .81, 'precision': .8})
    monitor.record_resources({'cpu_percent': 25, 'ram_percent': 45})
    markup = dashboard_html(path, metadata={'model': '<unsafe>', 'parameters': 1234})
    assert all(tab in markup for tab in TABS)
    assert '0.4000' in markup and '0.8100' in markup and '25.0%' in markup
    assert 'Epoch 1 / 5' in markup and 'Epoch 2 / 5' not in markup
    assert 'Not reported' in markup and '&lt;unsafe&gt;' in markup
    assert '<unsafe>' not in markup
    assert 'data:image/svg+xml;base64,' in markup


def test_advanced_dashboard_filters_and_empty(tmp_path):
    path = tmp_path / 'live.json'
    TrainingMonitor(path, epochs=5, batches_per_epoch=2)
    markup = dashboard_html(path, tab='Training', show_resources=False, show_charts=False,
                            show_epochs=False, show_metrics=False, show_diagnostics=False)
    assert 'Hardware monitor' not in markup and 'Loss per epoch' not in markup
    assert 'Val F1 score' not in markup and 'Training health' not in markup
    assert 'Epoch 0 / 5' in markup and 'Verified</span>' not in markup
