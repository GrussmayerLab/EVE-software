import os
import numpy as np
import h5py
import pandas as pd
import unittest
from unittest.mock import MagicMock, patch
import sys
import shutil

# Add project root to path
sys.path.append('/home/philippehenry/Documents/Master_DSAIT/0MastersThesis/EVE-software')

# Mock PyQt5 generic
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['PyQt5.QtGui'] = MagicMock()
sys.modules['PyQt5.QtWebEngineWidgets'] = MagicMock()

# Mock Napari/Vispy
sys.modules['napari'] = MagicMock()
sys.modules['napari.qt'] = MagicMock()
sys.modules['napari.layers'] = MagicMock()
sys.modules['napari.utils.events'] = MagicMock()
sys.modules['vispy'] = MagicMock()
sys.modules['vispy.color'] = MagicMock()

# Critical: Mock matplotlib's QT backend so it doesn't try to load real PyQt5
# But allow other parts of matplotlib to be real (loaded from .venv)
mock_qt5agg = MagicMock()
sys.modules['matplotlib.backends.backend_qt5agg'] = mock_qt5agg

# Import after mocking
from GUI_main import DataAnalysisWidget
import logging

class TestBatchAnalysis(unittest.TestCase):
    def setUp(self):
        # Configure logging to see output
        logging.basicConfig(level=logging.DEBUG)
        
        # Create temp data directory
        self.test_dir = os.path.abspath('./temp_test_batch_analysis') # Use absolute path to be safe
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Create dummy data
        self.create_dummy_data('dataset1.hdf5')
        self.create_dummy_data('dataset2.npy')
        self.create_dummy_data('dataset3.raw') # Just empty file for check
        
        # Mock Parent and Widget
        self.mock_parent = MagicMock()
        # Mock loadRawData to return dummy events
        self.mock_parent.loadRawData.side_effect = self.mock_load_data
        
        self.widget = DataAnalysisWidget(self.mock_parent)
        
        # Setup mocks for GUI elements that we can't instantiate fully
        self.widget.analysisDropdown = MagicMock()
        self.widget.analysisDropdown.currentText.return_value = "Event Structural Ratio" 
        self.widget.functionNameToDisplayNameMapping = [("Event Structural Ratio", "run_analysis")]
        
        # Mock results text
        self.widget.resultsText = MagicMock()
        
        # Mock settings groupbox layout
        self.widget.settingsGroupbox = MagicMock()
        self.widget.settingsGroupbox.layout.return_value = MagicMock()
        self.widget.settingsGroupbox.layout().count.return_value = 0 # No extra args

    def create_dummy_data(self, filename):
        path = os.path.join(self.test_dir, filename)
        if filename.endswith('.hdf5'):
             with h5py.File(path, 'w') as f:
                 # Create a dummy dataset just so it's a valid HDF5 file
                 f.create_dataset('events', data=np.zeros((10, 4)))
        else:
             with open(path, 'wb') as f:
                 f.write(b"dummy content")

    def mock_load_data(self, path):
        # Return a dummy events dict containing simple events
        # Column 0 is ts, 1 is x, 2 is y, 3 is p
        # Create 100 random events
        events = np.zeros((100, 4))
        events[:, 0] = np.arange(100) # ts
        events[:, 1] = np.random.randint(0, 100, 100) # x
        events[:, 2] = np.random.randint(0, 100, 100) # y
        events[:, 3] = np.random.randint(0, 2, 100) # p
        
        # DataAnalysisWidget expects a structured array usually? 
        # Or dictionary of arrays? 
        # Let's check GUI_main.py: loadRawData returns eventsDict. 
        # eventsDict[0] is usually the mixture.
        # But wait, utilsHelper.loadRawData might return something else.
        # In run_analysis_callback: "if len(eventsDict) > 1: np.vstack..."
        # So it returns a dict where values are numpy arrays.
        
        # Let's return a structured array to match 'events' expectation in analysis scripts usually
        # But for 'Event Structural Ratio' (eventStructuraRatio.py), it uses ev['x'], ev['y'].
        # So it expects a structured array.
        
        dtype = [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('p', '<u1')]
        structured_events = np.zeros(100, dtype=dtype)
        structured_events['t'] = np.arange(100)
        structured_events['x'] = np.random.randint(0, 256, 100)
        structured_events['y'] = np.random.randint(0, 256, 100)
        structured_events['p'] = np.random.randint(0, 2, 100)
        
        return {0: structured_events}

    @patch('GUI_main.logging')
    @patch('GUI_main.QMessageBox')
    @patch('builtins.eval')
    @patch('matplotlib.pyplot.figure')
    def test_batch_run(self, mock_figure, mock_eval, mock_msgbox, mock_logging):
        # Setup eval to return (figure, float_score)
        mock_fig = MagicMock()
        mock_eval.return_value = (mock_fig, 0.85)

        # Run
        print(f"Test Dir: {self.test_dir}")
        print(f"Files in test dir: {os.listdir(self.test_dir)}")
        
        print(f"Widget type: {type(self.widget)}")
        print(f"Has _run_batch_analysis? {hasattr(self.widget, '_run_batch_analysis')}")
        # print(f"Dir widget: {dir(self.widget)}")
        
        self.widget._run_batch_analysis(self.test_dir)
        
        # DEBUG CHECKS
        if mock_msgbox.information.called:
             print("DEBUG: QMessageBox.information called:", mock_msgbox.information.call_args)
        if mock_logging.error.called:
             print("DEBUG: logging.error called:", mock_logging.error.call_args_list)
        
        # Check results dir created
        results_dirs = [d for d in os.listdir(self.test_dir) if d.startswith('Analysis_Results_')]
        if not results_dirs:
            print("DEBUG: No Analysis_Results dir found in", os.listdir(self.test_dir))
        
        self.assertTrue(len(results_dirs) > 0)
        results_path = os.path.join(self.test_dir, results_dirs[0])
        
        # Check CSV created
        self.assertTrue(os.path.exists(os.path.join(results_path, 'summary_comparison.csv')))
        
        # Check images created
        # We have 3 files. dataset3.raw might fail loadRawData if real, but here mock_load_data catches it.
        # However, glboal.glob detects it.
        # We expect 3 images.
        pngs = [f for f in os.listdir(results_path) if f.endswith('.png')]
        self.assertEqual(len(pngs), 3)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
