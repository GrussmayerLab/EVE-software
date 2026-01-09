#Quick util that cuts a hdf5 based on specific x/y values
import h5py
import numpy as np


# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "CutHDF_xy": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "",
            "display_name": "Cut HDF5 in XY"
        },
        "CutHDF_time": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "",
            "display_name": "Cut HDF5 in time"
        },
        "RAW_to_HDF": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "",
            "display_name": "Raw to HDF"
        }
    }

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog

#Imports for PyQt5 (GUI)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor, QTextCursor, QIntValidator, QColor
from PyQt5.QtWidgets import QHBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QComboBox, QLineEdit, QToolBar, QCheckBox,QDesktopWidget, QMessageBox, QTextEdit, QSlider, QSpacerItem, QTableView, QFrame, QScrollArea, QProgressBar, QMenu, QMenuBar, QColorDialog
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication, QTimer, QFileSystemWatcher, QFile, QThread, pyqtSignal, QObject
import typing
# from Utils import utils, utilsHelper

def CutHDF_xy_run(loadfile,savefile,xyStretch):
    xyStretch=eval(xyStretch)
    dataLocation = loadfile
    #Get the time slice from a hdf5 file for index, after running findIndexFromTimeSliceHDF
    print('Starting to read')
    with h5py.File(dataLocation, mode='r') as file:
        events = file['CD']['events']
        #filter out all events that are not in the xyStretch:
        events = events[(events['x'] >= xyStretch[0]) & (events['x'] <= xyStretch[1]) & (events['y'] >= xyStretch[2]) & (events['y'] <= xyStretch[3])]

    print('Starting to save')
    #Store these events as a new hdf5:
    with h5py.File(savefile, mode='w') as file:
        events = file.create_dataset('CD/events', data=events, compression="gzip")

    print('Saved')

def CutHDF_time_run(loadfile,savefile,timeStretch):
    timeStretch=eval(timeStretch)
    #ms to us
    timeStretch=[timeStretch[0]*1000,timeStretch[1]*1000]
    dataLocation = loadfile
    #Get the time slice from a hdf5 file for index, after running findIndexFromTimeSliceHDF
    print('Starting to read')
    with h5py.File(dataLocation, mode='r') as file:
        events = file['CD']['events']
        #filter out all events that are not in the timeStretch:
        events = events[(events['t'] >= timeStretch[0]) & (events['t'] <= timeStretch[1])]

    print('Starting to save')
    #Store these events as a new hdf5:
    with h5py.File(savefile, mode='w') as file:
        events = file.create_dataset('CD/events', data=events, compression="gzip")

    print('Saved')


def readRawTimeStretch(filepath,metaVisionPath,buffer_size = 10e7, n_batches=5e7, timeStretchMs=[0,1000]):
    import logging
    #Function to read only part of the raw file, between time stretch [0] and [1]
    sys.path.append(metaVisionPath)
    from metavision_core.event_io.raw_reader import RawReader
    record_raw = RawReader(filepath,max_events=int(buffer_size))
    #First seek to the start-time:
    record_raw.seek_time(timeStretchMs[0]*1000)
    #Then load the time in a single batch:
    events = record_raw.load_delta_t(timeStretchMs[1]*1000-timeStretchMs[0]*1000)
    record_raw.reset()
    return events

def raw_to_hdf_run(loadfile,savefile,parent,TimeMsPerStep):

    import logging

    totEvents = []
    logging.info('Starting raw to hdf5 conversion')

    continuing = 1
    i=0 #counter
    while continuing:
        try:
            logging.info('Reading raw at time '+str(TimeMsPerStep*i))
            events = readRawTimeStretch(loadfile,parent.globalSettings['MetaVisionPath']['value'],buffer_size = 1e7, n_batches=1e6, timeStretchMs=[TimeMsPerStep*(i),TimeMsPerStep*(i+1)])
            if len(totEvents)==0:
                totEvents = events
            else:
                totEvents = np.append(totEvents,events)
            i=i+1
            if len(events) == 0:
                logging.info('Finished at time '+str(TimeMsPerStep*i))
                continuing = 0
        except:
            logging.info('Finished (errored) at time '+str(TimeMsPerStep*i))
            continuing=0

    logging.info('Starting to save')
    #Store these events as a new hdf5:
    with h5py.File(savefile, mode='w') as file:
        events = file.create_dataset('CD/events', data=totEvents, compression="gzip")

    logging.info('Saved')

def CutHDF_xy(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:

    try:
        from eve_smlm.Utils import utils
    except ImportError:
        from Utils import utils

    window = utils.SmallWindow(parent,windowTitle="Cut HDF5 file in xy")
    window.addDescription("This function allows you to cut an hdf5 file between certain x,y coordinates. Please find the file location, specify the save location, and specify the XY boundaries (e.g. '(100,200,200,250)' to cut 100-200 in x, 200-250 in y)")
    loadfileloc = window.addFileLocation()
    savefileloc = window.addFileLocation(labelText="Save location:", textAddPrePeriod = "_xyCut")

    xyStretchText = window.addTextEdit(labelText="XY boundaries:",preFilledText="(0,np.inf,0,np.inf)")

    button = window.addButton("Run")
    button.clicked.connect(lambda: CutHDF_xy_run(loadfileloc.text(),savefileloc.text(),xyStretchText.text()))

    window.show()
    pass

def CutHDF_time(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:

    try:
        from eve_smlm.Utils import utils
    except ImportError:
        from Utils import utils

    window = utils.SmallWindow(parent,windowTitle="Cut HDF5 file in time")
    window.addDescription("This function allows you to cut an hdf5 file between certain time coordinates. Please find the file location, specify the save location, and specify the time boundaries (e.g. '(0,10000)' to cut 0-10000 ms)")
    loadfileloc = window.addFileLocation()
    savefileloc = window.addFileLocation(labelText="Save location:", textAddPrePeriod = "_timeCut")

    timeStretchText = window.addTextEdit(labelText="Time boundaries:",preFilledText="(0,10000)")

    button = window.addButton("Run")
    button.clicked.connect(lambda: CutHDF_time_run(loadfileloc.text(),savefileloc.text(),timeStretchText.text()))

    window.show()
    pass

def RAW_to_HDF(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:

    try:
        from eve_smlm.Utils import utils
    except ImportError:
        from Utils import utils

    window = utils.SmallWindow(parent,windowTitle="RAW to HDF")
    window.addDescription("Transform a RAW file to a HDF5 format for quicker future reading")
    loadfileloc = window.addFileLocation()
    savefileloc = window.addFileLocation(labelText="Save location:", textAddPrePeriod = "_hdf",textPostPeriod = "hdf5")
    TimeMsPerStepText = window.addTextEdit(labelText="Time per step (ms):",preFilledText="120000")

    button = window.addButton("Run")
    button.clicked.connect(lambda: raw_to_hdf_run(loadfileloc.text(),savefileloc.text(),parent,int(TimeMsPerStepText.text())))

    window.show()
    pass
