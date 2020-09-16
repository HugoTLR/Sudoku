import os
import datetime

CLASSES = ['0','1','2','3','4','5','6','7','8','9']

MIN_LR = 1e-5
MAX_LR = 1e-2
BATCH_SIZE = 32
STEP_SIZE = 8
CLR_METHOD = 'triangular2'
NUM_EPOCHS = 80

FMT_DATE = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

TRAINING_PATH = os.path.sep.join(['output',FMT_DATE])
PLOT_PATH = os.path.sep.join([TRAINING_PATH,"Plots"])
TMP_PATH = os.path.sep.join([TRAINING_PATH,"tmp"])

CP_PATH = TMP_PATH + "/cp-{epoch:02d}-{val_loss:.4f}.h5"

LRFIND_PLOT_PATH = os.path.sep.join(['output','lrfind_plot.png'])
TRAINING_PLOT_PATH = os.path.sep.join([PLOT_PATH, "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join([PLOT_PATH, "clr_plot.png"])

SUMMARY_PATH = os.path.sep.join([TRAINING_PATH, "architecture.txt"])