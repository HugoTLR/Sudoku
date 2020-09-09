import os

CLASSES = ['0','1','2','3','4','5','6','7','8','9']

MIN_LR = 1e-5
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
CLR_METHOD = 'triangular2'
NUM_EPOCHS = 48


LRFIND_PLOT_PATH = os.path.sep.join(['output','lrfind_plot.png'])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])