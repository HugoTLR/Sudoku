from cv2 import bitwise_not
from cv2 import cvtColor
from cv2 import imread
from cv2 import resize
from numpy import float32
from skimage.metrics import structural_similarity as ssim

from cv2 import COLOR_BGR2GRAY
from cv2 import INTER_AREA

class Tracker:
  SSIM = .6
  def __init__(self,fn_pattern):
    self.pattern = cvtColor(imread(fn_pattern),COLOR_BGR2GRAY)
    self.pattern_h = self.pattern.shape[0]
    self.pattern_w = self.pattern.shape[1]
    self.ssim_range = self.pattern.max()-self.pattern.min()

  
    self.last_puzzle = None
    self.puzzle = None


  def compare(self,th_warp):
    #Compare thresholded warp with sudoku pattern
    #Return true if it match great
    cmp_warp = bitwise_not(th_warp)
    cmp_warp = resize(cmp_warp,(self.pattern_w,self.pattern_h),interpolation=INTER_AREA)
    score = ssim(cmp_warp,self.pattern,data_range = self.ssim_range)
    score = score * .5 + .5 #[-1;1] to [0;1]
    return score >= Tracker.SSIM


  def get_cnts_points(self,last=False):
    if last:
        return float32([p for p in self.last_puzzle]).reshape(-1, 1, 2)
    else:
        return float32([p for p in self.puzzle]).reshape(-1, 1, 2)