import time
import cv2 as cv
import numpy as np

from skimage.metrics import structural_similarity as ssim
from Utils.display import Display
from Utils.extractor import FeatureExtractor
from Utils.tracker import Tracker
from AR.KPExtractor import KPExtractor


#Const
SSIM_TRESHOLD = .6
R_H,R_W = 288,288
CELL_SIZE = 28

W,H = 640,480
#FOCAL
F = 800
#INTRINSIC CAMERA MATRIX
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

ex = FeatureExtractor()
di = Display()
fe = KPExtractor(K)

fn_pattern = "./pattern2.jpg"

tr = Tracker(fn_pattern)



result = [[0,8,0,0,0,7,0,0,9],\
    [1,0,0,0,0,0,6,0,0],\
    [0,0,0,3,0,0,0,8,0],\
    [0,0,2,0,3,0,0,0,7],\
    [0,0,0,2,1,4,0,0,0],\
    [5,0,0,0,9,0,4,0,0],\
    [0,5,0,0,0,3,0,0,0],\
    [0,0,4,0,0,0,0,0,3],\
    [6,0,0,1,0,0,0,2,0]]



ui = di.build_sudoku_ui(tr.pattern,500,500,result)
ui_pts = np.float32([[0, 0], [ui.shape[1] - 1, 0], [ui.shape[1] - 1, ui.shape[0] - 1], [0, ui.shape[0] - 1] ]).reshape(-1, 1, 2)



cap = cv.VideoCapture(0)
while True:
  s_loop = time.time()
  ret,orig = cap.read()
  if not ret:
    break

  #Process Frames
  frame = ex.auto_process(orig)
  gray = ex.auto_process(orig,blur=False)


  warp = None

  #Look for 4 squared cnt
  puzzle, thresh = ex.find_puzzle(frame)

  if puzzle is not None:

    #Unwrap it
    warp_orig = ex.unwrap(orig,puzzle)
    warp_thresh = ex.unwrap(thresh,puzzle)
    warp = ex.unwrap(gray,puzzle)



    if tr.compare(warp_thresh):
      tr.puzzle = puzzle

      if tr.last_puzzle is not None:
        #Grab Tracking data
        src_pts = tr.get_cnts_points(True)
        dst_pts = tr.get_cnts_points(False)
        src_pts = ex.order_points(src_pts.reshape((4,2))[::-1])
        dst_pts = ex.order_points(dst_pts.reshape((4,2))[::-1])

        #Build homography and reprojection matrix
        H,mask = cv.findHomography(src_pts,dst_pts,cv.RANSAC,5.0)
        matrix = fe.project_matrix(H)

        #Render
        orig = di.render_warp(orig,ui,matrix,ui_pts,src_pts,dst_pts)

      tr.last_puzzle = tr.puzzle
  loop_time = time.time()-s_loop
  # print(f"Matches={len(matches)}\tAvg FPS : {1/loop_time}")
  if di.show(a=[orig]):
      break



cv.destroyAllWindows()
cap.release()
