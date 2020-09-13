import time
import cv2 as cv
import numpy as np

from skimage.metrics import structural_similarity as ssim
from Utils.display import Display
from Utils.extractor import FeatureExtractor

from .KPExtractor import KPExtractor

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
di = Display(LOOP_DELAY=1)
fe = KPExtractor(K)
# Load Sudoku pattern for feature matching
pat = cv.cvtColor(cv.imread("./pattern2.jpg"),cv.COLOR_BGR2GRAY)
h,w = pat.shape


result = [[0,8,0,0,0,7,0,0,9],\
    [1,0,0,0,0,0,6,0,0],\
    [0,0,0,3,0,0,0,8,0],\
    [0,0,2,0,3,0,0,0,7],\
    [0,0,0,2,1,4,0,0,0],\
    [5,0,0,0,9,0,4,0,0],\
    [0,5,0,0,0,3,0,0,0],\
    [0,0,4,0,0,0,0,0,3],\
    [6,0,0,1,0,0,0,2,0]]
ui = di.build_sudoku_ui(pat,500,500,result)
ui = cv.rotate(ui,cv.ROTATE_90_COUNTERCLOCKWISE)
cv.imshow("ui",ui)
cv.waitKey()
ui_pts = np.float32([[0, 0], [0, ui.shape[0] - 1], [ui.shape[1] - 1, ui.shape[0] - 1], [ui.shape[1] - 1, 0]]).reshape(-1, 1, 2)


last_puzzle = None

LOCATED = False

cap = cv.VideoCapture(0)
while True:
  s_loop = time.time()
  ret,orig = cap.read()

  if not ret:
    break

  #Process Frames
  frame = ex.auto_process(orig)
  gray = ex.auto_process(orig,blur=False)


  # Pas besoin de mask ici -> utilisé pour isoler capot de voiture sur dashcam
  # Isoler tout ce qui n'est pas dans le puzzle semble etre une mauvaise idée (ou alors mask en cercle autour du centre du countour)
  # if fe.orb_mask is None:
  #   fe.build_mask(frame)
  # matches, Rt = fe.extract(gray)


  warp = None
  last_warp = None



  #Look for 4 squared cnt
  puzzle, thresh = ex.find_puzzle(frame)
  if puzzle is not None:

    #Unwrap it
    warp_orig = ex.unwrap(orig,puzzle)
    warp_thresh = ex.unwrap(thresh,puzzle)
    warp = ex.unwrap(gray,puzzle)

    #Prepare for SSIM comparison
    warp_cmp = cv.bitwise_not(warp_thresh)
    warp_cmp = cv.resize(warp_cmp,(w,h),interpolation=cv.INTER_AREA)

    #Compare and rescale score to 0..1
    score_ssim = ssim(warp_cmp,pat, data_range=pat.max() - pat.min())
    score_ssim = score_ssim*.5 +.5


    # print(ui_pts)
    # ui_pts = ex.order_points(ui_pts)[0]
    # print(ui_pts)
    #If it looks like a sudoku
    if score_ssim > SSIM_TRESHOLD:
      LOCATED = True

      
      print("1")

      if last_puzzle is not None:

        print("2")
        src_pts = np.float32([p for p in last_puzzle]).reshape(-1, 1, 2)
        dst_pts = np.float32([p for p in puzzle]).reshape(-1, 1, 2)






        H,mask = cv.findHomography(src_pts,dst_pts,cv.RANSAC,5.0)

        # # compute Homography
        # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)


        # print(H)
        # print(H)

        # orig = di.draw_homography(orig,warp_orig,H,src_pts)
        matrix = fe.project_matrix(H)

        # print(matrix)

        # print(matrix)
        orig = di.render_warp(orig,ui,matrix,ui_pts,src_pts,dst_pts)
      last_warp_orig = warp_orig
      last_warp = warp
      cv.drawContours(orig,[puzzle],0,(255,0,0),3)
      # if LOCATED:
      #   matches, Rt = fe.extract(gray)
      #   if Rt is not None:
      #     nRT = Rt[:,0],Rt[:,1], Rt[:,3]
      #     H = np.dot(fe.K,nRT)
      #     src_pts = np.float32([p for p in last_puzzle]).reshape(-1, 1, 2)
      #     # print(fe.K,nRT)
      #     # print(f"{H=}")

      #     dst_pts = np.float32([pp for pp in puzzle]).reshape(-1, 1, 2)
      #     # compute Homography
      #     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

      #     print(H,M)
      #     orig = di.draw_homography(orig,warp_orig,M,dst_pts)

      #     cv.waitKey()
          # print(len(matches))
          # src_pts = np.float32([p for p in last_puzzle]).reshape(-1, 1, 2)
          # orig = di.render_warp(fe,orig,warp_orig,Rt,src_pts)

          # cv.waitKey()


    last_puzzle = puzzle
    # last_warp = warp_orig
    
  loop_time = time.time()-s_loop
  # print(f"Matches={len(matches)}\tAvg FPS : {1/loop_time}")
  if di.show(a=[orig]):
      break
    
  # print(f"Looped in {loop_time:.3f} secs")





cv.destroyAllWindows()
cap.release()
