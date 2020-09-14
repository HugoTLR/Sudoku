import cv2 as cv
import numpy as np
import glob

#Built-in
from collections import defaultdict
from time import time
#3rd party
# from imutils import grab_contours
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#Local Package
from SudokuDLX.SudokuDLX import SudokuDLX
from Utils.display import Display
from Utils.extractor import FeatureExtractor
from Utils.tracker import Tracker
from AR.KPExtractor import KPExtractor



if __name__ == "__main__":
  R_H,R_W = 288,288
  CELL_SIZE = 28

  W,H = 640,480
  #FOCAL
  F = 800
  #INTRINSIC CAMERA MATRIX
  K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

  fn_pattern = "./pattern2.jpg"

  # Load Classification model
  # model_path = "./Model/Output/optimized.h5"
  model_path = "./Model/Output/tmp/cp-38-0.0509.h5"
  sudoku_model = load_model(model_path)


  #Instanciate Classes
  di = Display()
  ex = FeatureExtractor()
  tr = Tracker(fn_pattern)
  kp = KPExtractor(K)

  ui = None

  SOLVED = False

  cap = cv.VideoCapture(0)
  while True:
    s_loop = time()
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

      #If it looks like a sudoku
      if tr.compare(warp_thresh):
        tr.puzzle = puzzle
        #If we alread found and decoded it skip
        if SOLVED:
          if tr.last_puzzle is not None:
            #Grab Tracking data
            src_pts = tr.get_cnts_points(True)
            dst_pts = tr.get_cnts_points(False)

            #Build homography and reprojection matrix
            H,mask = cv.findHomography(src_pts,dst_pts,cv.RANSAC,5.0)
            matrix = kp.project_matrix(H)

            #Render
            orig = di.render_warp(orig,ui,matrix,ui_pts,src_pts,dst_pts)
        
        else:
          #Resize to Classifier input size
          warp = cv.resize(warp,(R_W,R_H))

          #Retrieve Sudoku cells
          cells = ex.extract_cells(warp)
          cells = ex.threshold_cells(cells)
          cells,digits = ex.clear_cells(cells)

          #Ui for debugging purpose
          # ui = ex.ui(cells)

          #TODO: Look under
          preds = {}

          #Initiate Sudoku Empty board
          board = [[0 for _ in range(9)] for _ in range(9)]
          for i,(c,d) in enumerate(zip(cells,digits)):
            #If digit in cell
            if d:
              roi = cv.resize(c,(CELL_SIZE,CELL_SIZE))
              roi = roi.astype("float") / 255.0
              roi = img_to_array(roi)
              roi = np.expand_dims(roi,axis=0)

              #Predict Digit
              #If even with a good model we didn't manage to predict correctly,
              #Run X preds and select best for each to smooth potential errors (6/8/9),(1/7)...
              prediction = sudoku_model.predict(roi).argmax(axis=1)[0]
              board[i//9][i%9] = prediction


          
          #Run SudokuDLX with the predicted board
          sudoku = SudokuDLX()
          SudokuDLX.print_solution(board)
          solved = sudoku.solve(board,True)
          if solved[0]:

            ui = di.build_sudoku_ui(tr.pattern,500,500,solved[1])
            ui = cv.rotate(ui,cv.ROTATE_90_COUNTERCLOCKWISE)
            ui_pts = np.float32([[0, 0], [0, ui.shape[0] - 1], [ui.shape[1] - 1, ui.shape[0] - 1], [ui.shape[1] - 1, 0]]).reshape(-1, 1, 2)

            SOLVED = True


        tr.last_puzzle = tr.puzzle
          




    loop_time = time()-s_loop
    if di.show(a=[orig]):
      break
    
    # print(f"Looped in {loop_time:.3f} secs")





  cv.destroyAllWindows()
  cap.release()



