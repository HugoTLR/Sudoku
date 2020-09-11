from extractor import FeatureExtractor
from display import Display
from imutils import grab_contours
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import numpy as np

from time import time

from collections import defaultdict
from solver import Sudoku
import glob

from cv2 import bitwise_not
import logging

from SudokuDLX import *


if __name__ == "__main__":

  logging_fn = f"./Videos/{len(glob.glob('./Videos/*.*'))}.log"
  logging.basicConfig(filename=logging_fn,format='%(levelname)s:%(message)s', level=logging.INFO)



  SSIM_TRESHOLD = .6
  R_H,R_W = 288,288
  CELL_SIZE = 28

  # model_path = "./Model/Output/optimized.h5"
  model_path = "./Model/Output/tmp/cp-38-0.0509.h5"
  sudoku_model = load_model(model_path)

  pat = cv.cvtColor(cv.imread("./pattern2.jpg"),cv.COLOR_BGR2GRAY)
  h,w = pat.shape



  di = Display()
  ex = FeatureExtractor()


  frame_cpt = 0 


  FOUND = False


  cap = cv.VideoCapture(0)
  while True:
    s_loop = time()
    ret,frame = cap.read()

    if not ret:
      break

    orig = frame
    gray = ex.auto_process(orig,blur=False)

    warp = None


    
    s_e = time()
    frame = ex.auto_process(orig)
    logging.info(f" ex.auto_process\t:\t{time()-s_e:.3f} seconds")


    puzzle, thresh = ex.find_puzzle(frame)
    if puzzle is not None:

      #Unwrap it
      s_e = time()
      warp_thresh = ex.unwrap(thresh,puzzle)
      warp_cmp = bitwise_not(warp_thresh)
      warp_cmp = cv.resize(warp_cmp,(w,h),interpolation=cv.INTER_AREA)

      warp = ex.unwrap(gray,puzzle)

      logging.info(f" ex.unwrap\t:\t{time()-s_e:.3f} seconds")
      
      #Create a version to compare with an empty sudoku pattern
      s_e = time()
      score_ssim = ssim(warp_cmp,pat, data_range=pat.max() - pat.min())
      score_ssim = score_ssim*.5 +.5
      logging.info(f" ssim {score_ssim}\t:\t{time()-s_e:.3f} seconds")



      #If it looks like a sudoku
      if score_ssim > SSIM_TRESHOLD:
        # logging.warning(f"----- WE ARE IN -----")
        cv.drawContours(orig,[puzzle],0,(255,0,0),3)
        if FOUND: continue
        #Resize
        warp = cv.resize(warp,(R_W,R_H))


        # s_e = time()
        cells = ex.extract_cells(warp)
        # logging.info(f" ex.extract_cells\t:\t{time()-s_e:.3f} seconds")

        # s_e = time()
        cells = ex.threshold_cells(cells)
        # logging.info(f" ex.threshold_cells\t:\t{time()-s_e:.3f} seconds")

        # s_e = time()
        cells,digits = ex.clear_cells(cells)
        # logging.info(f" ex.clear_cells\t:\t{time()-s_e:.3f} seconds")

        ui = ex.ui(cells)

        preds = {}
        board = [[0 for _ in range(9)] for _ in range(9)]
        for i,(c,d) in enumerate(zip(cells,digits)):
          if d:
            roi = cv.resize(c,(CELL_SIZE,CELL_SIZE))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = sudoku_model.predict(roi).argmax(axis=1)[0]

            # print(prediction)

            board[i//9][i%9] = prediction
        
        # sudoku = Sudoku()
        # sudoku.initialize_solved_board(board)
        # sudoku.solve()
        # if sudoku.unsolvable:
        #   print("unsolvable sudok")
        # else:
        #   print("solved")
        #   sudoku.show()
        #Debugging purpose lul
        sudoku = Sudoku()
        sudoku.initialize_solved_board(board)
        sudoku = SudokuDLX.SudokuDLX()
        solved = sudoku.solve(board,True)
        if solved[0]:
          orig_warp = ex.unwrap(orig,puzzle)
          orig_warp = cv.resize(orig_warp,(R_W,R_H))
          drawed_warp = di.draw(orig_warp,ex.extract_cells(orig_warp),solved[1])
          FOUND = True
          cv.imshow("draw_warp",drawed_warp)



          #REWARP HERE
          
          cv.waitKey()


        # cv.imshow("warp",np.hstack([warp,ui]))



    if di.show([orig]):
      break
    
    loop_time = time()-s_loop
    # print(f"Looped in {loop_time:.3f} secs")





  cv.destroyAllWindows()
  cap.release()



