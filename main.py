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


if __name__ == "__main__":
  #Const
  SSIM_TRESHOLD = .6
  R_H,R_W = 288,288
  CELL_SIZE = 28

  # Load Classification model
  # model_path = "./Model/Output/optimized.h5"
  model_path = "./Model/Output/tmp/cp-38-0.0509.h5"
  sudoku_model = load_model(model_path)

  # Load Sudoku pattern to make sure it's a sudoku we've detected
  pat = cv.cvtColor(cv.imread("./pattern2.jpg"),cv.COLOR_BGR2GRAY)
  h,w = pat.shape

  #Instanciate Displayer and Extractor
  di = Display()
  ex = FeatureExtractor()

  FOUND = False

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
      warp_thresh = ex.unwrap(thresh,puzzle)
      warp = ex.unwrap(gray,puzzle)

      #Prepare for SSIM comparison
      warp_cmp = cv.bitwise_not(warp_thresh)
      warp_cmp = cv.resize(warp_cmp,(w,h),interpolation=cv.INTER_AREA)

      #Compare and rescale score to 0..1
      score_ssim = ssim(warp_cmp,pat, data_range=pat.max() - pat.min())
      score_ssim = score_ssim*.5 +.5


      #If it looks like a sudoku
      if score_ssim > SSIM_TRESHOLD:
        cv.drawContours(orig,[puzzle],0,(255,0,0),3)
        #If we alread found and decoded it skip
        if FOUND: continue

        #Resize to Classifier input size
        warp = cv.resize(warp,(R_W,R_H))

        #Retrieve Sudoku cells
        cells = ex.extract_cells(warp)
        cells = ex.threshold_cells(cells)
        cells,digits = ex.clear_cells(cells)

        #Ui for debugging purpose
        ui = ex.ui(cells)

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
          orig_warp = ex.unwrap(orig,puzzle)
          orig_warp = cv.resize(orig_warp,(R_W,R_H))
          drawed_warp = di.draw(orig_warp,ex.extract_cells(orig_warp),solved[1])
          FOUND = True
          cv.imshow("draw_warp",drawed_warp)



          #TODO: AR Part here
          
          cv.waitKey()

        # Debug
        # cv.imshow("warp",np.hstack([warp,ui]))



    if di.show([orig]):
      break
    
    loop_time = time()-s_loop
    # print(f"Looped in {loop_time:.3f} secs")





  cv.destroyAllWindows()
  cap.release()



