from extractor import FeatureExtractor
from display import Display
from imutils import grab_contours
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import numpy as np
from solver import Sudoku


if __name__ == "__main__":
  SSIM_TRESHOLD = .6
  R_H,R_W = 288,288
  CELL_SIZE = 28

  model_path = "./Model/Output/optimized.h5"

  sudoku_model = load_model(model_path)

  cpt = 0 
  pat = cv.cvtColor(cv.imread("./pattern2.jpg"),cv.COLOR_BGR2GRAY)
  h,w = pat.shape
  # assert R_S * 9 == R_H and R_S * 9 == R_W, "Incorrect size"

  di = Display()
  ex = FeatureExtractor()

  cap = cv.VideoCapture(0)
  while True:
    ret,frame = cap.read()
    if not ret:
      break
    orig = frame
    frame = ex.auto_process(frame)
    canny = ex.auto_canny(frame)
    # canny = ex.canny_process(canny)


    warp = None
    cnts = cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    if cnts:
      #Retrieve corners of the contours the most "squared"
      corners = ex.grab_best_square(cnts)
      #Unwrap it
      warp = ex.unwrap(orig,corners)
       #Apply processing function
      warp = ex.prepare_warp(warp)
      #Resize
      warp = cv.resize(warp,(R_W,R_H))

      #Create a version to compare with an empty sudoku pattern
      warp_cmp = cv.resize(warp,(w,h),interpolation=cv.INTER_AREA)
      score_ssim = ssim(warp_cmp,pat, data_range=pat.max() - pat.min())
      score_ssim = score_ssim*.5 +.5
      # print(f" Ssim={score_ssim}")

      #If it looks like a sudoku
      if score_ssim > SSIM_TRESHOLD:
        cells = ex.extract_cells(warp)
        cells = ex.threshold_cells(cells)
        cells,digits = ex.clear_cells(cells)

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

            print(prediction)

            board[i//9][i%9] = prediction

        sudoku = Sudoku()
        sudoku.initialize_solved_board(board)

            #PREDICT
            # preds[i] = 





        cv.imshow("warp",np.hstack([warp,ui]))
        cv.waitKey()

      cv.drawContours(orig,[corners],0,(255,0,0),3)


    if di.show([orig]):
      break


  cv.destroyAllWindows()
  cap.release()



