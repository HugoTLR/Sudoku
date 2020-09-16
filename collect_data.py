import cv2 as cv
import numpy as np
import glob


from Utils.display import Display
from Utils.extractor import FeatureExtractor
from Utils.tracker import Tracker
from time import time
# from tensorflow.keras.preprocessing.image import img_to_array

if __name__ == "__main__":
  CELL_SIZE = 28
  R_H,R_W = CELL_SIZE*9,CELL_SIZE*9

  W,H = 640,480

  fn_pattern = "./pattern2.jpg"

  collect_folder = "./Model/Data/test/"
  nb_files = len(glob.glob(f"{collect_folder}*.jpg"))


  #Instanciate Classes
  di = Display()
  ex = FeatureExtractor()
  tr = Tracker(fn_pattern)


  debug = None
  


  original_images_folder = "./images/"
  images = [cv.resize(cv.imread(f),(W,H)) for f in glob.glob(f"{original_images_folder}*.jpg")]
  descriptors = [open(f,'r').readlines() for f in glob.glob(f"{original_images_folder}*.dat")]
  nb_files = [len(glob.glob(f"{collect_folder}{f}/*.jpg")) for f in range(10) ]
  print(f"{len(images)=}, {len(descriptors)=}, {nb_files=}")



  for orig,des in zip(images,descriptors):
    des = des[2:]
    board = []
    for row in des:
      board.append([int(r) for r in row if r != ' ' and r != '\n'])
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

        #COLLECT
        warp = cv.resize(warp,(R_W,R_H))

        #Retrieve Sudoku cells
        cells = ex.extract_cells(warp)
        cells = ex.threshold_cells(cells)
        cells,digits = ex.clear_cells(cells)

        for i,(c,d) in enumerate(zip(cells,digits)):
          if digits:
            roi = cv.resize(c,(CELL_SIZE,CELL_SIZE))
            value = board[i//9][i%9]
            fn = f"{collect_folder}{value}/{nb_files[value]:04d}.jpg"
            print(fn)
            nb_files[value] += 1
            cv.imwrite(fn,roi)


        #Ui for debugging purpose
        debug = ex.ui(cells)






  cv.destroyAllWindows()