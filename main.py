from extractor import FeatureExtractor
from display import Display
import cv2 as cv
import numpy as np
if __name__ == "__main__":

  di = Display()
  ex = FeatureExtractor()

  cap = cv.VideoCapture(0)
  while True:
    ret,frame = cap.read()
    if not ret:
      break
    orig = frame
    frame = ex.auto_process(frame)
    # if ex.is_black_bg(frame):
    #   print("BlackBG")
    #   frame = cv.blur(frame,(11,11))
    canny = ex.auto_canny(frame)
    canny = ex.canny_process(canny)

    warp = None
    _, cnts, hier = cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if cnts:
      corners = ex.grab_best_square(cnts)
      warp = ex.unwrap(orig,corners)
      cv.drawContours(orig,[corners],0,(255,0,0),3)


    if di.show([orig,warp]):
      break


  cv.destroyAllWindows()
  cap.release()



