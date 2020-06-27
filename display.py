import cv2 as cv


class Display:
  RED = (0,0,255)
  GREEN = (0,255,0)
  BLUE = (255,0,0)
  def __init__(self,LOOP_DELAY=1,h=0,w=0):
    self.delay = LOOP_DELAY
    self.h = h
    self.w = w

  def show(self,frames):
    for i,frame in enumerate(frames):
      if frame is None:
        continue
      cv.imshow(f"Frame {i}",frame)
    return cv.waitKey(self.delay) == ord('q') & 0xFF
