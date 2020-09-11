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

  def draw(self,warp,cells,result):
    
    step = len(warp)//9

    c = int(step/4)
    for j in range(9):
      for i in range(9):
        cells[j*9+i] = cv.putText(cells[j*9+i],str(result[j][i]),(c,int(3*c)),cv.FONT_HERSHEY_SIMPLEX,.75,(255,0,0,125),1,cv.LINE_AA)
        warp[step*j:step*(j+1),step*i:step*(i+1)]
    return warp

