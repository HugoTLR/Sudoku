import cv2 as cv
import numpy as np 

class Display:
  RED = (0,0,255)
  GREEN = (0,255,0)
  BLUE = (255,0,0)
  def __init__(self,LOOP_DELAY=1,h=0,w=0):
    self.delay = LOOP_DELAY
    self.h = h
    self.w = w

  def show(self,**kwargs):
    for i,stack in enumerate(kwargs.values()):
      if not stack:
        continue
      cv.imshow(f"Stack {i}",np.hstack(stack))
    return cv.waitKey(self.delay) == ord('q') & 0xFF

  #Draw Sudoku UI
  def draw(self,warp,cells,result):
    
    step = len(warp)//9

    c = int(step/4)
    for j in range(9):
      for i in range(9):
        cells[j*9+i] = cv.putText(cells[j*9+i],str(result[j][i]),(c,int(3*c)),cv.FONT_HERSHEY_SIMPLEX,.75,(255,0,0,125),1.25,cv.LINE_AA)
        warp[step*j:step*(j+1),step*i:step*(i+1)] = cells[j*9+i]
    return warp

  def build_sudoku_ui(self,pattern,H,W,result):
    step = H//9
    color = (255,125,0)
    ui = np.zeros((H,W),dtype=np.uint8)
    pattern = 255 - cv.resize(pattern,(H,W),cv.INTER_NEAREST )
    ui = cv.bitwise_or(ui,pattern)
    c = int(step/4)
    for j in range(9):
      for i in range(9):
        cv.putText(ui[step*j:step*(j+1),step*i:step*(i+1)],str(result[j][i]),(int(1.25*c),int(3.2*c)),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3,cv.LINE_AA)
    return ui

  #Draw KP matches
  def draw_matches(self,ex,frame,matches):
    for pt1,pt2 in matches:
      x0,y0,z0 = ex.denormalize(pt1)
      x1,y1,z1 = ex.denormalize(pt2)
      frame = cv.line(frame, (x0,y0), (x1,y1), Display.BLUE,2)
      frame = cv.circle(frame,(x0,y0),3,Display.GREEN,-1)
    return frame

  def draw_homography(self,frame,warp,H,pts):
    h,w,c = warp.shape
    dst = cv.perspectiveTransform(pts, H)  
    frame = cv.polylines(frame, [np.int32(dst)], True, (0,255,0), 10, cv.LINE_AA) 
    return frame


  def render_warp(self,frame,ui,matrix,ui_pts,src_pts,dst_pts):
    # scale_matrix = np.eye(3)*3
    # src_pts = np.array([p[0] for p in src_pts])

    T = cv.getPerspectiveTransform(ui_pts,dst_pts)


    # src_pts = np.array([[p[0], p[1]] for p in src_pts])
    
    # elevated_pts = np.array([[p[0], p[1], 100] for p in src_pts])



    # dst = cv.perspectiveTransform(elevated_pts.reshape(-1, 1, 3), matrix)
    # imgpts = np.int32(dst)

    # elevated = cv.polylines(frame, [imgpts], True, (255,255,255), 3, cv.LINE_AA) 
 
    # cv.imshow("ui",ui)
    remap = cv.warpPerspective(ui,T,(frame.shape[1],frame.shape[0]))
    remap = cv.threshold(remap,125,255,cv.THRESH_BINARY)[1]

    mask = remap.copy()
    not_mask = cv.bitwise_not(mask)

    remap = cv.cvtColor(remap,cv.COLOR_GRAY2BGR)
    blank = 255 * np.ones((frame.shape[0],frame.shape[1],3),dtype=np.uint8)


    
    blue = np.zeros(frame.shape, frame.dtype)
    blue[:,:] = (255, 0, 0)
    blue_mask = blue & remap

    ui_part = cv.bitwise_and(blue_mask,blue_mask,mask=mask)
    fr_part = cv.bitwise_and(frame,frame,mask=not_mask)
    output = cv.bitwise_or(ui_part,fr_part)

    alpha = .4
    cv.addWeighted(output, alpha, frame, 1-alpha, 0, frame)

    # cv.imshow("elevate",elevated)

    return frame