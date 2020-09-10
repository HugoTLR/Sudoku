#Global import
import math
#Functions
from cv2 import adaptiveThreshold, approxPolyDP, arcLength
from cv2 import bitwise_and, bitwise_not
from cv2 import calcHist, Canny, connectedComponentsWithStats, contourArea, convertScaleAbs, convexHull
from cv2 import countNonZero, cvtColor
from cv2 import drawContours
from cv2 import filter2D, findContours
from cv2 import GaussianBlur, getStructuringElement
from cv2 import morphologyEx
from cv2 import threshold

from imutils import perspective, grab_contours

from numpy import argsort, array
from numpy import median
from numpy import min as npmin
from numpy import ptp
from numpy import zeros

from time import time


from skimage.segmentation import clear_border
#ATTR/Enum
from cv2 import ADAPTIVE_THRESH_GAUSSIAN_C
from cv2 import CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY
from cv2 import MORPH_GRADIENT, MORPH_RECT
from cv2 import RETR_EXTERNAL
from cv2 import THRESH_BINARY, THRESH_BINARY_INV, THRESH_OTSU
from numpy import uint8

import logging

class FeatureExtractor:

  Kernel = array([[-1,1,-1], [1,1,1], [-1,1,-1]])
  Canny_Kernel = getStructuringElement(MORPH_RECT,(3,3))
  def __init__(self):
    pass

  def auto_process(self,frame):
    frame = cvtColor(frame,COLOR_BGR2GRAY)
    frame = self.auto_constrast(frame)
    frame = GaussianBlur(frame, (7, 7), 3)
    # frame = filter2D(frame,-1,self.Kernel)
    return frame

  def find_puzzle(self,frame):
    puzzle_cnt = None
    s_ee = time()
    thresh = adaptiveThreshold(frame, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2)
    thresh = bitwise_not(thresh)
    logging.info(f" \tself.find_puzzle.thresh\t:\t{time()-s_ee:.3f} seconds")
    s_ee = time()
    cnts = findContours(thresh.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=contourArea, reverse=True)
    cnts = cnts[:1000] # Limit to a thousand cnts to make the functions stable
    logging.info(f" \tself.find_puzzle.cnts\t:\t{time()-s_ee:.3f} seconds")

    for c in cnts:
      peri = arcLength(c, True)
      approx = approxPolyDP(c, .02 * peri, True)

      if len(approx) == 4:
        puzzle_cnt = approx
        break

    return puzzle_cnt, thresh
  def prepare_warp(self,warp):
    return self.auto_constrast(cvtColor(warp,COLOR_BGR2GRAY))

  def prepare_cells(self,gradient):
    cells = self.extract_cells(gradient)
    cells = self.clear_cells(cells)
    cells = self.threshold_cells(cells)
    return cells

  def auto_constrast(self,frame):
    histSize = 256
    clipHistPercent = 1
    hist = calcHist([frame],[0],None,[256],[0,256])
    
    accumulator = []
    app = accumulator.append
    app(hist[0])

    for i in range(1,256):
        app(accumulator[i-1] + hist[i])

    maxVal = accumulator[-1]
    clipHistPercent = clipHistPercent * (maxVal / 100.0)
    clipHistPercent = clipHistPercent / 2.0

    minGray = 0
    while accumulator[minGray] < clipHistPercent:
        minGray = minGray + 1

    maxGray = 255
    while accumulator[maxGray] >= (maxVal - clipHistPercent):
        maxGray = maxGray - 1

    inputRange = maxGray - minGray
    if inputRange == 0:
      return frame
    alpha = 255 / inputRange
    beta = -minGray *alpha
    adjusted = convertScaleAbs(frame,alpha=alpha,beta=beta)
    return adjusted


  def ui(self,cells):
    assert len(cells) == 81, "Invalid number of cells"
    h,w = cells[0].shape

    ui = zeros((w*9,h*9),dtype=uint8)
    for j in range(9):
      for i in range(9):
        ui[j*h:(j+1)*h,i*w:(i+1)*w] = cells[j*9+i]
    return ui

  def select_best_components(self,ui,min_size = 275):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = connectedComponentsWithStats(ui,8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    # min_size = 275  

    #your answer image
    img2 = zeros((output.shape),uint8)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

  

  def corner_points(self,mask):
    points = []
    epsilon = .1
    cpt = 0

    approx = approxPolyDP(mask,epsilon,True)
    cpt = 0
    while cpt < 60:
      si = len(approx)
      if si < 4:
        epsilon /= 1.1
      elif si > 4:
        epsilon *= 1.1
      else:
        break
      approx = approxPolyDP(mask,epsilon,True)
      cpt += 1
    if len(approx) != 4:
      return array( [(0,0),(0,1),(1,0),(1,1)])
    return array([tuple(a[0]) for a in approx])

  ##TOP LEFT, TOP RIGHT,BOT RIGHT,BOT LEFT  not exactly qr but almost
  def order_points(self,pts):

    # sort the points based on their x-coordinates
    #print("POINTS : {}".format(pts))
    xSorted = pts[argsort(pts[:, 0]), :]
   
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
   
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
   
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point

    ##WORKNG BETTER ON TRAPEZOIDAL QUAD
    rightMost = rightMost[argsort(rightMost[:,1]),:]
    (tr,br) = rightMost
   
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return array([tl, tr, br, bl], dtype="float32")

  ####Automatic canny edge detection
  def auto_canny(self,image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = Canny(image, lower, upper)
 
    # return the edged image
    return edged

  def canny_process(self,canny):
    canny = morphologyEx(canny,MORPH_GRADIENT,self.Canny_Kernel,iterations=1)
    return canny

  
  def get_best_contours(cnts):

    best_score = 0
    best_corners = None

    for i,cnt in enumerate(cnts):
      hull = convexHull(cnt)
      corners = self.corner_points(hull)

      air = contourArea(corners)
      perimeter = arcLength(corners,True)

      #Square property u know...
      estimate_seg_size = perimeter/4 
      estimate_air = estimate_seg_size**2

      error = abs(1-air/estimate_air)
      if error < .1:
        score = air*((1-error)**2)
        if score > best_score:
          best_score = score
          best_corners = corners

    return best_corners

  
  def cmp_to_square(self,quad):
    tl,tr,br,bl = quad[0], quad[1], quad[2], quad[3]
    dist_d1 = self.dist(tl,br)
    dist_d2 = self.dist(tr,bl)
    dist_diff = abs(dist_d1-dist_d2)

    center_d1 = self.center(tl,br)
    center_d2 = self.center(tr,bl)
    dist_center = self.dist(center_d1,center_d2)


    angle_in_degree = self.get_angle(tl,center_d1,tr)
    diff_angle = abs(90-angle_in_degree)
    # print(f"D1: {tl}-{br}\tC1:{center_d1}")
    # print(f"D2: {tr}-{bl}\tC2:{center_d2}")
    # print(f"Diag Diff: {dist_diff}\tCenter Diff:{dist_center}\tangle: {angle_in_degree}")

    return dist_diff,dist_center,diff_angle

  def dist(self,p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

  def center(self,p1,p2):
    return [int(round((p1[0]+p2[0])/2)),int(round((p1[1]+p2[1])/2))]

  def get_angle(self,a,b,c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

  def clear_cell(self,cell):
    cnts = findContours(cell, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
      return (zeros(cell.shape, dtype="uint8"),False)

    c = max(cnts, key=contourArea)
    mask = zeros(cell.shape, dtype="uint8")
    drawContours(mask, [c], -1, 255, -1)
    (h,w) = cell.shape

    pct_filled = countNonZero(mask) / float(w*h)
    if pct_filled < .03:
      return (zeros(cell.shape, dtype="uint8"),False)

    cell = bitwise_and(cell,cell,mask=mask) 
    return (cell,True)

  def extract_cells(self,warp):
    h,w = warp.shape
    assert len(warp)%9 == 0, "extract_cells(): H%9 != 0"
    assert len(warp[0])%9 == 0, "extract_cells(): W%9 != 0"
    assert len(warp) == len(warp[0]), "extract_cells(): W != H"

    step = len(warp)//9
    cells = []
    for j in range(0,len(warp),step):
      for i in range(0,len(warp),step):
        roi = warp[j:j+step,i:i+step]


        cells.append(roi)

    return cells

  def clear_cells(self,cells):
    cleared_cells, digits = [],[]
    for cell in cells:
      cl,digit = self.clear_cell(cell)
      cleared_cells.append(cl)
      digits.append(digit)
    return cleared_cells,digits

  def threshold_cells(self,cells):
    return [clear_border(threshold(c,0,255,THRESH_BINARY_INV | THRESH_OTSU)[1]) for c in cells]

  def grab_best_square(self,cnts):
    airs,sim_d,sim_c,sim_a = [], [], [], []
    squares_similarity = []
    if len(cnts) > 1000:
      cnts = sorted(cnts,key=contourArea,reverse=True)[:1000]

    s_ee = time()
    hulls = [convexHull(c) for c in cnts]
    logging.info(f" \tself.grab_best_square.hulls\t:\t{time()-s_ee:.3f} seconds")

    s_ee = time()
    corners = [self.order_points(self.corner_points(h)) for h in hulls]
    logging.info(f" \tself.grab_best_square.corners\t:\t{time()-s_ee:.3f} seconds")

    for i,quad in enumerate(corners):
      airs.append(contourArea(quad))
      dist_diff,dist_center,dist_angle = self.cmp_to_square(quad)

      sim_d.append(dist_diff)
      sim_c.append(dist_center)
      sim_a.append(dist_angle)
      squares_similarity.append(dist_diff+dist_center+dist_angle) #Maybe not the best formula to sum them up

    airs = (airs-npmin(airs)) / ptp(airs)
    squares_similarity = (squares_similarity-npmin(squares_similarity) )/ ptp(squares_similarity)
    sim_d = (sim_d-npmin(sim_d) )/ ptp(sim_d)
    sim_c = (sim_c-npmin(sim_c) )/ ptp(sim_c)
    sim_a = (sim_a-npmin(sim_a) )/ ptp(sim_a)

    scores = []
    for air,sq in zip(airs,squares_similarity):
      scores.append(air*(1-sq))

    best = [(int(c[0]),int(c[1])) for c in corners[scores.index(max(scores))]]
    return array(best)

  def is_black_bg(self,frame):
    _,th = threshold(frame,0,255,THRESH_BINARY+THRESH_OTSU)
    h,w = th.shape
    total_px = h*w
    return countNonZero(th) > total_px/2 

  def unwrap(self,frame,corners):
    warp = perspective.four_point_transform(frame,corners.reshape(4,2))
    return warp
