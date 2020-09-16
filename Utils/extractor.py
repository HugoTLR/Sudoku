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
from numpy import newaxis

from time import time
from scipy.spatial import distance as dist

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
  MAX_CNTS = 1000
  def __init__(self):
    pass

  def auto_process(self,frame,blur = True):
    frame = cvtColor(frame,COLOR_BGR2GRAY)
    frame = self.auto_constrast(frame)
    if blur:
      frame = GaussianBlur(frame, (7, 7), 3)
    return frame

  def prepare_warp(self,warp):
    return self.auto_constrast(cvtColor(warp,COLOR_BGR2GRAY))

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

  def order_points(self,pts):
    return perspective.order_points(pts)
  ##TOP LEFT, TOP RIGHT,BOT RIGHT,BOT LEFT
  # def order_points(self,pts):
  #   # sort the points based on their x-coordinates
  #   xSorted = pts[argsort(pts[:, 0]), :]
   
  #   # grab the left-most and right-most points from the sorted
  #   # x-roodinate points
  #   leftMost = xSorted[:2, :]
  #   rightMost = xSorted[2:, :]
   
  #   # now, sort the left-most coordinates according to their
  #   # y-coordinates so we can grab the top-left and bottom-left
  #   # points, respectively
  #   leftMost = leftMost[argsort(leftMost[:, 1]), :]
  #   (tl, bl) = leftMost
  #   print(f"{tl=}")
  #   print(f"{bl=}")
   
  #   # now that we have the top-left coordinate, use it as an
  #   # anchor to calculate the Euclidean distance between the
  #   # top-left and right-most points; by the Pythagorean
  #   # theorem, the point with the largest distance will be
  #   # our bottom-right point

  #   ##WORKNG BETTER ON TRAPEZOIDAL QUAD
  #   # rightMost = rightMost[argsort(rightMost[:,1]),:]
  #   # (tr,br) = rightMost
  #   D = dist.cdist(tl[newaxis],rightMost,'euclidean')[0]
  #   (br,tr) = rightMost[argsort(D)[::-1],:]

  #   # return the coordinates in top-left, top-right,
  #   # bottom-right, and bottom-left order
  #   return array([tl, tr, br, bl], dtype="float32")

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

  def extract_cells(self,warp):
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

  def threshold_cells(self,cells):
    return [clear_border(threshold(c,0,255,THRESH_BINARY_INV | THRESH_OTSU)[1]) for c in cells]

  def find_puzzle(self,frame):
    puzzle_cnt = None
    thresh = adaptiveThreshold(frame, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2)
    thresh = bitwise_not(thresh)
    cnts = findContours(thresh.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=contourArea, reverse=True)
    cnts = cnts[:FeatureExtractor.MAX_CNTS] # Limit to a thousand cnts to make the functions stable

    for c in cnts:
      peri = arcLength(c, True)
      approx = approxPolyDP(c, .02 * peri, True)
      if len(approx) == 4:
        puzzle_cnt = approx
        break
    return puzzle_cnt, thresh

  def unwrap(self,frame,corners):
    warp = perspective.four_point_transform(frame,corners.reshape(4,2))
    return warp
