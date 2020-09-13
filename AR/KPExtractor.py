#Classes
from cv2 import ORB_create
from cv2 import BFMatcher
from cv2 import KeyPoint
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
#Functions
from cv2 import goodFeaturesToTrack
from cv2 import findHomography
from cv2 import rectangle
from math import sqrt
from numpy import array
from numpy import concatenate, cross
from numpy import dot
from numpy import float32
from numpy import mat
from numpy import ones
from numpy import reshape
from numpy import stack,sum
from numpy import zeros
from numpy.linalg import det,inv,norm,svd
from skimage.measure import ransac
#Parameters
from cv2 import NORM_HAMMING
from cv2 import RANSAC
from numpy import uint8




class KPExtractor(object):
  def __init__(self,K,nfeatures=5000,norm=NORM_HAMMING,crosscheck=False):
    self.orb = ORB_create(nfeatures)
    self.bfm = BFMatcher(norm,crosscheck)
    self.orb_mask = None
    self.last = None
    self.K = K
    self.Kinv = inv(self.K)

    self.f_est_avg = []

  def normalize(self,pts):
    return dot(self.Kinv,self.add_ones(pts).T).T[:, 0:2]

  def denormalize(self,pt):
    ret = dot(self.K, array([pt[0],pt[1],1.0]).T)
    ret /= ret[2]
    return int(round(ret[0])),int(round(ret[1])),int(round(ret[2]))

  def build_orb_mask(self,frame):
    h,w= frame.shape
    c = 1
    self.orb_mask = zeros((h,w,c),dtype=uint8)
    rectangle(self.orb_mask,(0,0),(w,int(h*0.75)),(255,255,255),-1)
    # self.orb_mask = cvtColor(self.orb_mask,COLOR_BGR2GRAY)

  def extractRt(self,E):
    U,w,Vt = svd(E)
    assert det(U) > 0
    if det(Vt) < 0:
      Vt *= -1.0

    #Find R and T from Hartleyy and Zisserman
    W = mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    R = dot(dot(U,W),Vt)
    if sum(R.diagonal()) < 0:
      R = dot(dot(U,W.T),Vt)
    t = U[:,2]
    Rt = concatenate([R,t.reshape(3,1)],axis=1)
    return Rt
      
  # [x,y] to [x,y,1]
  def add_ones(self,x):
    return concatenate([x, ones((x.shape[0],1))],axis=1)

  def extract(self,frame,maxCorners=5000,qualityLevel=.01,minDistance=3):
    #Detect
    feats = goodFeaturesToTrack(frame,maxCorners,qualityLevel,minDistance)#,mask=self.orb_mask)
    #Extract
    kps = [KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in feats]
    kps,des = self.orb.compute(frame,kps)

    
    #Match
    ret = []
    if self.last is not None:
      #THE QUERY IMAGE IS THE ACTUAL IMAGE &
      #THE TRAIN IMAGE IS THE LAST IMAGE
      #U STOUPID KUNT
      matches = self.bfm.knnMatch(des,self.last['des'],k=2)
      for m,n in matches:
        if m.distance < 0.7 * n.distance:
          ret.append((kps[m.queryIdx].pt,self.last['kps'][m.trainIdx].pt))



    #filter   
    Rt = None
    if len(ret) > 0:
      ret = array(ret)
      #Normalize
      ret[:,0,:] = self.normalize(ret[:,0,:])
      ret[:,1,:] = self.normalize(ret[:,1,:])

      try:
        # print(f"{len(ret)=}, {ret[:,0].shape=}, {ret[:,1].shape=}")
        model, inliers = ransac((ret[:, 0],ret[:, 1]),
                                  #FundamentalMatrixTransform,
                                  EssentialMatrixTransform,
                                  min_samples=8,
                                  residual_threshold=0.005,
                                  max_trials=200)

        ret = ret[inliers]

        Rt = self.extractRt(model.params)
      except:
        pass

    self.last = {'kps': kps,'des':des}
    return ret,Rt


  #warp is pattern for now
  def homography(self,last_puzzle,puzzle,warp,maxCorners=5000,qualityLevel=.01,minDistance=3):
    #Detect
    f_feats = goodFeaturesToTrack(frame,maxCorners,qualityLevel,minDistance)#,mask=self.orb_mask)
    #Extract
    f_kps = [KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in f_feats]
    f_kps,f_des = self.orb.compute(frame,f_kps)

    #Detect
    w_feats = goodFeaturesToTrack(warp,maxCorners,qualityLevel,minDistance)#,mask=self.orb_mask)
    #Extract
    w_kps = [KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in w_feats]
    w_kps,w_des = self.orb.compute(warp,w_kps)

    #Match
    ret = []
    
    # #THE QUERY IMAGE IS THE ACTUAL IMAGE &
    # #THE TRAIN IMAGE IS THE LAST IMAGE
    # #U STOUPID KUNT
    matches = self.bfm.knnMatch(w_des,f_des,k=2)
    for m,n in matches:
      # if m.distance <= 1 * n.distance:
      ret.append((w_kps[m.queryIdx].pt,f_kps[m.trainIdx].pt))

    #filter   
    Rt = None
    H, mask, warp_pts, orig_pts = None,None,None,None
    if len(ret) > 0:
        warp_pts = float32([r[0] for r in ret]).reshape(-1,1,2)
        orig_pts = float32([r[1] for r in ret]).reshape(-1,1,2)
        H,mask = findHomography(warp_pts,orig_pts,RANSAC,5.0)


    return H,mask,warp_pts,orig_pts

  def project_matrix(self,homography):
    homography *= (-1)
    rot_and_transl = dot(self.Kinv, homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = sqrt(norm(col_1, 2) * norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = cross(rot_1, rot_2)
    d = cross(c, p)
    rot_1 = dot(c / norm(c, 2) + d / norm(d, 2), 1 / sqrt(2))
    rot_2 = dot(c / norm(c, 2) - d / norm(d, 2), 1 / sqrt(2))
    rot_3 = cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = stack((rot_1, rot_2, rot_3, translation)).T
    return dot(self.K, projection)