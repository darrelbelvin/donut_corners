import pickle
import cv2
from donut_corners import DonutCorners

img = cv2.imread('images/bldg-1.jpg')
#crop
img = img[:200, 650:950]

dc = DonutCorners(img)
dc.score_all()

pickle.dump( dc, open( "save.p", "wb" ) )