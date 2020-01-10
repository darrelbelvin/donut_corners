import pickle
import cv2
from donut_corners import DonutCorners
import time

t0 = time.time()

img = cv2.imread('images/bldg-1.jpg')
#crop
img = img[:200, 650:950]

print(f'Load image time: {time.time() - t0:.2f} seconds')
t0 = time.time()

dc = DonutCorners(img)

print(f'Init time: {time.time() - t0:.2f} seconds')
t0 = time.time()

dc.find_corners()

print(f'Find corners time: {time.time() - t0:.2f} seconds')
t0 = time.time()

dc.score_all()

print(f'Score all time: {time.time() - t0:.2f} seconds')
t0 = time.time()

pickle.dump( dc, open( "save.p", "wb" ) )

print(f'Pickle time: {time.time() - t0:.2f} seconds')