import cv2
import numpy as np
from scipy import signal

from multiprocessing import Pool

class DonutCorners():
    rot90 = np.array([[0, -1], [1, 0]])

    # pylint: disable=too-many-instance-attributes
    def __init__(self, image, **kwargs):
        if isinstance(image, str):
            self.src = cv2.imread(image)
        else:
            self.src = image

        self.peaks_params = {'height':30, 'threshold':None, 'distance':None, 'prominence':None, 'width':None, 'wlen':None, 'rel_height':0.5, 'plateau_size':None}
        self.sobel_params = {'ksize':3, 'scale':1, 'delta':0, 'ddepth':cv2.CV_8U, 'borderType':cv2.BORDER_DEFAULT}

        # donut params
        self.radii = [20, 40]
        self.round = False
        self.masks = [DonutCorners.donut_mask(r, self.round) for r in self.radii]
        self.nearest = True

        # grid params
        self.grid_size = 10

        self.__dict__.update(kwargs)
        self.scored = None
        
        self.preprocess()

    def preprocess(self):
        find_peaks = signal.find_peaks

        edges_x = cv2.Sobel(self.src, dx=1, dy=0, **self.sobel_params)
        edges_y = cv2.Sobel(self.src, dx=0, dy=1, **self.sobel_params)

        edges_x_max = np.max(edges_x, axis=-1)
        edges_y_max = np.max(edges_y, axis=-1)
        self.slopes = np.stack((edges_y_max, edges_x_max), axis=-1)
        
        interest_x = np.zeros(edges_x_max.shape, dtype=bool)
        interest_y = interest_x.copy()

        for edges, interest in ((edges_y_max, interest_y), (edges_x_max, interest_x)):
            for data, bools in zip(edges, interest):
                peaks = find_peaks(data, **self.peaks_params)
                bools[peaks[0].astype(int)] = True
        
        self.interest = np.stack((interest_y, interest_x), axis=-1)


    def score_point(self, point):
        dozen = [self.bake_donut(point, mask) for mask in self.masks]
        return sum([self.score_donut(donut) for donut in dozen])


    def bake_donut(self, point, mask):
        # identify points of interest
        m2 = self.clip_mask(mask + point)
        pois = self.get_pois(m2)
        #pois = m2[np.nonzero(np.any(dc.interest[m2[:,0],m2[:,1]], axis=-1))]

        # trace rays from each point
        rays = np.array([self.get_ray(p, point) for p in pois])

        profiles = np.array([self.profile_ray(ray) for ray in rays])

        # find the strength of each ray
        strengths = np.array([DonutCorners.score_ray(profile) for profile in profiles])
        
        return rays, profiles, strengths, m2


    def score_donut(self, donut):
        rays, profiles, strengths, mask = donut
        if len(strengths) > 4:
            return sum(np.partition(strengths, -4)[-4:])
        else:
            return sum(strengths)


    def get_pois(self, mask):
        return mask[np.nonzero(np.any(self.interest[mask[:,0],mask[:,1]], axis=-1))]


    @classmethod
    def donut_mask(cls, radius, round=False):
        if round:
            pass

        d = 1+radius * 2
        mask = np.empty((d*4-4,2),dtype=int)
        edge = np.arange(d) - radius
        mask[:d, 0] = radius
        mask[:d, 1] = edge
        mask[d-1:2*d-1,0] = -edge
        mask[d-1:2*d-1,1] = radius
        mask[2*d-2:3*d-2,0] = -radius
        mask[2*d-2:3*d-2,1] = -edge
        mask[3*d-3:4*d-4,0] = edge[:-1]
        mask[3*d-3:4*d-4,1] = -radius
        return mask


    def clip_mask(self, mask):
        return mask[np.all((mask >= 0) & (mask < self.src.shape[:-1]), axis=1)]


    def get_ray(self, p_1, p_2):
        rnd = np.round
        uv = p_2 - p_1
        l = np.linalg.norm(uv)
        uv = uv/l
        perp = DonutCorners.rot90.dot(uv)

        return uv, perp, np.array([p_1 + rnd(uv*i).astype(int) for i in np.arange(1,l)])


    def profile_ray(self, ray):
        uv, perp, coords = ray
        return [perp.dot(coord) for coord in coords]


    def score_row(self, y):
        return [self.score_point([y,x]) for x in range(self.src.shape[1])]


    def score_all(self, multithread = True):
        
        if multithread:
            with Pool(7) as p:
                out = p.map(self.score_row, range(self.src.shape[0]))
        
        else:
            out = np.array([self.score_row(y) for y in range(img.shape[0])])
        
        self.scored = out
        return out


    @classmethod
    def score_ray(cls, profile):
        return np.average(np.abs(profile))


if __name__ == "__main__":
    from visualizing_donut_corners import *

    img = cv2.imread('images/bldg-1.jpg')
    #crop
    img = img[0:150, 750:850]
    pt = [50,50]

    dc = DonutCorners(img)
    
    print(img.shape)

    #import sys
    #dc.score_all('pydevd' not in sys.modules)

    #show_std(dc)
    
    print(dc.score_point(pt))
    data = list(np.ndindex(dc.src.shape[:2]))
    
    dm = paint_donut(get_2dimg(dc, 'slopes'), dc, pt, rays = True)
    show_imgs((dm, dc.src))

    #show_imgs((dc.interest[...,0], dc.slopes[...,0]))
    
    print('done')
    print('leaving')