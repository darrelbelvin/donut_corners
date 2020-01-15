import cv2
import numpy as np
from scipy import signal, optimize
#from scipy.optimize import basinhopping

from multiprocessing import Pool, cpu_count
from math import pi, atan2
import random

class DonutCorners():
    rot90 = np.array([[0, -1], [1, 0]])

    # pylint: disable=too-many-instance-attributes
    def __init__(self, image, **kwargs):
        
        # passed on params
        self.peaks_params = {'height':30, 'threshold':None, 'distance':None, 'prominence':None, 'width':None, 'wlen':None, 'rel_height':0.5, 'plateau_size':None}
        self.sobel_params = {'ksize':3, 'scale':1, 'delta':0, 'ddepth':cv2.CV_32F, 'borderType':cv2.BORDER_DEFAULT}

        # donut params
        self.radii = [15, 20]
        self.round = False
        self.msk0, self.mask_splits = DonutCorners.donut_mask(self.radii, self.round)

        # grid params
        self.grid_size = 20

        self.__dict__.update(kwargs)

        # data init
        if isinstance(image, str):
            self.src = cv2.imread(image)
        else:
            self.src = image
        self.dims = np.array(self.src.shape[:2], dtype=int)

        self.scored = None
        self.scored_partial = np.empty(self.dims)
        self.scored_partial[:] = np.NaN

        self.zones = np.empty(self.dims, dtype=int)
        self.zones[:] = -1
        self.zones_mask = np.concatenate([[[0,0]],DonutCorners.donut_mask(list(range(3)))[0]]) 

        self.corners = []
        
        self.preprocess()

    def preprocess(self):
        find_peaks = signal.find_peaks

        edges_x = cv2.Sobel(self.src, dx=1, dy=0, **self.sobel_params)
        edges_y = cv2.Sobel(self.src, dx=0, dy=1, **self.sobel_params)

        def absmaxND(a, axis=None):
            amax = a.max(axis)
            amin = a.min(axis)
            return np.where(-amin > amax, amin, amax)

        edges_x_max = absmaxND(edges_x, axis=-1)
        edges_y_max = absmaxND(edges_y, axis=-1)
        # edges_x_max = np.max(edges_x, axis=-1)
        # edges_y_max = np.max(edges_y, axis=-1)
        self.slopes = np.stack((edges_y_max, edges_x_max), axis=-1)
        
        interest_x = np.zeros(edges_x_max.shape, dtype=bool)
        interest_y = interest_x.copy()

        for edges, interest in ((edges_y_max, interest_y), (edges_x_max, interest_x)):
            for data, bools in zip(edges, interest):
                peaks = find_peaks(np.abs(data), **self.peaks_params)
                bools[peaks[0].astype(int)] = True
        
        self.interest = np.stack((interest_y, interest_x), axis=-1)

    # information gathering methods
    def bake_donut(self, point):
        # identify points of interest
        mask = self.clip_mask(self.msk0 + np.array(point, dtype=int))
        pois = self.get_pois(mask)

        # trace rays from each point
        rays = np.array([self.get_ray(p, point) for p in pois])

        profiles = np.array([self.profile_ray(ray) for ray in rays])

        # find the strength of each ray
        strengths = np.array([DonutCorners.score_ray(profile) for profile in profiles])

        angles = np.array([atan2(ray[0][0], ray[0][1]) for ray in rays])

        #topids = np.concatenate(self.get_top_rays(strengths, angles), dtype=int)
        topids = np.concatenate([self.get_top_rays(s, a) for s, a in \
             zip(np.split(strengths, self.mask_splits), np.split(angles, self.mask_splits))]).astype(int)

        return rays, profiles, strengths, angles, mask, topids


    @classmethod
    def donut_mask(cls, radii, round=False):
        if round:
            raise NotImplementedError
        
        mask = []

        for radius in radii:
            d = 1+radius * 2
            ring = np.empty((d*4-4,2),dtype=int)
            edge = np.arange(d) - radius
            ring[:d, 0] = radius
            ring[:d, 1] = edge
            ring[d-1:2*d-1,0] = -edge
            ring[d-1:2*d-1,1] = radius
            ring[2*d-2:3*d-2,0] = -radius
            ring[2*d-2:3*d-2,1] = -edge
            ring[3*d-3:4*d-4,0] = edge[:-1]
            ring[3*d-3:4*d-4,1] = -radius
            mask.append(ring)
        
        splits = np.cumsum([len(ring) for ring in mask[:-1]])
        
        return np.concatenate(mask), splits


    def clip_mask(self, mask):
        return mask[np.all((mask >= 0) & (mask < self.src.shape[:-1]), axis=1)]


    def get_pois(self, mask):
        return mask[np.nonzero(np.any(self.interest[mask[:,0],mask[:,1]], axis=-1))]


    def get_ray(self, p_1, p_2):
        rnd = np.round
        uv = p_2 - p_1
        l = np.linalg.norm(uv)
        uv = uv/l
        perp = DonutCorners.rot90.dot(uv)

        return uv, perp, np.array([p_1 + rnd(uv*i).astype(int) for i in np.arange(1,l)])


    def profile_ray(self, ray):
        uv, perp, coords = ray
        return [perp.dot(self.slopes[coord[0],coord[1]]) for coord in coords]

    # scoring methods
    def get_score(self, point):
        point = np.array(point, dtype=int)

        if not np.all((point >= 0) & (point < self.src.shape[:-1])):
            return 0
        if self.scored is not None:
            return self.scored[point[0],point[1]]
        
        if np.isnan(self.scored_partial[point[0],point[1]]):
            self.scored_partial[point[0],point[1]] = self.score_point(point)
        
        return self.scored_partial[point[0],point[1]]


    def score_point(self, point):
        return self.score_donut(self.bake_donut(point))


    def score_donut(self, donut):
        rays, profiles, strengths, angles, mask, topids = donut
        return np.mean(strengths[topids]) if len(topids) > 0 else 0


    def get_top_rays(self, strengths, angles, n = 3, width = 0.4, double_ended = True):
        if len(strengths) == 0:
            return []
        
        c = 1 if double_ended else 2
        strengths = strengths.copy()
        out = []

        top = np.argmax(strengths)

        while len(out) < n and strengths[top] > 0:
            out.append(top)
            diffs = (angles - angles[top]) % (c*pi)
            strengths[np.argwhere((diffs < width) | (diffs > (c*pi)-width))] = 0
            top = np.argmax(strengths)

        return out

    @classmethod
    def score_ray(cls, profile):
        return np.mean(np.abs(profile))
    

    def score_row(self, y):
        return [self.score_point([y,x]) for x in range(self.src.shape[1])]


    def score_all(self, multithread = True):
        
        if multithread:
            with Pool(cpu_count() - 1) as p:
                out = p.map(self.score_row, range(self.src.shape[0]))
        
        else:
            out = [self.score_row(y) for y in range(self.src.shape[0])]
        
        out = np.array(out)
        
        self.scored = out
        return out
        
        
    def donut_slider(self, point, ray_len = 5, max_iters = 50, min_score = 100):
        axis_def = np.array([[1,1],[1,-1]])
        corner_id = len(self.corners)

        for _ in range(max_iters):
            # print(point, self.get_score(point))
            # if np.array_equal(point, [10, 90]):
            #     print('here')

            if self.zones[point[0], point[1]] not in [-1, corner_id]:
                self.zones[self.zones == corner_id] = self.zones[point[0], point[1]]
                return self.corners[self.zones[point[0], point[1]]]

            mask = self.clip_mask(self.zones_mask + point.astype(int))
            self.zones[mask[:,0],mask[:,1]] = corner_id

            rays, profiles, strengths, angles, mask, topids = self.bake_donut(point) 

            if len(topids) == 0:
                axis = axis_def
            else:
                #uv, perp, coords = ray
                axis = np.array([ray[0] for ray in rays[topids]])
            
            scores = np.array([[self.get_score(np.round(point + ax * i).astype(int)) for i in range(-ray_len, ray_len + 1)] for ax in axis])

            if np.max(scores) == 0:
                point = point + ray_len * random.choice((2, -2)) * axis[random.randint(0,axis.shape[0]-1)]
                point = np.minimum(np.maximum(point,[0,0]),self.dims-1)
                continue

            cscore = scores[0,ray_len]
            scores[:,ray_len] = 0

            arg = np.unravel_index(np.argmax(scores), shape = scores.shape)
            
            if cscore > scores[arg]:
                if cscore > min_score:
                    self.corners.append(point)
                    return point
                
                self.corners.append([-1,-1])
                return [-1,-1]


            lr = arg[1] - ray_len

            if abs(lr) == ray_len:
                point = np.round(point + (2 * lr - 1.5) * axis[arg[0]]).astype(int)
                point = np.minimum(np.maximum(point,[0,0]),self.dims-1)
            else:
                point = np.round(point + lr * axis[arg[0]]).astype(int)
        
        self.corners.append([-1,-1])
        return [-1,-1]
    

    def find_corners(self):
        negative = lambda *args: -1 * self.get_score(*args)

        for point in np.swapaxes(np.mgrid[self.grid_size//2:self.dims[0]:self.grid_size,
                    self.grid_size//2:self.dims[1]:self.grid_size], 0,2).reshape(-1,2):
            
            result = optimize.minimize(negative, np.array(point, dtype=int), method='Nelder-Mead', tol=0.1,
                            options={'initial_simplex':np.array([point, point - self.grid_size//2, point - [0,self.grid_size//2]])})
            self.corners.append(result['x'].astype(int))
            


if __name__ == "__main__":
    from visualizing_donut_corners import *

    img = cv2.imread('images/bldg-1.jpg')
    #crop
    #img = img[:200, 650:950]
    img = img[25:125, 750:850]
    
    dc = DonutCorners(img)
    
    print(dc.score_point(np.array([100,100])))
    import sys


    #dc.score_all('pydevd' not in sys.modules)
    
    dc.find_corners()

    sc = np.nan_to_num(dc.scored_partial, nan=-0.5*np.max(np.nan_to_num(dc.scored_partial)))
    sc = sc / np.max(sc) * 255
    sc = np.pad(sc[...,None], ((0,0),(0,0),(0,2)), mode='constant').astype(int)

    show_img(paint_zones(paint_corners(np.maximum(dc.src[...,[2,1,0]], sc), dc), dc))
    show_img(sc)
    show_img(paint_corners(sc, dc))

    #import sys
    #dc.score_all('pydevd' not in sys.modules)

    #show_std(dc)
    
    #print(dc.score_point(pt))strongest several
    #data = list(np.ndindex(dc.dims))
    
    #dm = paint_donut(get_2dimg(dc, 'slopes'), dc, pt, rays = True)
    #show_imgs((dm, dc.src))

    #show_imgs((dc.interest[...,0], dc.slopes[...,0]))
    
    print('done')
    print('leaving')