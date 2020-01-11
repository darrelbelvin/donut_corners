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

        # vortex & lighthouse
        self.angle_count = 48 # must be multiple of 4
        self.vortex_radius = 30
        self.vortex_diameter = 1 + self.vortex_radius * 2
        self.vortex_inner_radius = 0
        self.vortex_round = True

        self.eval_method = {'sectional': False}

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
        self.zones_mask = np.array(list(np.ndindex(7,7))) - [3,3]

        self.corners = []

        angles = np.linspace(0,pi,self.angle_count//2, endpoint=False)
        angles = np.tile(angles, (2,))
        angles[self.angle_count//2:] += pi
        self.baked_angles = angles
        self.vortex()
        
        self.preprocess()

    def preprocess(self):
        edges_x = cv2.Sobel(self.src, dx=1, dy=0, **self.sobel_params)
        edges_y = cv2.Sobel(self.src, dx=0, dy=1, **self.sobel_params)

        def absmaxND(a: np.ndarray, axis=None, keepdims=False):
            amax = a.max(axis, keepdims=keepdims)
            amin = a.min(axis, keepdims=keepdims)
            return np.where(-amin > amax, amin, amax)

        edges_x_max = absmaxND(edges_x, axis=-1)
        edges_y_max = absmaxND(edges_y, axis=-1)
        
        self.slopes = np.stack((edges_y_max, edges_x_max), axis=-1)

        uvs = np.stack((np.cos(self.baked_angles),np.sin(self.baked_angles)), axis=-1)
        #uvs = np.stack((np.cos(self.baked_angles + pi/2),np.sin(self.baked_angles + pi/2)), axis=-1)
        
        angled_slopes = np.array([self.slopes.dot(uv) for uv in uvs])
        # angled_slopes = np.tile(angled_slopes, (2,1,1))
        # angled_slopes[self.angle_count//2:] *= -1
        self.angled_slopes = np.pad(angled_slopes, ((0,0),
            (self.vortex_radius,self.vortex_radius),(self.vortex_radius,self.vortex_radius)),
             mode='constant')

    def vortex(self):
        r, d, ir = self.vortex_radius, self.vortex_diameter, self.vortex_inner_radius
        #r = d/2
        spiral = np.zeros((self.angle_count,d,d), dtype=bool)

        ind = np.array(list(np.ndindex(spiral.shape[1:]))).reshape((d,d,2))
        delta = ind - r

        # dirs = delta / lens[...,None]
        # dirs[lens > r] = 0

        angles = np.arctan2(-delta[...,0], delta[...,1]) % (2*pi) # standard for polar graphing
        angles[angles > self.baked_angles[-1] + self.baked_angles[1]/2] -= (2*pi)
        
        angle_args = np.concatenate((np.searchsorted(self.baked_angles, angles - self.baked_angles[1]/2)[...,None], ind), axis=-1)
        spiral[angle_args[...,0], angle_args[...,1], angle_args[...,2]] = True

        if self.vortex_round:
            lens = np.linalg.norm(delta,axis=-1)
            mask = (lens < d/2) & (lens > ir - 0.5)
            #lens[radius, radius] = -1
            spiral = spiral & mask

        else:
            spiral[r-ir, r+ir:r-ir, r+ir] = False

        self.spiral = spiral #np.roll(spiral, self.angle_count // 4, axis=1)

        self.angle_index = np.argwhere(spiral)[...,0]
        self.angle_jumps = np.argwhere(self.angle_index[1:] != self.angle_index[:-1]).flatten() + 1


    # scoring methods
    def get_score(self, point):
        if type(point) !=np.ndarray:
            point = np.array(point)

        if not np.all((point >= 0) & (point < self.src.shape[:-1])):
            return 0
        if self.scored is not None:
            return self.scored[point[0],point[1]]
        
        if np.isnan(self.scored_partial[point[0],point[1]]):
            self.scored_partial[point[0],point[1]] = self.score_point(point)
        
        return self.scored_partial[point[0],point[1]]


    def score_point(self, point):
        scores = self.angled_slopes[:,point[0] : point[0] + self.vortex_diameter, point[1] : point[1] + self.vortex_diameter][self.spiral]
        scores = np.abs(scores)

        if not self.eval_method['sectional']:
            return np.mean(scores)

        score_sections = np.split(scores, self.angle_jumps)
        maxs = np.array([np.max(sect) for sect in score_sections])
        means = np.array([np.mean(sect) for sect in score_sections])
        
        def get_max(vals, w = 1):
            arg = np.argmax(vals)
            ret = vals[arg]
            ind = np.arange(arg-w, arg + w + 1) % len(vals)
            vals[ind] = 0
            vals[(ind + len(vals)//2) % len(vals)] = 0 #eliminate double counting of edges
            return ret

        return(np.mean([get_max(means, w=2) for _ in range(3)]))
        

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
    

    def clip_mask(self, mask):
        return mask[np.all((mask >= 0) & (mask < self.src.shape[:-1]), axis=1)]
    

    def donut_slider(self, point, ray_len = 5, max_iters = 50, min_score = 10):
        axis = np.array([[0,1],[1,1],[1,0],[1,-1]])
        corner_id = len(self.corners)

        for _ in range(max_iters):
            #print(point, self.get_score(point))
            if self.zones[point[0], point[1]] not in [-1, corner_id]:
                self.zones[self.zones == corner_id] = self.zones[point[0], point[1]]
                return self.corners[self.zones[point[0], point[1]]]

            mask = self.clip_mask(self.zones_mask + point.astype(int))
            self.zones[mask[:,0],mask[:,1]] = corner_id
            
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
        for point in np.swapaxes(np.mgrid[self.grid_size//2:self.dims[0]:self.grid_size,
                    self.grid_size//2:self.dims[1]:self.grid_size], 0,2).reshape(-1,2):
            self.donut_slider(point)


if __name__ == "__main__":
    from visualizing_donut_corners import *

    # vortex, angles, ring = DonutCorners.vortex(2)
    # print(np.moveaxis(vortex, 2,0))
    # print(angles)

    img = cv2.imread('images/bldg-1.jpg')
    #crop
    img = img[:200, 650:950]
    #img = img[125:130, 800:805]

    dc = DonutCorners(img)
    print(dc.score_point(np.array([100,100])))
    import sys


    dc.score_all('pydevd' not in sys.modules)
    
    dc.find_corners()

    sc = dc.scored - np.min(dc.scored)
    sc = sc / np.max(sc) * 255
    sc = np.pad(sc[...,None], ((0,0),(0,0),(0,2)), mode='constant').astype(int)

    show_img(paint_zones(paint_corners(np.maximum(dc.src[...,[2,1,0]], sc), dc), dc))
    show_img(sc)
    show_img(paint_corners(sc, dc))



    #show_img(dc.src[...,[2,1,0]])
    # dc.find_corners()

    # print(img.shape)
    # img

    # p2 = dc.donut_slider(np.array([100,150]))
    # print(p2)
    # print(dc.get_score(p2))

    #show_std(dc)
    
    #print(dc.score_point(pt))strongest several
    #data = list(np.ndindex(dc.dims))
    
    #dm = paint_donut(get_2dimg(dc, 'slopes'), dc, pt, rays = True)
    #show_imgs((dm, dc.src))

    #show_imgs((dc.interest[...,0], dc.slopes[...,0]))
    
    print('done')
    print('leaving')