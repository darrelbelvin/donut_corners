import cv2
import numpy as np
from scipy import optimize

from multiprocessing import Pool, cpu_count
from math import pi
import random

import sys
sys.path.append('../')

class DonutCorners():
    rot90 = np.array([[0, -1], [1, 0]])
    
    # pylint: disable=too-many-instance-attributes
    def __init__(self, image, **kwargs):

        # passed on params
        self.sobel_params = {'ksize':3, 'scale':1, 'delta':0,
                             'ddepth':cv2.CV_32F, 'borderType':cv2.BORDER_DEFAULT}

        # vortex & lighthouse
        self.angle_count = 12 # must be multiple of 4
        self.beam_count = self.angle_count * 3
        self.beam_width = 2
        self.vortex_radius = 30
        self.vortex_inner_radius = 0
        self.vortex_round = True
        self.beam_width = 2

        self.eval_method = {'sectional': False, 'elimination_width': self.beam_count // 30, 'max_n': 3, 'elim_double_ends': False}

        # grid params
        self.grid_size = 20
        self.min_corner_score = 10
        
        self.__dict__.update(kwargs)

        # data init
        if isinstance(image, str):
            self.src = cv2.imread(image)
        else:
            self.src = image
        self.dims = np.array(self.src.shape[:2], dtype=int)

        self.vortex_diameter = 1 + self.vortex_radius * 2

        self.scored = None
        self.scored_partial = np.empty(self.dims)
        self.scored_partial[:] = np.NaN

        self.zones = np.empty(self.dims, dtype=int)
        self.zones[:] = -1
        self.zones_mask = np.array(list(np.ndindex(7,7))) - [3,3]

        self.corners = []

        self.baked_angles = np.linspace(0, 2*pi, self.angle_count, endpoint=False)
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

        uvs = np.stack((np.sin(self.baked_angles),np.cos(self.baked_angles)), axis=-1)
        #uvs = np.stack((np.cos(self.baked_angles + pi/2),np.sin(self.baked_angles + pi/2)), axis=-1)
        
        img_dirs = np.arctan2(self.slopes[...,0], self.slopes[...,1])
        angle_deltas = np.abs((img_dirs[None,...] - self.baked_angles[:,None,None])%pi - (pi/2))

        n = 30
        sharpening_factor = n**(1*angle_deltas)

        angled_slopes = np.array([self.slopes.dot(uv) for uv in uvs])
        angled_slopes = angled_slopes * sharpening_factor
        # angled_slopes = np.tile(angled_slopes, (2,1,1))
        # angled_slopes[self.angle_count//2:] *= -1
        self.angled_slopes = np.pad(angled_slopes, ((0,0),
            (self.vortex_radius,self.vortex_radius),(self.vortex_radius,self.vortex_radius)),
             mode='constant', constant_values=0)
    

    def vortex(self):
        r, d, ir = self.vortex_radius, self.vortex_diameter, self.vortex_inner_radius
        mult = self.beam_count / self.angle_count
        #r = d/2
        #spiral = np.zeros((self.beam_count,d,d), dtype=bool)

        ind = np.array(list(np.ndindex((d,d)))).reshape((d,d,2))
        delta = ind - r

        beam_angles = np.linspace(0,2*pi, self.beam_count, endpoint=False)
        
        beam_uvs = np.stack((np.sin(beam_angles), np.cos(beam_angles)), axis=-1)
        beam_perps = np.matmul(beam_uvs, DonutCorners.rot90)

        len_on_line = np.array([delta.dot(uv) for uv in beam_uvs])
        dist_to_line = np.array([delta.dot(perp) for perp in beam_perps])
        
        spiral = (np.abs(dist_to_line) < self.beam_width / 2) & (len_on_line > ir) & (len_on_line < d/2)

        #spiral = np.roll(spiral, - int(mult/2), axis=0)
        spiral = np.roll(spiral, self.angle_count // 4, axis=0)

        self.spiral = spiral #np.roll(spiral, 3*self.angle_count // 4, axis=1)

        self.beam_index = np.argwhere(spiral)[...,0]
        self.beam_jumps = np.argwhere(self.beam_index[1:] != self.beam_index[:-1]).flatten() + 1


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
        scores = self.angled_slopes[:,
                                    point[0] : point[0] + self.vortex_diameter,
                                    point[1] : point[1] + self.vortex_diameter
                                    ][self.spiral]

        #return np.mean(np.abs(scores))

        if not self.eval_method['sectional']:
            return np.mean(np.abs(scores))

        score_sections = np.split(scores, self.beam_jumps)
        means = np.array([np.abs(np.mean(sect)) for sect in score_sections])

        return(np.mean([DonutCorners.get_max(means, w=self.eval_method['elimination_width'],
                no_doubles = self.eval_method['elim_double_ends']) for _ in range(self.eval_method['max_n'])]))
    

    @staticmethod
    def get_max(vals, w = 1, no_doubles = True, gradual = True):
        arg = np.argmax(vals)
        ret = vals[arg]
        ind = np.arange(arg-w, arg + w + 1) % len(vals)
        vals[ind] = 0

        if gradual:
            

            if no_doubles:
                pass

        elif no_doubles:
            vals[(ind + len(vals)//2) % len(vals)] = 0 #eliminate double counting of edges
        
        return ret
        

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
    

    def find_corner(self, point):
        negative = lambda *args: -1 * self.get_score(*args)
        result = optimize.minimize(negative, np.array(point, dtype=int), method='Nelder-Mead', tol=0.1,
                        options={'initial_simplex':np.array([point, point - self.grid_size//2, point - [0,self.grid_size//2]])})

        best = result['x'].astype(int)
        best_val = abs(result['fun'])
        
        best2 = np.array([-1,-1])
        brute_radius = 2

        while not np.all(best2 == best):
            best2 = best
            brute_grid = np.swapaxes(np.mgrid[best[0] - brute_radius:best[0] + brute_radius + 1,
                                            best[1] - brute_radius:best[1] + brute_radius + 1], 0,2).reshape(-1,2)

            for p in brute_grid:
                if self.get_score(p) > best_val:
                    best_val = self.get_score(p)
                    best2 = p

        if best_val >= self.min_corner_score:
            self.corners.append(best)
        
        return best


    def find_corners(self, multithread = False):
        grid = np.swapaxes(np.mgrid[self.grid_size//2:self.dims[0]:self.grid_size,
                self.grid_size//2:self.dims[1]:self.grid_size], 0,2).reshape(-1,2)

        if multithread:
            with Pool(cpu_count() - 1) as p:
                self.corners = p.map(self.find_corner, grid)
        
        else:
            for point in grid:
                self.find_corner(point)    


if __name__ == "__main__":
    from visualizing_donut_corners import *
    # vortex, angles, ring = DonutCorners.vortex(2)
    # print(np.moveaxis(vortex, 2,0))
    # print(angles)

    img = cv2.imread('images/bldg-1.jpg')
    #crop
    img = img[:200, 650:950]
    #img = img[125:130, 800:805]

    kwargs = {'angle_count': 12 * 7, # must be multiple of 4
            'beam_count': 12 * 7,
            'beam_width': 2,
            'vortex_radius': 30,
            'vortex_inner_radius': 5,
            'vortex_round': True,
            'eval_method': {'sectional': True, 'elimination_width': 7, 'max_n': 3, 'elim_double_ends': True}
            }

    dc = DonutCorners(img, **kwargs)

    #show_vortex(dc)

    print(dc.score_point(np.array([100,100])))
    import sys


    dc.score_all('pydevd' not in sys.modules)
    #dc.find_corner(np.array([30,30]))
    dc.find_corners()#'pydevd' not in sys.modules)

    #sc = np.nan_to_num(dc.scored_partial, nan=-0.5*np.max(np.nan_to_num(dc.scored_partial)))
    sc = dc.scored
    sc = sc / np.max(sc) * 255
    sc = np.pad(sc[...,None], ((0,0),(0,0),(0,2)), mode='constant').astype(int)

    #show_img(paint_zones(paint_corners(np.maximum(dc.src[...,[2,1,0]], sc), dc), dc))
    show_img(paint_corners(np.maximum(dc.src[...,[2,1,0]], sc), dc))
    show_img(sc)
    show_img(paint_corners(sc, dc))

    #show_std(dc)
    
    print('done')
    print('leaving')