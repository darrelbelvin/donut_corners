import cv2
import numpy as np
from scipy import optimize
from scipy.optimize._minimize import _minimize_neldermead

from multiprocessing import Pool, cpu_count
from math import pi
import random

import sys
sys.path.append('..')
from simplex_descent_quantized.simplex_descent_quantized import simplex_descent_quantized

class DonutCorners():
    rot90 = np.array([[0, -1], [1, 0]])
    
    # pylint: disable=too-many-instance-attributes
    def __init__(self, **kwargs):
        # passed on params
        self.sobel_params = {'ksize':3, 'scale':1, 'delta':0,
                             'ddepth':cv2.CV_32F, 'borderType':cv2.BORDER_DEFAULT}

        # beam & lighthouse
        self.angle_count = 12 # must be multiple of 4
        self.beam_count = self.angle_count * 3
        self.beam_width = 3
        self.fork_spread = 2
        self.beam_length = 30
        self.beam_start = 0
        self.beam_round = True
        self.beam_width = 2

        self.eval_method = {'sectional': False, 'elimination_width': self.beam_count // 30, 'max_n': 3, 'elim_double_ends': False}

        # grid params
        self.grid_size = 20
        self.min_corner_score = 0
        
        self.__dict__.update(kwargs)

        self.beam_diameter = 1 + self.beam_length * 2

        self.scored = None
        self.scored_partial = None
        self.basins = None
        self.corners = []

        self.baked_angles = np.linspace(0, 2*pi, self.angle_count, endpoint=False)
        self.beam()

    def fit(self, image):
        if isinstance(image, str):
            self.src = cv2.imread(image)
        else:
            self.src = image
        
        self.dims = np.array(self.src.shape[:2], dtype=int)
        self.scored_partial = np.empty(self.dims)
        self.scored_partial[:] = np.NaN
        self.basins = np.zeros(self.dims, dtype=int)

        self.preprocess()

    def preprocess(self):
        self.bw = np.mean(self.src, axis=-1)
        self.bw = np.pad(self.bw, (
            (self.beam_length,self.beam_length),(self.beam_length,self.beam_length)),
             mode='edge').astype('float32')

    def beam(self):
        r, d, ir = self.beam_length, self.beam_diameter, self.beam_start
        w, spr, count = self.beam_width, self.fork_spread, self.beam_count

        ind = np.array(list(np.ndindex((d,d)))).reshape((d,d,2))
        delta = ind - r

        beam_angles = np.linspace(0,2*pi, count, endpoint=False)
        
        beam_uvs = np.stack((np.sin(beam_angles), np.cos(beam_angles)), axis=-1)
        beam_perps = np.matmul(beam_uvs, DonutCorners.rot90)

        len_on_line = np.array([delta.dot(uv) for uv in beam_uvs])
        dist_to_line = np.array([delta.dot(perp) for perp in beam_perps])
        
        # make the prongs
        prong1 = np.maximum(w / 2 - np.abs(dist_to_line - spr / 2), 0)
        prong2 = np.minimum(-w / 2 + np.abs(dist_to_line + spr / 2), 0)

        # clip to length & side
        prong1[(len_on_line < ir) | (len_on_line > r) | (dist_to_line <= 0)] = 0
        prong2[(len_on_line < ir) | (len_on_line > r) | (dist_to_line > 0)] = 0

        # normalize
        c_1, c_2 = np.sum(prong1 != 0, axis=(1,2)), np.abs(np.sum(prong2 != 0, axis=(1,2)))
        c_big = np.max(np.array((c_1, c_2)), axis=0)
        prong1 = prong1 / np.sum(prong1, axis=(1,2))[:, None, None] #(np.sum(prong1, axis=(1,2)) / c_2 * c_big)[:, None, None]
        prong2 = prong2 / np.sum(prong2, axis=(1,2))[:, None, None] #(np.sum(prong2, axis=(1,2)) / c_1 * c_big)[:, None, None]

        # combine
        spiral = prong1 - prong2

        # store
        self.spiral = spiral.astype('float32')
        self.spiral_mask = spiral != 0
        self.weights = [self.spiral[i,...][self.spiral_mask[i,...]] for i in range(count)]
        self.beam_index = np.argwhere(self.spiral_mask)[...,0]
        self.beam_jumps = np.argwhere(self.beam_index[1:] != self.beam_index[:-1]).flatten() + 1


    # scoring methods
    def get_score(self, point):
        point = np.array(point, dtype=int)

        if not np.all((point >= 0) & (point < self.src.shape[:-1])):
            return 0
        if self.scored is not None:
            return self.scored[point[0],point[1]]
        
        if np.isnan(self.scored_partial[point[0],point[1]]):
            self.scored_partial[point[0],point[1]] = self.score_point(point)[0]
        
        return self.scored_partial[point[0],point[1]]


    def score_point(self, point):
        region = self.bw[point[0] : point[0] + self.beam_diameter,
                         point[1] : point[1] + self.beam_diameter]
        
        interest = [region[beam] for beam in self.spiral_mask]
        means = np.array([np.abs(np.mean(w * i)) for w, i in zip(self.weights, interest)])

        # scores = self.weights * region
        # means = np.abs(np.mean(scores, axis=(1,2)))

        # scores = self.angled_slopes[:,
        #                             point[0] : point[0] + self.beam_diameter,
        #                             point[1] : point[1] + self.beam_diameter
        #                             ][self.spiral]

        if not self.eval_method['sectional']:
            return (np.mean(means),)

        #score_sections = np.split(scores, self.beam_jumps)
        #means = np.array([np.abs(np.mean(sect)) for sect in score_sections])

        maxs = np.array([DonutCorners.get_max_idx(means, w=self.eval_method['elimination_width'],
                no_doubles = self.eval_method['elim_double_ends']) for _ in range(self.eval_method['max_n'])])

        beam_strengths = maxs[:,1]
        beam_ids = maxs[:,0].astype(int)
        angles = self.baked_angles[beam_ids]

        return np.mean(beam_strengths), angles, beam_strengths, beam_ids
    

    @staticmethod
    def get_max_idx(vals, w = 1, no_doubles = True, gradual = False):
        arg = np.argmax(vals)
        val = vals[arg]
        ind = np.arange(arg-w, arg + w + 1) % len(vals)
        vals[ind] = 0

        if gradual:
            if no_doubles:
                pass

        elif no_doubles:
            vals[(ind + len(vals)//2) % len(vals)] = 0 #eliminate double counting of edges
        
        return [arg, val]
        

    def score_row(self, y):
        return [self.score_point([y,x])[0] for x in range(self.src.shape[1])]


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

        def callback(*args, **kwargs):
            print('Callback')
            print(args)
            print(kwargs)

        negative = lambda *args: -1 * self.get_score(*args)
        # result = optimize.minimize(negative, np.array(point, dtype=int), method='Nelder-Mead', tol=0.1,
        #                 options=dict(
        #                     initial_simplex=np.array([point, point - self.grid_size//2, point - [0,self.grid_size//2]]),
        #                     callback=callback)
        #                 )

        # result = _minimize_neldermead(negative, np.array(point, dtype=int), xatol=1.5, fatol=10, callback=callback, return_all=True,
        #                     initial_simplex=np.array([point, point - self.grid_size//2, point - [0,self.grid_size//2]]))

        result = simplex_descent_quantized(negative, 1, np.array(point, dtype=int), max_iters = 1000,
                basin_mapping = True, current_basin_map = self.basins, max_step = 20,
                bounds = np.stack([np.zeros_like(self.dims), self.dims], axis=1))

        #self.basins = result['new_basins']

        if result['exit_cause'] == "maximum found":
            self.corners.append([-result['y'], result['x']])
            return [-result['y'], result['x']]
        else:
            return None
        
        best2 = result['x'].astype(int)
        best_val = self.get_score(best2) #abs(result['fun'])
        
        best = np.array([-1,-1])
        brute_radius = 3

        while not np.all(best == best2):
            best = best2
            brute_grid = np.swapaxes(np.mgrid[best[0] - brute_radius:best[0] + brute_radius + 1,
                                            best[1] - brute_radius:best[1] + brute_radius + 1], 0,2).reshape(-1,2)

            for p in brute_grid:
                if self.get_score(p) > best_val:
                    best_val = self.get_score(p)
                    best2 = p

        if best_val >= self.min_corner_score:
            self.corners.append([best_val, best])
        
        return best


    def find_corners(self, multithread = False):
        offset = 5
        while np.min(self.basins[offset:-offset,offset:-offset]) == 0:
            true_ind = np.argwhere(self.basins == 0)
            chosen = np.random.randint(0, true_ind.shape[0])
            self.find_corner(true_ind[chosen])
            print(np.mean(self.basins != 0))

        # grid = np.swapaxes(np.mgrid[self.grid_size//2:self.dims[0]:self.grid_size,
        #         self.grid_size//2:self.dims[1]:self.grid_size], 0,2).reshape(-1,2)

        # if multithread:
        #     with Pool(cpu_count() - 1) as p:
        #         self.corners = p.map(self.find_corner, grid)

        # else:
        #     for point in grid:
        #         self.find_corner(point)


if __name__ == "__main__":
    from visualizing_donut_corners import *
    # beam, angles, ring = DonutCorners.beam(2)
    # print(np.moveaxis(beam, 2,0))
    # print(angles)

    # beam, angles, ring = DonutCorners.beam(2)
    # print(np.moveaxis(beam, 2,0))
    # print(angles)

    img = cv2.imread('images/bldg-1.jpg')
    #img = cv2.imread('images/tex-1.JPG')
    #crop
    img = img[:200, 650:950]
    #img = img[500:1500:5, 500:1500:5]

    kwargs = {'angle_count': 12 * 4, # must be multiple of 4
            'beam_count': 12 * 4,
            'beam_width': 4,
            'fork_spread': 0,
            'beam_length': 20,
            'beam_start': 5,
            'beam_round': True,
            'eval_method': {'sectional': False, 'elimination_width': 6, 'max_n': 3, 'elim_double_ends': True}
            }

    dc = DonutCorners(**kwargs)
    dc.fit(img)
    print(dc.score_point(np.array([50,50])))

    #show_beam(dc)
    
    #print(dc.score_point(np.array([50,50])))
    import sys
    #dc.score_all('pydevd' not in sys.modules)
    
    #dc.find_corner(np.array([50,70]))
    
    dc.find_corners()#'pydevd' not in sys.modules)

    if dc.scored is not None:
        sc = dc.scored
    else:
        sc = np.nan_to_num(dc.scored_partial, nan=-0.5*np.max(np.nan_to_num(dc.scored_partial)))
    sc = sc / np.max(sc) * 255
    sc = np.pad(sc[...,None], ((0,0),(0,0),(0,2)), mode='constant').astype(int)

    #show_img(paint_zones(paint_corners(np.maximum(dc.src[...,[2,1,0]], sc), dc), dc))
    show_img(paint_basins(paint_corners(np.maximum(dc.src[...,[2,1,0]], sc), dc), dc))
    show_img(sc)
    show_img(paint_corners(sc, dc))

    #show_std(dc)
    
    print('done')
    print('leaving')