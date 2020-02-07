from visualizing_donut_corners import *
from donut_corners import DonutCorners
from skimage import io

def test_rigidized():
    img = io.imread('images/tex-1.JPG')
    img = img[500:1500:5, 500:1500:5]

    kwargs = {'angle_count': 12 * 7, # must be multiple of 4
            'beam_width': 3,
            'beam_length': 50,
            'beam_start': 15,
            'eval_method': {'sectional': True, 'elimination_width': 7, 'max_n': 2, 'elim_double_ends': True}
            }

    dc = DonutCorners(**kwargs)
    dc.init(img)
    # dc.find_corners_grid(single_point = np.array([50,70]))
    dc.find_corners_grid()

    if dc.scored is not None:
        sc = dc.scored
    else:
        sc = np.nan_to_num(dc.scored_partial, nan=-0.5*np.max(np.nan_to_num(dc.scored_partial)))
    sc = sc / np.max(sc) * 255
    sc = np.pad(sc[...,None], ((0,0),(0,0),(0,2)), mode='constant').astype(int)

    show_img(paint_corners(np.maximum(dc.src[...,[2,1,0]], sc), dc))
    #show_img(sc)
    show_img(paint_corners(sc, dc))

def test_building(bldg_no = 1, crop = (slice(0,200), slice(650,950)), score_all = True, save_prefix = None):
    img = io.imread(f'images/bldg-{bldg_no}.jpg')
    if crop is not None:
        img = img[crop]
    #img = img[500:1500:5, 500:1500:5]

    kwargs = {'angle_count': 100,
            'beam_width': 2,
            'fork_spread': 2,
            'beam_length': 30,
            'beam_start': 5,
            'min_corner_score': 0.1,
            'eval_method': {'sectional': True, 'elimination_width': 6, 'max_n': 3, 'elim_double_ends': False}
            }

    dc = DonutCorners(**kwargs)
    dc.init(img)

    import sys
    if score_all:
        dc.score_all('pydevd' not in sys.modules)
    
    #dc.find_corner(np.array([50,70]))
    #dc.find_corners()#'pydevd' not in sys.modules)
    dc.find_corners_grid()

    if dc.scored is not None:
        sc = dc.scored
    else:
        sc = np.nan_to_num(dc.scored_partial, nan=0)
    sc = sc / np.max(sc) * 255
    sc = np.pad(sc[...,None], ((0,0),(0,0),(0,2)), mode='constant').astype(int)

    im_1 = paint_corners(np.maximum(dc.src, sc), dc)
    im_2 = paint_corners(sc.copy(), dc)

    if save_prefix is not None:
        io.imsave(save_prefix + "_all.png", im_1, check_contrast=False)
        io.imsave(save_prefix + "_scores_only.png", sc, check_contrast=False)
        io.imsave(save_prefix + "_scores_corners.png", im_2, check_contrast=False)
    else:
        show_img(im_1)
        if score_all:
            show_img(sc)
        show_img(im_2)


def beam_demo():
    kwargs = {'angle_count': 16, # must be multiple of 4
            'beam_width': 4,
            'fork_spread': 0,
            'beam_length': 30,
            'beam_start': 10,
            'eval_method': {'sectional': True, 'elimination_width': 7, 'max_n': 2, 'elim_double_ends': True}
            }
    dc = DonutCorners(**kwargs)
    show_beam(dc)

def beam_demo_small():
    kwargs = {'angle_count': 12,
            'beam_width': 1.5,
            'fork_spread': 1.2,
            'beam_length': 4.3,
            'beam_start': 0.5,
            'grid_size': 4,
            'search_args': dict(img_shape=(8,8), min_grid=0.1, top_n=2),
            'eval_method': {'sectional': True, 'elimination_width': 2, 'max_n': 2, 'elim_double_ends': True},
            }
    dc = DonutCorners(**kwargs)
    show_beam(dc)

if __name__ == "__main__":
    #test_rigidized()
    test_building(1, score_all=False)
    #test_building(1, None, score_all=False)
    #test_building(2, (slice(1000,1300), slice(1000,1300)), False)
    #beam_demo_small()
    #sobel_demo()