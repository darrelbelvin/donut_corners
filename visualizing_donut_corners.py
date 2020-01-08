import matplotlib.pyplot as plt
import numpy as np
from donut_corners import DonutCorners

def show_img(img):
    plt.figure()
    plt.imshow(img)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def show_imgs(imgs):
    if len(imgs) == 1:
        show_img(imgs[0])
        return
    
    fig, axs = plt.subplots(ncols=len(imgs))
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i])
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def show_std(dc):
    show_imgs((dc.scored, get_2dmap(dc, 'slopes'), get_2dmap(dc, 'interest'), dc.src))

def get_2dmap(dc, maptype = 'slopes'):
    map = {'slopes': dc.slopes, 'interest': dc.interest * 255}[maptype]
    return np.pad(map,((0,0),(0,0),(0,1)), mode='constant')

def paint_donut(map, dc: DonutCorners, point, rays = False):
    m2 = dc.clip_mask(dc.masks[0] + point)
    map = map.copy()
    map[m2[:,0], m2[:,1]] += 255
    if rays:
        map = paint_rays(map, dc, point)
    return map

def paint_rays(map, dc: DonutCorners, point):
    map = map.copy()
    for poi in dc.get_pois(dc.clip_mask(dc.masks[0] + point)):
        uv, perp, coords = dc.get_ray(point, poi)
        map[coords[:,0], coords[:,1]] += 255
    return map