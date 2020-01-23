import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
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
    show_imgs((dc.scored, get_2dimg(dc, 'slopes'), get_2dimg(dc, 'interest'), dc.src))

def get_2dimg(dc, imgtype = 'slopes'):
    img = {'slopes': dc.slopes, 'interest': dc.interest * 255}[imgtype]
    return np.pad(img,((0,0),(0,0),(1,0)), mode='constant')

def paint_donut_old(img, dc: DonutCorners, point, rays = False):
    m2 = dc.clip_mask(dc.msk0 + point)
    img = img.copy()
    img[m2[:,0], m2[:,1]] = 255
    if rays:
        img = paint_rays(img, dc, point)
    return img


def paint_donut(img, donut):
    add_img = np.zeros_like(img)

    rays, profiles, strengths, angles, mask, topids = donut

    #for ray, strength in zip(rays, strengths.astype(int)):
    for i in range(len(rays)):
        ray = rays[i]
        strength = strengths[i]
        uv, perp, coords = ray
        colors = [0,1,2] if i in topids else [0]
        for color in colors:
            add_img[coords[:,0], coords[:,1], color] = 255#np.maximum(add_img[coords[:,0], coords[:,1], color], strength)

    max_val = np.max(add_img)
    if max_val > 0:
        add_img = add_img / max_val * 255
    
    add_img[mask[:,0], mask[:,1], 0] = 255

    return np.max(np.array([img, add_img]), axis=0)


def paint_rays(img, dc: DonutCorners, point):
    return paint_donut(img, dc.bake_donut(point))


def paint_zones(img, dc: DonutCorners):
    n=1
    s1 = np.pad(dc.zones[:,:-n], ((0,0),(n,0)), mode='constant', constant_values = -1)
    s2 = np.pad(dc.zones[:-n,:], ((n,0),(0,0)), mode='constant', constant_values = -1)

    add_img = ((dc.zones != s1) | (dc.zones != s2)).astype(int)
    
    add_img = np.pad(add_img[:,:,None], ((0,0),(0,0),(2,0)), mode='edge')

    return np.max(np.array([img, add_img * 255]), axis=0)


def paint_corners(img, dc: DonutCorners):
    add_img = np.zeros_like(img)

    for point in dc.corners:
        add_img[point[0], point[1], :] = 255
    
    return np.max(np.array([img, add_img]), axis=0)


def show_beam(dc: DonutCorners):
    points = np.array(list(np.ndindex(dc.spiral.shape)))[dc.spiral.flatten()]
    fig = px.scatter_3d(x=points[:,1], y=points[:,2], z=points[:,0], color=points[:,0], opacity=0.5)
    fig.show()