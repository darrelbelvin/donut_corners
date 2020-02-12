import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from donut_corners import DonutCorners

def show_img(img, cmap=None):
    plt.figure()
    plt.imshow(img, cmap=cmap)
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


def paint_basins(img, dc: DonutCorners):
    n=1
    s1 = np.pad(dc.basins[:,:-n], ((0,0),(n,0)), mode='constant', constant_values = -1)
    s2 = np.pad(dc.basins[:-n,:], ((n,0),(0,0)), mode='constant', constant_values = -1)

    add_img = ((dc.basins != s1) | (dc.basins != s2)).astype(int)
    
    add_img = np.pad(add_img[:,:,None], ((0,0),(0,0),(2,0)), mode='edge')

    return np.max(np.array([img, add_img * 255]), axis=0)


def paint_corners(img, dc: DonutCorners):
    add_img = np.zeros_like(img, dtype=float)
    #di = int(dc.beam_diameter)

    for point in dc.corners:
        add_img[point[1][0], point[1][1], :] = point[0]

        # region = add_img[point[0] : point[0] + di,
        #                  point[1] : point[1] + di, 1]

        score, angles, beam_strengths, beam_ids = point[2]
        #beam_strengths = beam_strengths / np.max(beam_strengths)
        for angle, strength in zip(angles, beam_strengths):
            for r in range(int(dc.beam_start), int(dc.beam_length)):
                ray_point = point[1] + np.round(r*np.array((np.sin(angle),np.cos(angle)))).astype(int)
                if not dc.out_of_bounds(ray_point):
                    add_img[ray_point[0], ray_point[1], 1] = strength

    
    if np.max(add_img) != 0:
        add_img = (add_img / np.max(add_img) * 255)
    add_img = add_img.astype(img.dtype)

    img[add_img != 0] += add_img[add_img != 0]
    return (img / np.max(img) * 255).astype(int)
    return np.max(np.array([img, add_img]), axis=0)


def show_beam(dc: DonutCorners):
    show_3d_kernel(dc.spiral)


def show_3d_kernel(arr, ret=False):
    points = np.array(list(np.ndindex(arr.shape)))[arr.flatten() != 0]
    fig = px.scatter_3d(x=points[:,1], y=points[:,2], z=points[:,0], color=arr[points[:,0], points[:,1], points[:,2]], opacity=0.5)
    if ret:
        return fig
    fig.show()