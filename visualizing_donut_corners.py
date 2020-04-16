import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from PIL import Image

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
                ray_point = point[1] + np.round(r*np.array((-1*np.sin(angle),np.cos(angle)))).astype(int)
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
    fig = px.scatter_3d(y=points[:,1], x=points[:,2], z=points[:,0], color=arr[points[:,0], points[:,1], points[:,2]], opacity=0.5)
    fig.update_xaxes(autorange="reversed")
    #fig.update_layout(dict(xaxis_autorange="reversed"))
    if ret:
        return fig
    fig.show()


def show_slope_polar(arr, ret = False):
    area = arr.shape[0] * arr.shape[1]
    max_points = 1000
    step = int((area/max_points)**0.5)
    points = arr[step//2::step, step//2::step, :]
    y = np.repeat(np.r_[arr.shape[0]:step//2:-step], points.shape[1])
    x = np.tile(np.r_[step//2:arr.shape[1]:step], points.shape[0])
    u = (np.cos(points[:,:,0]) * points[:,:,1]).flatten()
    v = (np.sin(points[:,:,0]) * points[:,:,1]).flatten()

    fig = ff.create_quiver(x, y, u, v)

    if ret:
        return fig
    fig.show()

def show_src_slopes(dc: DonutCorners, x0 = 0, x1 = None, y0 = 0, y1 = None, max_points = 1000, scale_factor = 3, ret = False):
    if x1 is None:
        x1 = dc.src.shape[1]
    if y1 is None:
        y1 = dc.src.shape[0]
    
    area = (x1 - x0) * (y1 - y0)
    if area > max_points:
        step = int((area/max_points)**0.5)
    else:
        step = 1
    
    y = np.r_[y1:y0:-step]
    x = np.r_[x0:x1:step]
    xv, yv = np.meshgrid(x, y)
    
    if y0 == 0:
        y0 = None
    u = dc.uv[1][y1:y0:-step, x0:x1:step].flatten()
    v = dc.uv[0][y1:y0:-step, x0:x1:step].flatten()
    return show_img_and_quiver(
        xv.flatten() * scale_factor,
        (dc.src.shape[0] - yv.flatten()) * scale_factor,
        u, -v, dc.src, scale_factor, ret)

def show_img_and_quiver(x, y, u, v, img, scale_factor = 3, ret = False):
    fig = ff.create_quiver(x, y, u, v,
                       scale=.25,
                       arrow_scale=.4,
                       name='quiver',
                       line_width=1,
                       marker={'color': 'red'})

    img = Image.fromarray(img)

    # Constants
    img_width = img.size[0]
    img_height = img.size[1]

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            #marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    if ret:
        return fig
    fig.show(config={'doubleClick': 'reset'})


def show_img_plotly(img, ret = False):
    img = Image.fromarray(img)
    # Create figure
    fig = go.Figure()

    # Constants
    img_width = img.size[0]
    img_height = img.size[1]
    scale_factor = 1

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/
    
    if ret:
        return fig
    fig.show(config={'doubleClick': 'reset'})


if __name__ == "__main__":
    dc = DonutCorners()
    img = io.imread('images/bldg-1.jpg')
    img = img[100:200, 850:950]
    dc.init(img)

    #show_3d_kernel(dc.spiral)
    #show_img_plotly(dc.src)
    #show_slope_polar(dc.polar)
    show_src_slopes(dc)
