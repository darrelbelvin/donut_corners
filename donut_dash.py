import base64
from io import BytesIO as _BytesIO
import time
import json
from textwrap import dedent as d

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import cv2

import numpy as np
from math import atan2

import plotly.graph_objs as go
from PIL import Image

from donut_corners import DonutCorners
from visualizing_donut_corners import *

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# Variables
HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '

# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Image utility functions
def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """
    t_start = time.time()

    buff = _BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    t_end = time.time()
    if verbose:
        print(f"PIL converted to b64 in {t_end - t_start:.3f} sec")

    return encoded


def numpy_to_b64(np_array, enc_format='png', scalar=False, **kwargs):
    """
    Converts a numpy image into base 64 string for HTML displaying
    :param np_array:
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :param scalar:
    :return:
    """
    #correct color order
    np_array = np_array[...,[2,1,0]]

    #correct for out of bounds
    min_val = np.min(np_array)
    if min_val < 0:
        np_array -= min_val
    
    max_val = np.max(np_array)
    if max_val > 255:
        np_array = np_array / max_val * 255

    # Convert from 0-1 to 0-255
    if scalar:
        np_array = np.uint8(255 * np_array)
    else:
        np_array = np.uint8(np_array)

    im_pil = Image.fromarray(np_array)

    return pil_to_b64(im_pil, enc_format, **kwargs)


def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = Image.open(buffer)

    return im


def b64_to_numpy(string, to_scalar=True):
    im = b64_to_pil(string)
    np_array = np.asarray(im)

    if to_scalar:
        np_array = np_array / 255.

    return np_array

# Custom Image Components
def DisplayImageNumpy(image_id, image, **kwargs):
    encoded_image = numpy_to_b64(image, enc_format='png')

    return html.Img(
        id=image_id,
        src=HTML_IMG_SRC_PARAMETERS + encoded_image,
        width='100%',
        **kwargs
    )


def InteractiveImageNumpy(image_id,
                        image,
                        enc_format='png',
                        display_mode='fixed',
                        dragmode='select',
                        verbose=False,
                        **kwargs):

    if display_mode.lower() in ['scalable', 'scale']:
        display_height = '{}vw'.format(round(60 * height / width))
    else:
        display_height = '95%'

    return dcc.Graph(
        id=image_id,
        figure=FigureForInteractiveImageNumpy(image, enc_format, dragmode, verbose),
        style=_merge({
            'height': display_height,
            'width': '100%'
        }, kwargs.get('style', {})),

        config={
            'modeBarButtonsToRemove': [
                'sendDataToCloud',
                'autoScale2d',
                'toggleSpikelines',
                'hoverClosestCartesian',
                'hoverCompareCartesian'
            ]
        },

        **_omit(['style'], kwargs)
    )


def FigureForInteractiveImageNumpy(image, enc_format='png', dragmode='select', verbose=False):
    if enc_format == 'jpeg':
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        encoded_image = numpy_to_b64(image, enc_format=enc_format, verbose=verbose, quality=80)
    else:
        encoded_image = numpy_to_b64(image, enc_format=enc_format, verbose=verbose)

    height, width = image.shape[:2]

    data = np.array(list(np.ndindex(image.shape[:2])))
    go_data = go.Scatter(y=data[:,0], x=data[:,1], mode='markers', opacity=0)

    return {
            'data': [go_data],
            'layout': {
                'margin': go.layout.Margin(l=30, b=20, t=10, r=10),
                'xaxis': {
                    'range': (0, width),
                    'scaleanchor': 'y',
                    'scaleratio': 1
                },
                'yaxis': {
                    'range': (height, 0)
                },
                'images': [{
                    'xref': 'x',
                    'yref': 'y',
                    'x': 0,
                    'y': 0,
                    'yanchor': 'top',
                    'sizing': 'stretch',
                    'sizex': width,
                    'sizey': height,
                    'layer': 'below',
                    'source': HTML_IMG_SRC_PARAMETERS + encoded_image,
                }],
                'dragmode': dragmode,
                'hovermode':'closest'
            }
        }

nl = '\n'

img = cv2.imread('images/bldg-1.jpg')
#crop
img = img[25:125, 750:850]

dc = DonutCorners(img)
pt = [-1,-1]
donut = dc.bake_donut(pt, dc.masks[0])

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Link(href='/assets/main.css', rel='stylesheet'),
    html.H1('Donut Corners'),
    html.Div('A corner detection method'),
    html.Div([
        html.Div([
            InteractiveImageNumpy(
                image_id='source',
                image=img
            ),
            html.Div([
                dcc.Markdown(d("""
                    **Click Data**
                    Click on points in the picture.
                """))
            ], id='click-data'),
        ],style={'width':'33%'}),
        html.Div([
            DisplayImageNumpy(
                image_id='inter',
                image=get_2dimg(dc, 'slopes')
            ),
            html.Div([
                dcc.Markdown(d("""
                    **Ray Data**
                    Click on points in the picture.
                """))
            ], id='ray-data'),
        ],style={'width':'33%'}),
        html.Div([
            DisplayImageNumpy(
                image_id='result',
                image=img
            )
        ],style={'width':'33%'})
    ], style={'display':'flex'})
], style={'textAlign': 'center'})


def update_point(clickData):
    global donut, pt
    newpt = [clickData['points'][0]['y'], clickData['points'][0]['x']]
    if newpt != pt:
        donut = dc.bake_donut(newpt, dc.masks[0])
        pt = newpt


@app.callback(
    Output('inter', 'src'),
    [Input('source', 'clickData')])
def display_ray_image(clickData):
    image = get_2dimg(dc, 'interest')

    if clickData:
        update_point(clickData)
        image = paint_donut(image, donut)
    
    encoded_image = numpy_to_b64(image, enc_format='png')
    return HTML_IMG_SRC_PARAMETERS + encoded_image


# @app.callback(
#     Output('inter', 'figure'),
#     [Input('source', 'clickData')])
# def display_ray_image(clickData):
#     image = get_2dimg(dc, 'interest')

#     if clickData:
#         update_point(clickData)
#         image = paint_donut(image, donut)
    
#     return FigureForInteractiveImageNumpy(image)


@app.callback(
    Output('ray-data', 'children'),
    [Input('source', 'clickData')])
def display_ray_data(clickData):
    if clickData:
        update_point(clickData)

        rays, profiles, strengths, ring = donut
        angles = [atan2(ray[0][0], ray[0][1]) for ray in rays]
        return dcc.Markdown(f"**Ray Data**{nl}{nl.join([f'Ray {i} - str: {strengths[i]:.2f} dir:{angles[i]:.2f}' for i in range(len(rays))])}", className='multiline')
    
    return dcc.Markdown(f"**Ray Data**{nl}Click on points in the picture.", className='multiline')


@app.callback(
    Output('click-data', 'children'),
    [Input('source', 'clickData')])
def display_click_data(clickData):
    if clickData:
        update_point(clickData)

        return dcc.Markdown(f"**Click Data**{nl}Point: x = {pt[1]}, y = {pt[0]}", className='multiline')
    
    return dcc.Markdown(f"**Click Data**{nl}Click on points in the picture.", className='multiline')


if __name__ == '__main__':
    app.run_server(debug=True)