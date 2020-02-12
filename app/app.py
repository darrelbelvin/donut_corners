import time
import importlib

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State

import utils.dash_reusable_components as drc
import utils.figures as figs

import sys
sys.path.append('.')

#import donut_corners.donut_corners as donut_corners
from donut_corners import DonutCorners
from visualizing_donut_corners import show_3d_kernel

dash_app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = dash_app.server

dc = DonutCorners()


dash_app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Donut Corners Demo",
                                    href="https://github.com/darrelbelvin/donut-corners/",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        # html.A(
                        #     id="banner-logo",
                        #     children=[
                        #         html.Img(src=app.get_asset_url("dash-logo-new.png"))
                        #     ],
                        #     href="https://plot.ly/products/dash/",
                        # ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Beam Count",
                                            id="slider-kernel-beam-count",
                                            min=5,
                                            max=100,
                                            step=5,
                                            marks={
                                                str(i): str(i)
                                                for i in [5, 20, 40, 60, 80, 100]
                                            },
                                            value=20,
                                        ),
                                        drc.NamedSlider(
                                            name="Beam Width",
                                            id="slider-kernel-beam-width",
                                            min=1,
                                            max=10,
                                            marks={str(i): str(i) for i in range(1,11)},
                                            step=0.25,
                                            value=2,
                                        ),
                                        drc.NamedSlider(
                                            name="Beam Start",
                                            id="slider-kernel-beam-start",
                                            min=0,
                                            max=15,
                                            marks={str(i): str(i) for i in range(0, 15, 3)},
                                            step=0.25,
                                            value=5,
                                        ),
                                        drc.NamedSlider(
                                            name="Beam End",
                                            id="slider-kernel-beam-end",
                                            min=5,
                                            max=40,
                                            marks={str(i): str(i) for i in range(5, 41, 5)},
                                            step=0.25,
                                            value=20,
                                        )
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button(
                                            "Reset Threshold",
                                            id="button-zero-threshold",
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="last-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Kernel",
                                            id="dropdown-svm-parameter-kernel",
                                            options=[
                                                {
                                                    "label": "Radial basis function (RBF)",
                                                    "value": "rbf",
                                                },
                                                {"label": "Linear", "value": "linear"},
                                                {
                                                    "label": "Polynomial",
                                                    "value": "poly",
                                                },
                                                {
                                                    "label": "Sigmoid",
                                                    "value": "sigmoid",
                                                },
                                            ],
                                            value="rbf",
                                            clearable=False,
                                            searchable=False,
                                        ),
                                        drc.NamedSlider(
                                            name="Cost (C)",
                                            id="slider-svm-parameter-C-power",
                                            min=-2,
                                            max=4,
                                            value=0,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-2, 5)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-C-coef",
                                            min=1,
                                            max=9,
                                            value=1,
                                        ),
                                        drc.NamedSlider(
                                            name="Degree",
                                            id="slider-svm-parameter-degree",
                                            min=2,
                                            max=10,
                                            value=3,
                                            step=1,
                                            marks={
                                                str(i): str(i) for i in range(2, 11, 2)
                                            },
                                        ),
                                        drc.NamedSlider(
                                            name="Gamma",
                                            id="slider-svm-parameter-gamma-power",
                                            min=-5,
                                            max=0,
                                            value=-1,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-5, 1)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-gamma-coef",
                                            min=1,
                                            max=9,
                                            value=5,
                                        ),
                                        html.Div(
                                            id="shrinking-container",
                                            children=[
                                                html.P(children="Shrinking"),
                                                dcc.RadioItems(
                                                    id="radio-svm-parameter-shrinking",
                                                    labelStyle={
                                                        "margin-right": "7px",
                                                        "display": "inline-block",
                                                    },
                                                    options=[
                                                        {
                                                            "label": " Enabled",
                                                            "value": "True",
                                                        },
                                                        {
                                                            "label": " Disabled",
                                                            "value": "False",
                                                        },
                                                    ],
                                                    value="True",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=[
                                dcc.Tabs(
                                    id="tabs",
                                    value='kernel',
                                    children=[
                                        dcc.Tab(label='Kernel Playground', value='kernel'),
                                        dcc.Tab(label='Scoring Process', value='scoring'),
                                        dcc.Tab(label='Maximizer', value='opto'),
                                        dcc.Tab(label='Results', value='results'),
                                    ]
                                ),
                                html.Div(
                                    id="svm-graph-container",
                                    children=dcc.Loading(
                                        className="graph-wrapper",
                                        children=dcc.Graph(
                                            id="graph-sklearn-svm",
                                            figure=dict(
                                                layout=dict(
                                                    plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                                )
                                            ),
                                        ),
                                        style={"display": "none"},
                                    ),
                                ),
                            ]
                        ),
                    ],
                )
            ],
        ),
    ]
)


# @dash_app.callback(
#     Output("slider-svm-parameter-gamma-coef", "marks"),
#     [Input("slider-svm-parameter-gamma-power", "value")],
# )
# def update_slider_svm_parameter_gamma_coef(power):
#     scale = 10 ** power
#     return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


# @dash_app.callback(
#     Output("slider-svm-parameter-C-coef", "marks"),
#     [Input("slider-svm-parameter-C-power", "value")],
# )
# def update_slider_svm_parameter_C_coef(power):
#     scale = 10 ** power
#     return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


# @dash_app.callback(
#     Output("slider-threshold", "value"),
#     [Input("button-zero-threshold", "n_clicks")],
#     [State("graph-sklearn-svm", "figure")],
# )
# def reset_threshold_center(n_clicks, figure):
#     if n_clicks:
#         Z = np.array(figure["data"][0]["z"])
#         value = -Z.min() / (Z.max() - Z.min())
#     else:
#         value = 0.4959986285375595
#     return value


# # Disable Sliders if kernel not in the given list
# @dash_app.callback(
#     Output("slider-svm-parameter-degree", "disabled"),
#     [Input("dropdown-svm-parameter-kernel", "value")],
# )
# def disable_slider_param_degree(kernel):
#     return kernel != "poly"


# @dash_app.callback(
#     Output("slider-svm-parameter-gamma-coef", "disabled"),
#     [Input("dropdown-svm-parameter-kernel", "value")],
# )
# def disable_slider_param_gamma_coef(kernel):
#     return kernel not in ["rbf", "poly", "sigmoid"]


# @dash_app.callback(
#     Output("slider-svm-parameter-gamma-power", "disabled"),
#     [Input("dropdown-svm-parameter-kernel", "value")],
# )
# def disable_slider_param_gamma_power(kernel):
#     return kernel not in ["rbf", "poly", "sigmoid"]


@dash_app.callback(
    Output("svm-graph-container", "children"),
    [
        Input('tabs', 'value'),
        Input("slider-kernel-beam-count", "value"),
        Input("slider-kernel-beam-width", "value"),
        Input("slider-kernel-beam-start", "value"),
        Input("slider-kernel-beam-end", "value"),
    ],
)
def update_svm_graph(
    tab,
    beam_count,
    beam_width,
    beam_start,
    beam_end
):
    try:
        dc.set_params(self_correct=False,
                    angle_count=beam_count,
                    beam_width=beam_width,
                    beam_start=beam_start,
                    beam_length=beam_end)
    except ValueError:
        print('asdfasdfasdf')
    
    if tab == 'kernel':
        kernel_fig = show_3d_kernel(dc.spiral, True)
        kernel_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                scene_bgcolor='rgba(0,0,0,0)',
                                font_color='white',
                                font_size=14,
                                overwrite=True)

        return dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-svm", figure=kernel_fig),
                style={"display": "none"},
            )
    
    if tab == 'scoring':
        return html.Div([
            html.H2('Scoring Process')
        ])
    if tab == 'opto':
        return html.Div([
            html.H2('Optimization Process')
        ])    
    if tab == 'results':
        return html.Div([
            html.H2('End Result')
        ])
    return html.H2('Nonexistant Tab')

        # html.Div(
        #     id="graphs-container",
        #     children=[
        #         dcc.Loading(
        #             className="graph-wrapper",
        #             children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
        #         ),
        #         dcc.Loading(
        #             className="graph-wrapper",
        #             children=dcc.Graph(
        #                 id="graph-pie-confusion-matrix", figure=confusion_figure
        #             ),
        #         ),
        #     ],
        # ),


# Running the server
if __name__ == "__main__":
    dash_app.run_server(debug=True)