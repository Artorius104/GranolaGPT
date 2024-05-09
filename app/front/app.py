# To build and use the app, do the following :
# pip install dash
# pip install dash-bootstrap-components
# export REACT_VERSION=18.2.0
# pip install opencv-python
import base64

from dash import Dash, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
import cv2
from dash.exceptions import PreventUpdate

# app = Dash(__name__)
app = Dash(name=__name__, external_stylesheets=[dbc.themes.LUX])
# cap = cv2.VideoCapture(0)

navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 1", href="#"),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Are You Happy ?",
    brand_href="#",
    color="primary",
    dark=True,
)

app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Stack(
            [
                dbc.Row(
                    dbc.Col(
                        "Screenshots",
                        id="opencv-component",
                        width=4,
                        style={"text-align": "center"}
                    ),
                    justify="center",
                ),
                dbc.Row(
                    dbc.Col([
                        dbc.Button(
                            "Prendre une photo",
                            id="take-photo",
                            n_clicks=0,
                            style={"margin": "0 20px"},
                        ),
                        dbc.Button(
                            "Prendre une vidéo",
                            id="take-video",
                            n_clicks=0,
                            style={"margin": "0 20px"},
                        ),
                    ],
                        width="auto",
                    ),
                    justify="center",
                ),
                dbc.Row(
                    dbc.Col(
                        "Application de l'algorithme... Patience !",
                        id="loading-label",
                        width=4,
                        align="center",
                        style={"text-align": "center"}
                    ),
                    justify="center",
                ),
            ],
            id='page-inputs',
            gap=5
        ),
        html.Hr(),
        dbc.Stack(
            [
                dbc.Row(
                    dbc.Col(
                        "Résultat de l'algorithme",
                        id="page-outputs",
                        width=4,
                        align="center",
                        style={"text-align": "center"}
                    ),
                    justify="center",
                ),
            ],
            id="page_outputs",
            gap=3
        ),
    ], fluid=True)
])


# @callback(
#     Output("opencv-component", "children"),
#     [Input("take-photo", "n_clicks")]
# )
# def update_webcam_content(n_clicks):
#     print("PRINT 1")
#     if n_clicks is None:
#         raise PreventUpdate
#     print("PRINT 2")
#
#     ret, frame = cap.read()
#
#     ret, buffer = cv2.imencode('.jpg', frame)
#     frame_bytes = buffer.tobytes()
#     encoded_image = base64.b64encode(frame_bytes)
#     src = 'data:image/jpeg;base64,{}'.format(encoded_image.decode())
#
#     return html.Img(src=src)

if __name__ == '__main__':
    app.run(
        debug=True,
        dev_tools_hot_reload=True
    )
