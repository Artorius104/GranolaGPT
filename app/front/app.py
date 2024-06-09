# To build and use the app, do the following :
# export REACT_VERSION=18.2.0
import base64
import io

from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import cv2
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np

app = Dash(name=__name__, external_stylesheets=[dbc.themes.LUX])
# cap = cv2.VideoCapture(0)

# NAVBAR
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

# LAYOUT
app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Stack([
            dbc.Row(    # AFFICHAGE DE LA PARTIE WEBCAM
                dbc.Col(
                    "Screenshots",
                    id="opencv-component",
                    width=4,
                    style={"text-align": "center"}
                ),
                justify="center",
            ),
            dbc.Stack([
                dbc.Row([
                    dbc.Col([   # BOUTONS PHOTO ET VIDEO
                        dbc.Button(
                            "Prendre une photo",
                            id="take-photo",
                            n_clicks=0,
                            style={"margin": "0 20px 0 0"},
                        ),
                        dbc.Button(
                            "Prendre une vidéo",
                            id="take-video",
                            n_clicks=0,
                            style={"margin": "0 0 0 20px"},
                        )],
                        width="auto",
                    )],
                    justify="center"
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Upload( # UPLOAD BOUTON
                            id="upload-image",
                            children=dbc.Button([
                                'Drag and Drop or ',
                                html.A('Select Files')],
                                outline=True,
                                style={
                                    'border': '2px solid black',
                                }
                            ),
                        ),
                        width="auto",
                    ),
                    justify="center"
                )],
                gap=3
            ),
            dbc.Row(    # AFFICHAGE DE L'IMAGE OBTENUE
                dbc.Col(
                    html.Div(id='output-image-upload'),
                    id="image-outputs",
                    width="auto",
                    align="center",
                    style={"text-align": "center"}
                ),
                justify="center",
            ),
            dbc.Row([   # BOUTON CONFIRMATION
                dbc.Col([
                    dbc.Button(
                        "Confirmation",
                        id="confirm-image",
                        n_clicks=0,
                        disabled=True
                    )],
                    width="auto",
                )],
                justify="center",
            ),
            dbc.Row(
                dbc.Col(    # MESSAGE DE TRAITEMENT DE L'IMAGE PAR LE MODELE
                    "Application de l'algorithme... Patience !",
                    id="loading-label",
                    width=4,
                    align="center",
                    style={"text-align": "center"}
                ),
                justify="center",
            )],
            id='page-inputs',
            gap=4
        ),
        html.Hr(),
        dbc.Stack(  # AFFICHAGE DU RESULTAT
            [
                dbc.Row(
                    dbc.Col(
                        "Résultat de l'algorithme",
                        id="algo-text",
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


# RECUPERATION DE L'IMAGE VIA UPLOAD
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return image

# CALLBACKS
@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents')
)
def image_output(contents):
    if contents is not None:
        image = parse_contents(contents)
        image_array = np.array(image)
        return html.Div([
            html.H5("Image obtenue :"),
            html.Img(src=contents, style={'width': '50%'})
        ])
    else:
        return html.Div([
            html.H5("Aucune image")
        ])

@app.callback(
    Output('confirm-image', 'disabled'),
    Input('upload-image', 'contents')
)
def enable_confirm_button(contents):
    if contents is not None:
        return False
    else:
        return True

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
#     src = 'scrapping:image/jpeg;base64,{}'.format(encoded_image.decode())
#
#     return html.Img(src=src)

if __name__ == '__main__':
    app.run(
        debug=True,
        dev_tools_hot_reload=True
    )
