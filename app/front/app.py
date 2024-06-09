# To build and use the app, do the following :
# export REACT_VERSION=18.2.0
import base64
import io

from dash import Dash, html, Output, Input, dcc, callback_context
import dash_bootstrap_components as dbc
import cv2
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np

app = Dash(name=__name__, external_stylesheets=[dbc.themes.LUX])

camera = cv2.VideoCapture(0)

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
                    html.Img(
                        id="live-video-feed",
                        alt="Screenshot !"
                    ),
                    id="opencv-component",
                    width="auto",
                    align="center",
                    style={"text-align": "center"}
                ),
                justify="center",
            ),
            dcc.Interval(id='interval-component', interval=100, n_intervals=0),  # Ajout de l'Interval component
            dbc.Stack([
                dbc.Row([
                    dbc.Col([   # BOUTONS PHOTO ET VIDEO
                        dbc.Button(
                            "Prendre une photo",
                            id="get-photo",
                            n_clicks=0,
                            style={"margin": "0 20px 0 0"},
                        ),
                        dbc.Button(
                            "Prendre une vidéo",
                            id="get-video",
                            n_clicks=0,
                            disabled=True,
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
                    html.Div(id='image-uploaded'),
                    id="image-uploaded-container",
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
                        id="algo-result",
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
    Output('confirm-image', 'disabled'),
    Input('upload-image', 'contents')
)
def enable_confirm_button(contents):
    if contents is not None:
        return False
    else:
        return True

def capture_frame():
    success, frame = camera.read()
    if success:
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return f'data:image/jpeg;base64,{encoded_image}'
    return None

@app.callback(
    Output('live-video-feed', 'src'),
    [Input('interval-component', 'n_intervals')]
)
def update_live_feed(n_clicks):
    return capture_frame()

@app.callback(
    Output('image-uploaded', 'children'),
    [
        Input('upload-image', 'contents'),
        Input('get-photo', 'n_clicks')
    ]
)
def update_image_uploaded(contents, n_clicks):
    triggered_id = [p['prop_id'] for p in callback_context.triggered][0]

    if triggered_id == 'upload-image.contents':
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

    elif triggered_id == 'get-photo.n_clicks':
        if n_clicks > 0:
            src_image = capture_frame()
            return html.Div([
                html.H5("Image obtenue :"),
                html.Img(src=src_image)
            ])

    return html.Div([
        html.H5("Aucune image")
    ])

if __name__ == '__main__':
    app.run(
        debug=True,
        dev_tools_hot_reload=True
    )
