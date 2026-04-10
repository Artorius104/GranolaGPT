from load_lib import *

# To build and use the app, do the following :
# export REACT_VERSION=18.2.0
import os
import base64
import io

from dash import Dash, html, Output, Input, dcc, callback_context, no_update
import dash_bootstrap_components as dbc
import cv2
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np

app = Dash(name=__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Are You Happy ?"


for i in range(0, 5):
    camera = cv2.VideoCapture(i)
    if camera.isOpened():
        break
    print(f"Camera index {i} not available")

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
            dbc.Row(    # AFFICHAGE DU FLUX VIDEO
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
            dcc.Interval(id='interval-component', interval=100, n_intervals=0), # FLUX VIDEO
            dbc.Stack([
                dbc.Row(
                    dbc.Col(
                        dbc.DropdownMenu(
                            [
                                dbc.DropdownMenuItem(
                                    "A button", id="dropdown-button", n_clicks=0
                                ),
                                dbc.DropdownMenuItem(
                                    "Internal link", href="/docs/components/dropdown_menu"
                                ),
                                dbc.DropdownMenuItem(
                                    "External Link", href="https://github.com"
                                ),
                                dbc.DropdownMenuItem(
                                    "External relative",
                                    href="/docs/components/dropdown_menu",
                                    external_link=True,
                                ),
                            ],
                            label="Select the model to use",
                        ),
                        width="auto",
                    ),
                    justify="center"
                ),
                dbc.Row([
                    dbc.Col(   # BOUTON PHOTO
                        dbc.Button(
                            "Prendre une photo",
                            id="get-photo",
                            n_clicks=0,
                            style={"margin": "0 20px 0 0"},
                        ),
                        width="auto"
                    ),
                    dbc.Col(
                        dcc.Upload(  # UPLOAD BOUTON
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
                        width="auto"
                    )],
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
                        html.Div(id="model-result"),
                        id="model-result-container",
                        width="auto",
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


# CALLBACKS + UTILS FOR CALLBACKS
def parse_contents(contents):
    """Récupération de l'image via upload"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return image

@app.callback(
    Output('confirm-image', 'disabled'),
    Input('upload-image', 'contents')
)
def enable_confirm_button(contents):
    """Active/Désactive le bouton Confirmation"""
    if contents is not None:
        return False
    else:
        return True

_HERE         = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH     = os.path.join(_HERE, "..", "..", "my_lib", "target", "release", "libmy_lib.so")
MLP_IMAGE_SIZE = 32                        # doit correspondre à train_mlp.py
MLP_INPUT_DIM  = MLP_IMAGE_SIZE ** 2 * 3  # 32*32*3 = 3072

lib = load_my_lib(_LIB_PATH)
model_path = os.path.join(_HERE, "best_model.json")
try:
    mlp_ptr = load_mlp_model(lib, model_path)
    print(f"Modèle MLP chargé : {model_path}")
except Exception as e:
    print(f"Avertissement : impossible de charger le modèle ({e}). L'application démarre sans modèle.")
    mlp_ptr = None

EMOTIONS = {0: "HAPPY", 1: "NEUTRAL", 2: "SAD"}


def capture_frame():
    """Capture l'image caméra, prédit l'émotion et l'encode en base64."""
    rect_x, rect_y, rect_w, rect_h = 200, 150, 200, 200

    if not camera.isOpened():
        return None

    ret, frame = camera.read()
    if not ret or frame is None:
        return None

    cropped_frame = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
    resized_frame = cv2.resize(cropped_frame, (MLP_IMAGE_SIZE, MLP_IMAGE_SIZE))
    normalized_frame = np.array(resized_frame, dtype=np.float64) / 255.0

    # Prédiction en temps réel (si modèle chargé)
    if mlp_ptr is not None:
        preds = predict_with_mlp(mlp_ptr, normalized_frame, MLP_INPUT_DIM)
        emotion_label = EMOTIONS[int(np.argmax(preds))]
        cv2.putText(frame, emotion_label,
                    (rect_x, rect_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

@app.callback(
    Output('live-video-feed', 'src'),
    [Input('interval-component', 'n_intervals')]
)
def update_live_feed(n_clicks):
    """Update du flux vidéo"""
    return capture_frame()

@app.callback(
    [
        Output('image-uploaded', 'children'),
        Output('model-result', 'children')
    ],[
        Input('upload-image', 'contents'),
        Input('get-photo', 'n_clicks')
    ]
)
def update_image_uploaded(contents, n_clicks):
    """Capture d'écran du flux vidéo / Upload d'une image"""
    triggered_id = [p['prop_id'] for p in callback_context.triggered][0]

    no_model_msg = html.Div(
        html.H4("Aucun modèle chargé. Lancez d'abord : python train_mlp.py", style={"color": "orange"})
    )

    def predict_emotion(image_pil):
        """Prédit l'émotion à partir d'une image PIL."""
        img = image_pil.convert('RGB').resize((MLP_IMAGE_SIZE, MLP_IMAGE_SIZE))
        arr = np.array(img, dtype=np.float64) / 255.0
        preds = predict_with_mlp(mlp_ptr, arr, MLP_INPUT_DIM)
        return EMOTIONS[int(np.argmax(preds))]

    if triggered_id == 'upload-image.contents':
        if contents is not None:
            if mlp_ptr is None:
                return html.Div([
                    html.H5("Image obtenue :"),
                    html.Img(src=contents, style={'width': '50%'})
                ]), no_model_msg

            image = parse_contents(contents)
            emotion = predict_emotion(image)
            return html.Div([
                html.H5("Image obtenue :"),
                html.Img(src=contents, style={'width': '50%'})
            ]), html.Div(html.H1(f"You are {emotion} !"))
        else:
            return html.Div([html.H5("Aucune image")]), html.Div()

    elif triggered_id == 'get-photo.n_clicks':
        if n_clicks > 0:
            src_image = capture_frame()
            if src_image is None:
                return html.Div([html.H5("Caméra non disponible.")]), html.Div()
            if mlp_ptr is None:
                return html.Div([
                    html.H5("Image obtenue :"),
                    html.Img(src=src_image)
                ]), no_model_msg

            _, content_string = src_image.split(',')
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            emotion = predict_emotion(image)
            return html.Div([
                html.H5("Image obtenue :"),
                html.Img(src=src_image)
            ]), html.Div(html.H1(f"You are {emotion} !"))

    return html.Div([
        html.H5("Aucune image")
    ]), html.Div()
    

if __name__ == '__main__':
    app.run(
        debug=True,
        use_reloader=False,   # évite le 2ème processus qui monopolise la caméra
        dev_tools_hot_reload=False
    )
