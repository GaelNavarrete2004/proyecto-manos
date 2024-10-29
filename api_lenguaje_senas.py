from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Configuración de la API
app = FastAPI()

# Configuración de CORS
origins = ["http://35.160.120.126:5500"]  # Especifica el origen permitido
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo de lenguaje de señas
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Configuración de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Diccionario de etiquetas
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

# Estructura de la respuesta de la API
class SignResponse(BaseModel):
    recognized_text: str

# Endpoint para procesar la imagen y reconocer el signo
@app.post("/recognize-sign", response_model=SignResponse)
async def recognize_sign(file: UploadFile = File(...)):
    # Lee la imagen recibida
    image_data = await file.read()
    # Convierte la imagen para OpenCV
    np_img = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convertir a RGB y procesar con MediaPipe
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Predecir el carácter
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        return {"recognized_text": predicted_character}
    
    return {"recognized_text": "No se detectaron manos en la imagen"}

# Para correr la API en local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="35.160.120.126", port=5500)
