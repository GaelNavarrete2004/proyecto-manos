import requests

# URL de la API
url = "http://127.0.0.1:8000/recognize-sign"

# Ruta de la imagen que deseas enviar
file_path = "ab.png"

# Abrir la imagen en modo binario y enviarla en la solicitud
with open(file_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

# Mostrar la respuesta de la API
print(response.json())
