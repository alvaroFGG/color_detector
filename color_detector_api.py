from flask import Flask, request, jsonify
from colorthief import ColorThief
import cv2
import numpy as np

app = Flask(__name__)

def get_contour_color(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], 0, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(img, img, mask=mask)
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    return dominant_color

@app.route('/get_color', methods=['POST'])
def get_color():
    # Asegúrate de que el request sea una imagen
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha proporcionado un archivo de imagen'})

    file = request.files['file']

    # Asegúrate de que el archivo tenga una extensión de imagen válida
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Formato de imagen no admitido'})

    # Guarda el archivo temporalmente
    file_path = 'temp_image.png'
    file.save(file_path)

    # Obtiene el color del contorno de la prenda
    contour_color = get_contour_color(file_path)

    # Elimina el archivo temporal
    import os
    os.remove(file_path)

    return jsonify({'color': contour_color})

if __name__ == '__main__':
    app.run(debug=True)