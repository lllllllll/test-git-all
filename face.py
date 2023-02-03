from flask import Flask, request, jsonify
import cv2
import base64

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    image_data = request.json['image']

    image_data = base64.b64decode(image_data)

    image = np.frombuffer(image_data, dtype=np.uint8)

    # Convert the array to a image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_coordinates = []
    for (x, y, w, h) in faces:
        face_coordinates.append({'x': x, 'y': y, 'w': w, 'h': h})

    return jsonify({'face_coordinates': face_coordinates})

if __name__ == '__main__':
    app.run()