from flask import Flask, Response, render_template
import cv2
import pytesseract

app = Flask(__name__)

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
cascade_path = "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Video Capture
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plate_roi = img_gray[y:y + h, x:x + w]
            plate_text = pytesseract.image_to_string(plate_roi, config='--psm 8')
            cv2.putText(img, plate_text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/home')
def printf():
    print("Hello World printed to the server console")
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)
