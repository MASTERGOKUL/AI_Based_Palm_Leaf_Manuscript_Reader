import base64
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from model_code.numerals import numerals_main_code
from model_code.characters import characters_main_code

app = Flask(__name__)


# numerals code
def numerals(image):
    # reading the image as cv2 file from the web encoded image
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = img.copy()
    img2 = cv2.resize(img2, (600, 200))
    final_output, total = numerals_main_code(img)
    # Encode the processed image to base64 to export the output
    _, img_encoded = cv2.imencode('.jpg', final_output)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    _, img_encoded2 = cv2.imencode('.jpg', img2)
    img_base642 = base64.b64encode(img_encoded2).decode('utf-8')
    # Return the base64 encoded image
    return img_base64, img_base642, total


# characters
def characters(image):
    # reading the image as cv2 file from the web encoded image
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = img.copy()
    # img2=cv2.resize(img2,(600,200))
    final_output = characters_main_code(img)
    # Encode the processed image to base64 to export the output
    final_output = np.asarray(final_output)
    _, img_encoded = cv2.imencode('.jpg', final_output)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    _, img_encoded2 = cv2.imencode('.jpg', img2)
    img_base642 = base64.b64encode(img_encoded2).decode('utf-8')
    # Return the base64 encoded image
    return img_base64, img_base642


def characters_preprocess(image):
    # reading the image as cv2 file from the web encoded image
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = img.copy()
    final_output = characters_main_code(img)
    final_output = np.asarray(final_output)
    _, img_encoded = cv2.imencode('.jpg', final_output)
    img_base64_final = base64.b64encode(img_encoded).decode('utf-8')
    _, img_encoded2 = cv2.imencode('.jpg', img2)
    img_base64_original = base64.b64encode(img_encoded2).decode('utf-8')
    # Return the base64 encoded image
    return img_base64_final, img_base64_original

# login page
@app.route('/', methods=['POST', 'GET'])
def login():
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        if email == "tamil@gmail.com" and password == "123":
            return redirect(url_for('characters_upload'))
        else:
            return render_template('/login.html')
    return render_template('/login.html')

# numerals page
@app.route('/numerals', methods=['GET', 'POST'])
def numerals_upload():
    if request.method == 'POST' and 'image' in request.files:
        original_image = request.files['image']
        output_image, image, total = numerals(original_image)
        return render_template('/upload.html', output_image=output_image, image=image,
                               file_name=original_image.filename, total=total)
    return render_template('/upload.html')


# characters page
@app.route('/characters', methods=['GET', 'POST'])
def characters_upload():
    if request.method == 'POST' and 'image' in request.files:
        original_image = request.files['image']
        output_image, image = characters(original_image)
        return render_template('/characters.html', output_image=output_image, image=image,
                               file_name=original_image.filename)
    return render_template('/characters.html')


if __name__ == '__main__':
    app.run(debug=True)
