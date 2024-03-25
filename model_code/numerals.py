import cv2
import os
import tensorflow as tf

def findingWholeNumber(dict):
    array = [i[1] for i in dict]  # to convert this [(0, '7'), (180, '100'), (357, '5')]
    # this function works like this  6*100+5*10+3
    total = 0
    for i in range(0, len(array)):
        if (array[i] == "100" and (i != 0 or i != len(array) - 1)):
            continue
        if (array[i] == "10" and (i == 0)):
            total += 10
            continue
        total += int(array[i]) * (10 ** (len(array) - i - 1))

    return total


def findingContours(img):
    # Converting to inverse image
    thresh = ~img
    # Find the contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Converting back to normal image
    thresh = ~thresh
    return contours


def dilateErodeFunction(thresh, e_iteration1, d_iteration1):
    erosion = cv2.erode(thresh, None, iterations=e_iteration1)
    dilate = cv2.dilate(erosion, None, iterations=d_iteration1)
    return dilate


def removeNoice(thresh, cont, min_noice):
    # Loop over the contours
    for contour in cont:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        if area < min_noice:
            thresh = cv2.fillPoly(thresh, pts=[contour], color=(255))
    return thresh


def BGR2GRAY(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def drawingTheContours(thresh, cont, min_area, width, color):
    # Loop over the contours
    for contour in cont:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        if area < min_area:
            thresh = cv2.fillPoly(thresh, pts=[contour], color=(color))
    return thresh
def numerals_main_code(img):
    img_original = cv2.resize(img, (600, 200))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 199, 5)

    # Finding the contours
    contours = findingContours(thresh)

    # Define a minimum area for noise
    erode_val = 5
    dilate_val = 4
    min_noise1 = 102
    min_noise2 = 1000
    min_area = 5000

    # Removing the noise
    thresh = removeNoice(thresh, contours, min_noise1)

    # Eroding and dilating the image
    dilate = dilateErodeFunction(thresh, erode_val, dilate_val)  # ( image, erode, dilate )

    # -------finding conters after dilate the image-------------

    thresh = ~dilate

    # Find the contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Converting back to normal image
    thresh = ~thresh

    thresh_copy = thresh.copy()
    # Loop over the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        if area < min_noise2:
            thresh = cv2.fillPoly(thresh, pts=[contour], color=(255))
        if area < min_area:
            thresh_copy = cv2.fillPoly(thresh_copy, pts=[contour], color=(255))

    cont = findingContours(thresh_copy)
    path_to_pb_file = os.path.abspath('Numerals.model')
    loaded_model = tf.keras.models.load_model(path_to_pb_file)
    CATEGORIES = ['3', '4', '7', '9', '6', '5', '2', '8', '100', '10', '1']

    detected_numbers = {}  # to print it
    loop_inc = 0

    for contour in cont:
        # If the area is greater than or equal to the minimum area threshold, draw a green bounding box around it
        x, y, w, h = cv2.boundingRect(contour)
        im = thresh[y:y + h, x:x + w]
        IMG_SIZE = 50
        new_array = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
        image = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        prediction = loaded_model.predict([image])
        prediction1 = list(prediction[0])

        c = CATEGORIES[prediction1.index(max(prediction1))]
        detected_numbers[x] = c

        # Text printing
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_original, c, (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # saving the image
        loop_inc += 1

        img_final = cv2.resize(img_original, (600, 200))

        # sorting the dictionary based on x co-ordinates to get correct sequence of numbers
        sorted_detected_numbers = sorted(detected_numbers.items())
        print(sorted_detected_numbers)
        total = findingWholeNumber(sorted_detected_numbers)
        print(total)
        cv2.putText(img_final, str(total), (490, 190), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

    return img_final, str(total) # returning output image and predicted number
