# Importing the required libraries
import cv2
import numpy as np
import os
import tensorflow as tf
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def findingContours(img):
    # Converting to inverse image
    thresh = ~img
    # Find the contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Converting back to normal image
    thresh = ~thresh
    return contours


def drawingBoxes(thresh, min_area, max_area):
    cont = findingContours(thresh)
    for conts in cont:
        area = cv2.contourArea(conts)
        if area > min_area and area < max_area:
            # If the area is greater than or equal to the minimum area threshold, draw a green bounding box around it
            x, y, w, h = cv2.boundingRect(conts)

            # Drawing the bounding box
            cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return thresh


def segmentation(thresh):
    cont = findingContours(thresh)
    images = []
    for conts in cont:
        x, y, w, h = cv2.boundingRect(conts)
        im = thresh[y:y + h, x:x + w]
        images.append(im)
    return images


def dilateErodeFunction(thresh, e_iteration1, d_iteration1):
    erosion = cv2.erode(thresh, None, iterations=e_iteration1)
    dilate = cv2.dilate(erosion, None, iterations=d_iteration1)
    return dilate


def drawingTheContours(thresh, cont, min_area, width):
    # Loop over the contours
    for contour in cont:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        if area < min_area:
            thresh = cv2.drawContours(thresh, contour, 0, (0, 0, 0), width)
    return thresh


def binarization(gray, range1, range2):
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, range1, range2)
    return thresh


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


def findingTheNumberOfCharacters(width, avarageWidth, height, avarageHeight):
    noOfCharactersInRow = 1
    noOfCharactersInColumn = 1
    while (True):
        if (avarageWidth < width):
            noOfCharactersInRow = noOfCharactersInRow + 1
            width = width - avarageWidth
        else:
            break
    while (True):
        if (avarageHeight < height):
            noOfCharactersInColumn = noOfCharactersInColumn + 1
            height = height - avarageHeight
        else:
            break
    noOfCharacters = [noOfCharactersInRow, noOfCharactersInColumn]
    return noOfCharacters


def averageAreaWidthHeight(thresh, min_area, max_area):
    # disjoiningTheJoints(thresh)
    thresh = ~thresh

    # Find the contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Converting back to normal image
    thresh = ~thresh

    # Defining a array for the area of the contours
    area = []

    # loop for contours
    for cont in contours:
        area.append(cv2.contourArea(cont))

    w1 = []
    h1 = []
    for j in contours:
        x, y, w, h = cv2.boundingRect(j)
        w1.append(w)
        h1.append(h)

    totalArea = 0
    count = 0
    avarageWidth = 0
    avarageHeight = 0

    j = 0
    for i in area:
        if np.logical_and(i > min_area, i < max_area).all():
            avarageWidth = avarageWidth + w1[j]
            avarageHeight = avarageHeight + h1[j]
            totalArea = totalArea + i
            count = count + 1
            j = j + 1
        else:
            j = j + 1
            continue

    avarage = [int(totalArea / count), int(avarageWidth / count), int(avarageHeight / count)]
    return avarage


def checkJointed(thresh, cont, averageArea, averageWidth, averageHeight):
    # Finding the area
    area = cv2.contourArea(cont)

    # Condition for finding the jointed charcter
    if (area > averageArea):
        # Finding the coordinates of the jointed characters
        x, y, w, h = cv2.boundingRect(cont)

        # Finding whether the jointed character is jointed horizontal or vertical
        if (w > h):

            # Defining the empty list for the area of lift and right side of the point
            lArea = []
            rArea = []

            # Defining the point
            point = int(w / 2)

            # Segmenting the jointed characters to find the true jointed characters
            im1 = thresh[y:y + h, x:x + point]
            im2 = thresh[y:y + h, x + point:point * 2 + x]

            # Finding the contours
            cont1 = findingContours(im1)
            cont2 = findingContours(im2)

            for cont1s in cont1:
                # Variables for the segmented charecter area
                lArea.append(cv2.contourArea(cont1s))
            for cont2s in cont2:
                # Variables for the segmented charecter area
                rArea.append(cv2.contourArea(cont2s))

            # Finding the shape of the jointed image(*NOT SEGMENTED IMAGE)
            _, w1 = im1.shape
            _, w2 = im2.shape

            # Filtering for Finding the true jointed characters
            if (w1 > (averageWidth - (averageWidth * 0.40)) and max(lArea) > (
                    averageArea - (averageArea * 0.68)) and max(rArea) > (averageArea - (averageArea * 0.68))):
                return 1
            else:
                return 0
        elif (h > w):

            # Defining the empty list for the area of lift and right side of the point
            lArea = []
            rArea = []

            # Defining the point
            point = int(h / 2)

            # Segmenting the jointed characters to find the true jointed characters
            im1 = thresh[y:y + point, x:x + w]
            im2 = thresh[y + point: y + point * 2, x:x + w]

            # Finding the contours
            cont1 = findingContours(im1)
            cont2 = findingContours(im2)

            for cont1s in cont1:
                # Variables for the segmented charecter area
                lArea.append(cv2.contourArea(cont1s))
            for cont2s in cont2:
                # Variables for the segmented charecter area
                rArea.append(cv2.contourArea(cont2s))

            # Finding the shape of the jointed image(*NOT SEGMENTED IMAGE)
            h1, _ = im1.shape

            # Filtering for Finding the true jointed characters
            if (h1 > (averageHeight - (averageHeight * 0.25)) and max(lArea) > (
                    averageArea - (averageArea * 0.68)) and max(rArea) > (averageArea - (averageArea * 0.68))):
                return 1
            else:
                return 0
        else:
            # Need to figureItOut what to do when w==h !
            pass
    else:
        return 0


def prediction(loaded_model, im, CATEGORIES):
    IMG_SIZE = 50
    new_array = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    image = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    prediction = loaded_model.predict([image])

    prediction1 = list(prediction[0])

    c = CATEGORIES[prediction1.index(max(prediction1))]
    return c


def convert_to_binary(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a mask based on the brown color range
    lower_brown = np.array([5, 60, 135])
    upper_brown = np.array([255, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combine the thresholded image with the brown color mask
    result = cv2.bitwise_and(thresholded, mask)

    # Display the original and binary images

    R1 = cv2.erode(result, None, 1)

    contours1, vect = cv2.findContours(R1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours1:
        area = cv2.contourArea(contour)
        if (area < 25):
            G1 = cv2.fillPoly(R1, pts=[contour], color=(255))
    G1 = cv2.erode(G1, None, 1)

    contours1, vect = cv2.findContours(G1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours1:
        area = cv2.contourArea(contour)
        if (area < 300):
            G11 = cv2.fillPoly(G1, pts=[contour], color=(255))

    return G11


def preprocessing_g(gray):
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 21)
    # thresholded = thresholded[:, 10:]  # to remove the left corner of the palm leaf
    erode = cv2.erode(thresholded, None, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilate = cv2.dilate(erode, kernel, 1)
    return dilate


# _______________________________________________________________________________________________________________________
def characters_main_code(img):
    # Reading the image
    img_original = img[:, 80:]
    # Making copies of the original image for the future use
    img = img_original.copy()
    img_test = img_original.copy()

    # Converting to binary
    gray = BGR2GRAY(img)
    thresh = preprocessing_g(gray)
    # Defining the fixed values
    min_noise = 100
    min_area = 250
    max_area = 2000

    # Finding the contours
    cont = findingContours(thresh)

    # Removing the noise
    noiseRemoved = removeNoice(thresh, cont, min_noise)

    thresh1 = thresh

    # Finding the average area, height, and width of the characters
    averageArea, averageWidth, averageHeight = averageAreaWidthHeight(noiseRemoved, min_area, max_area)

    # Finding the contours
    conts = findingContours(noiseRemoved)

    # Loding the model
    path_to_pb_file = os.path.abspath('./Thirukkural.model')
    loaded_model = tf.keras.models.load_model(path_to_pb_file)
    CATEGORIES = ['ஆ', 'உ', 'ஐ', 'இ', 'ஏ', 'எ', 'ஒ', 'கி', 'க', 'அ', 'சி', 'டூ', 'ட', 'ஞ', 'டு', 'சு', 'ங', 'கு', 'டி',
                  'ச', 'ணு', 'து', 'ண', 'ணீ', 'தீ', 'ந', 'தி', 'நி', 'த', 'ணி', 'நீ', 'னா', 'பி', 'பு', 'னி', 'னீ',
                  'னு', 'பீ', 'ன', 'ப', 'பூ', 'மூ', 'யு', 'மீ', 'ம', 'யி', 'யூ', 'ய', 'மு', 'மி', 'ல', 'றா', 'லு', 'று',
                  'ரு', 'ற', 'லி', 'ர', 'ரி', 'றி', 'ழு', 'ளி', 'வு', 'ழ', 'ழி', 'வ', 'ா', 'ள', 'வி', 'ளு', 'ே', 'ெ',
                  'ை']

    # Drawing the character
    I1 = Image.fromarray(img)
    draw = ImageDraw.Draw(I1)
    # specified font size
    font = ImageFont.truetype(r'./CODE2000.TTF', 15)

    letter_counter = {}
    # Looping for disjointing the characters
    for cont in conts:
        area = cv2.contourArea(cont)
        if area > max_area:
            continue
        # Checking wheather the character is jointed or not
        elif checkJointed(thresh1, cont, averageArea + 100, averageWidth + 30, averageHeight):

            # Find the coordinates of the contours
            x, y, w, h = cv2.boundingRect(cont)

            # Finding the number of characters jointed horizontaly and vertically
            noRow, noCol = findingTheNumberOfCharacters(w, averageWidth + 20, h, averageHeight + 50)

            # Defining the incrementing variables
            k = 0
            temp = 500

            # Defining the threshold values
            for i in range(-20, 20, 2):

                # Defining the average cutting points
                point1 = (w // noRow) - i

                # Croping the image
                im = thresh1[y:y + h, x + point1:x + point1 + 2]

                # Finding the contours
                contours = findingContours(im)

                # Finding the accurate cutting point
                if len(contours) == 1:

                    _, _, _, h1 = cv2.boundingRect(contours[0])

                    # Finding the thinest point to cut
                    if temp > h1:
                        cuttingPoint = point1 + 2 - 1
                        temp = h1

                # The condition for, if it detects more than one contour
                elif len(contours) >= 2:
                    hA = []
                    for contour in contours:
                        _, _, _, h1 = cv2.boundingRect(contour)
                        hA.append(h1)
                    h1 = max(hA)
                    # Finding the thinest point to cut
                    if temp > h1:
                        cuttingPoint = point1 + 2 - 1
                        temp = h1

                k += 1

            # Cutting the characters using the accurate cutting points
            firstPoint = 0
            secondPoint = cuttingPoint
            m = 0
            while m < noRow and firstPoint < x + w:
                # Setting a different cutting parameters for the last box
                if (m == noRow - 1):
                    croppedImage = thresh[y:y + h, x + firstPoint:x + secondPoint]
                    if np.any(croppedImage) == True:
                        c = prediction(loaded_model, croppedImage, CATEGORIES)
                        print(c)
                        # Adding the images to the image list
                        draw.rectangle(((x + firstPoint, y), (x + w, y + h)), outline="green")

                        draw.text((x - 15 + firstPoint, y - 15), c, font=font, align="left")

                        letter_counter[x] = {y: c}
                        firstPoint = firstPoint + cuttingPoint
                        secondPoint = secondPoint + cuttingPoint
                    m += 1

                else:
                    croppedImage = thresh[y:y + h, x + firstPoint:x + secondPoint]
                    if np.any(croppedImage) == True:
                        c = prediction(loaded_model, croppedImage, CATEGORIES)
                        print(c)
                        # Adding the images to the image list
                        draw.rectangle(((x + firstPoint, y), (x + secondPoint, y + h)), outline="green")
                        draw.text((x - 15 + firstPoint, y - 15), c, font=font, align="left")

                        letter_counter[x] = {y: c}
                        firstPoint = firstPoint + cuttingPoint
                        secondPoint = secondPoint + cuttingPoint
                    m += 1

            # Cutting the vertically jointed characters
            if noCol > 1:

                # Defining the incrementing variables
                j1 = 5
                temp1 = 100
                k = 0

                # Finding the perfect spot to cut
                for i in range(-30, 30, 5):
                    point2 = (h // 2) - i
                    im3 = thresh[y + point2:y + point2 + j1, x:x + w]
                    cont4 = findingContours(im3)
                    cuttingPoint1 = point2
                    if len(cont4) == 1:
                        x4, y4, w4, h4 = cv2.boundingRect(cont4[0])

                        if temp1 > w4:
                            cuttingPoint1 = point2 + j1 - 1
                            temp1 = h4

                    elif len(cont4) >= 2:
                        hA1 = []
                        for cont41 in cont4:
                            _, _, _, h11 = cv2.boundingRect(cont41)
                            hA1.append(h11)
                        h11 = max(hA1)
                        # Finding the thinest point to cut
                        if temp > h11:
                            cuttingPoint1 = point2 + 2 - 1
                            temp = h11

                    k += 1

                # Adding the images to the image[] list
                img1 = img[y:y + cuttingPoint1, x:x + w]
                c1 = prediction(loaded_model, img1, CATEGORIES)
                print(c1)

                img2 = img[cuttingPoint1:cuttingPoint1 + (h - cuttingPoint1), x:x + w]
                c2 = prediction(loaded_model, img2, CATEGORIES)
                print(c2)

                # Drawing the bounding box
                draw.rectangle(((x, y), (x + w, y + cuttingPoint1)), outline="green")
                draw.text((x - 15, y - 15), c1, font=font, align="left")

                letter_counter[x] = {y: c1}
                draw.text((x - 15, y - 15 + cuttingPoint1), c2, font=font, align="left")
                letter_counter[x] = {y: c2}
        else:

            # Detecting the coordinates of the contours
            x, y, w, h = cv2.boundingRect(cont)

            img3 = thresh1[y:y + h, x:x + w]
            c3 = prediction(loaded_model, img3, CATEGORIES)
            print(c3)
            # Adding the images to the image list
            draw.rectangle(((x, y), (x + w, y + h)), outline="green")
            draw.text((x - 15, y - 15), c3, font=font, align="left")

            letter_counter[y] = {x: c3}

    return I1  # returning the output image
