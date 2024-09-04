import cv2 as cv
import matplotlib as plt
import numpy as np
import math
clicked_pixel = None
points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global clicked_pixel
        clicked_pixel = img[y,x]
def draw_shape(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
fname =  input("Enter file name of image to be read\n")  
img = cv.imread(fname)
print("\nSelect from following options\n")
choice = input(" 1. Change Pixel C  olor \n 2. Select Closed Shape\n")

if choice == '1':

    cv.imshow("Image",img)
    cv.setMouseCallback("Image",mouse_callback)
    cv.waitKey(0)

    print("Enter the replacement color pixel value for R,G,B")
    r = input()
    g = input()
    b = input()

    all_clicked_pixel = np.all(img == clicked_pixel, axis=-1)
    img[all_clicked_pixel] = [b, g, r]
    cv.imshow("Image",img)
    cv.waitKey(0)

elif choice == '2':
    
    cv.imshow("Image",img)
    cv.setMouseCallback("Image",draw_shape)
    cv.waitKey(0)
    # Create a black image with the same size as the original image
    mask = np.zeros_like(img)
    # Create a polygon from the points and fill it in the mask
    cv.fillPoly(mask, np.array([points], dtype=np.int32), (255,255,255))
    # Bitwise-and the mask with the original image
    roi = cv.bitwise_and(img, mask)
    cv.imshow("Selected_Polygon", roi)
    cv.waitKey(0)

    transformChoice = input("Choose from following\n 1. Translation\n 2.Rotation\n 3.Scaling\n")

    
    if transformChoice == '1':
        trans_x = float(input("\nEnter the translation value for x direction\n"))
        trans_y = float(input("Enter the translation value for y direction\n"))

        # Translate the points of the polygon
        translated_points = [(x+trans_x, y+trans_y) for x, y in points]

        # Create a new image
        translated_img = np.zeros_like(img)

        # Draw the translated polygon on the new image
        cv.fillPoly(translated_img, np.array([translated_points], dtype=np.int32), (255,255,255))

        # Bitwise-and the new image with the original image
        roi = cv.bitwise_and(img, translated_img)

        # Display the original and translated images
        cv.imshow('Original Image', img)
        cv.imshow('Translated Image', roi)
        cv.waitKey(0)
        cv.destroyAllWindows()


    elif transformChoice == '2':

            # Define center of rotation
        center = (img.shape[1] / 2, img.shape[0] / 2)
        angle = float(input("\nEnter the angle of Rotation\n"))
        angle = math.radians(angle)

        # Define the rotation matrix
        rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]])

        # Rotate the points of the polygon
        rotated_points = [np.dot(rotation_matrix, np.array([x-center[0], y-center[1]])) + center for x, y in points]

        # Create a new image
        rotated_img = np.zeros_like(img)

        # Draw the rotated polygon on the new image
        cv.fillPoly(rotated_img, np.array([rotated_points], dtype=np.int32), (255,255,255))

        # Bitwise-and the new image with the original image
        roi = cv.bitwise_and(img, rotated_img)

        # Display the original and rotated images
        cv.imshow('Original Image', img)
        cv.imshow('Rotated Image', roi)
        cv.waitKey(0)
        cv.destroyAllWindows()



    elif transformChoice == '3':
        scale_x = float(input("\nEnter the Scale factor for x direction\n"))
        scale_y = float(input("Enter the Scale factor for y direction\n"))

        # Scale the points of the polygon
        scaled_points = [(x*scale_x, y*scale_y) for x, y in points]

        # Create a new image
        scaled_img = np.zeros_like(img)

        # Draw the scaled polygon on the new image
        cv.fillPoly(scaled_img, np.array([scaled_points], dtype=np.int32), (255,255,255))

        # Bitwise-and the new image with the original image
        roi = cv.bitwise_and(img, scaled_img)

        # Display the original and scaled images
        cv.imshow('Original Image', img)
        cv.imshow('Scaled Image', roi)
        cv.waitKey(0)
        cv.destroyAllWindows()
        