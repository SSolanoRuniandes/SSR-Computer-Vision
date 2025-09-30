import cv2 as cv
import os
import numpy as np

"""OpenCV Functions"""

# https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
def HoughProb_with_filter(Image, threshold=150, min_long_line=1000000):
    """
    This function uses Probabilistic Huough for line detection
    INPUTS:
        Image: Image that is the result of a Canny Transformation or a Sobel Filter
        threshold: Certainty threshold or count to identify a line (default 150)
        min_long_line: The function will delete lines with length smaller than this (default 1000000)
    OUTPUTS:
        np.array(filtered_lines): A numpy array with all the lines in the forma [x1, y1, x2, y2]
    """

    #Gets the lines with Probabilistic Hough
    lines = cv.HoughLinesP(image=Image, rho=1, theta=np.pi / 180, threshold=threshold) 
    
    # It applies a filter removing lines below a threshold
    filtered_lines = []
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0] #access format returned by cv.HoughLinesP
            length = np.sqrt( ((l[2]-l[0])**2) + ((l[3]-l[1])**2) ) #calculates line length
            if length >= min_long_line:
                filtered_lines.append(l)

    return np.array(filtered_lines)

# https://opencv.org/blog/text-detection-and-removal-using-opencv
def EAST(Image, new_h=320, new_w=320, conf_thresh=0.8, nms_thresh=0.4):
    """
    This function uses a pretrained model located in ./models/frozen_east_text_detection.pb to detect text boxes using EAST
    INPUTS:
        Image: Image in BGR
        new_h: New heigth for resize (default 320)
        new_w: New width for resize (default 320)
        conf_thresh: Parameter for EAST (default 0.8)
        nms_thresh: Parameter for EAST (default 0.4)
    OUTPUTS:
        boxesEAST_simplified: A list of lists. Each list has the boxes where text was spotted in the format [xmin, ymin, xmax, ymax]
    """
    
    #Resize and copy
    Image_resized = cv.resize(Image, (int(new_w), int(new_h)))
    annotated_east_image = Image_resized.copy()

    #Sets input image size
    inputSize = (int(new_w), int(new_h))
 
    #Loads pre-trained models
    my_folder = os.getcwd() #gets main folder
    my_path=os.path.join(my_folder, "./models/frozen_east_text_detection.pb") #sets the path adding the model 

    #Creates EAST object
    textDetectorEAST = cv.dnn_TextDetectionModel_EAST(my_path)

    #If want to force GPU usage
    """
    #Use GPU (cuda) if available
    try:
        textDetectorEAST.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        textDetectorEAST.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("Using GPU (CUDA) for inference")
    except:
        textDetectorEAST.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        textDetectorEAST.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        print("CUDA not available, using CPU")
    """

    #Set parameters for the model
    textDetectorEAST.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
    textDetectorEAST.setInputParams(1.0, inputSize, (123.68, 116.78, 103.94), True)
    #Detect text using the model
    boxesEAST, _ = textDetectorEAST.detect(Image_resized)
    boxesEAST_simplified=[]
    
    #Process EAST detected boxes
    for box in boxesEAST:
        box_array_int=np.array(box, np.int32)
        boxesEAST_simplified.append([ np.min(box_array_int[:, 0]), np.min(box_array_int[:, 1]), np.max(box_array_int[:, 0]), np.max(box_array_int[:, 1]) ])

    return boxesEAST_simplified #returns the boxes detected



def skeletonize(Image):
    """
    Skeletonization (shape thinning in a binary image) using iterative morphological operations: erosion, dilation, subtraction, and bitwise combination
    INPUT:
        Image: Image in gray scale or unidimensional from Canny/Sobel
    OUTPUT:
        skel: Resulting skeleton image
    """

    #Starts copying
    img = Image.copy() #starts with a copy
    size = np.size(img)
    #empty unidimensional image
    skel = np.zeros(img.shape, np.uint8) 
    #kernel used for morphological operations
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)) #kernel

    done = False
    while not done:
        #Skeletonization using erosion, dilation, subtraction and a bitwise function
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()

        #counts the amount of zero-pixels in the current image to decide stop of operation
        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True
    return skel