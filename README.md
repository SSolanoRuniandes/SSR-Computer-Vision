Computer Vision Challenge - Santiago Solano



## Requirements



Tested on Python 3.8.0

* opencv-python
* numpy



## Execution



Execute the main.py file in a terminal opened in the working directory.



python main.py



When prompted to insert the relatives path to the partial image and full screenshot, you may use the photos found in the "images" folder. Possible inputs are:



./images/WEB\_prototype.png
./images/WEB.png
./images/SAP\_prototype.png
./images/SAP.png



Be sure to not remove the "models" folder from the working directory, since an OpenCV model is used to perform text detection. This model can also be downloaded from https://opencv.org/blog/text-detection-and-removal-using-opencv (EAST).

At the end, the tables identified are drawn over the screenshot, and saved in the same working directory.



## Solution description



For each image, the code performs simple operation trying to identify possible tables and its descriptors.



At first, simple preprocessing is applied to the screenshot, including binary thresholding, Sobel filters, dilatation, closure, skeletonization, and finally, Hough Probabilistic line detection. Once horizontal are vertical lines are detected, simple operations are performed to form groups of similar lines, this is, lines with the same orientation, same starting points in one coordinate, and same length, which form possible rows or columns in the table.



Also, the EAST algorithm is used in order to identify text boxes. These thext boxes are also organized in groups that could form rows or columns.



Then, the intersection between the bounding boxes of groups of similar horizontal and vertical lines, as well as rows and columns of text, are identified as possible tables, setting simple boundaries.



On each boundary, the text rows/columns and horizontal/vertical lines are used to narrow down possible boxes for columns and rows, as well as header.

From this procedure, a vector of numerical features is extracted, containing: average row height (irtw), average column width (irtw), number of columns, number of column lines, number of row lines (irth), header heigth (irtw) and average text size (irtw).
(irtw = in relation to table width)
(irth = in relation to table height)



Once the tables have been identified, and these features have been calculated, then the tables from both images can be compared using a simple normalized euclidean distance. If the distance is small, the tables are similar, and then the image with the boundaries found is generated, and otherwise, the tables do not match and nothing happens.

