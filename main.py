import cv2 as cv
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from boxlines_functions import *
from opencv_functions import *



def euclidean_distance(v1, v2):
    """
    Returns the normalized euclidean distance between two feature vectors
    INPUTS:
        v1: list of features
        v2: list of features
    OUTPUTS:
        dist: Eculidean distance normalized [0,1]. 0 is very similar, 1 is very different.
    """

    v1, v2 = np.array(v1), np.array(v2) #converts to arrays
    
    #gets maximum and minimum values of each vector
    v1_min = np.min(v1)
    v1_max = np.max(v1)
    v2_min = np.min(v2)
    v2_max = np.max(v2)
    
    #Normalized vectors
    v1_scaled = (v1) / (v1_max-v1_min)
    v2_scaled = (v2) / (v2_max-v2_min)
    
    #Distance in range [0,1]
    dist = np.linalg.norm(v1_scaled - v2_scaled) / np.sqrt(len(v1_scaled))
    return dist


def DrawTables(Tables):
    """
    Function that saves an image in the working directory with the highlights from the Table. 
    INPUTS: 
        Tables: List of dictionries containing each a table. 
    OUTPUTS:
        None
        Image saved on the working directory. 
    """

    #If Tables is empty returns
    if not Tables:
        return      

    #Sets path
    my_folder = os.getcwd()  #gets main folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"MatchingTables_output_{timestamp}.png"  #image name
    my_path = os.path.join(my_folder, filename)  # full path

    #makes sure the path does not exist already
    counter=1
    while os.path.exists(my_path):
        filename = f"MatchingTables_output_{timestamp}_{counter}.png"
        my_path = os.path.join(my_folder, filename)
        counter +=1

    #Gets the image and draws boxes
    for table in Tables:
        Image = table['Image'].copy()
             
        #Rows
        boxes = table['Row Boxes']
        for xmin, ymin, xmax, ymax in boxes:
            cv.rectangle(Image, (xmin, ymin), (xmax, ymax), color=(0,255,255), thickness=4)
        
        #Columns
        boxes = table['Column Boxes']
        for xmin, ymin, xmax, ymax in boxes:
            cv.rectangle(Image, (xmin, ymin), (xmax, ymax), color=(0,0,0), thickness=2)
        overlay = Image.copy()
        alpha = 0.3  # transparecy
        for xmin, ymin, xmax, ymax in boxes:
            step = 15  # Distance between lines
            #for x in range(xmin, xmax, step):
            #    cv.line(overlay, (x, ymin), (x, ymax), (0, 0, 0), 1)  # Vertical lines
            for y in range(ymin, ymax, step):
                cv.line(overlay, (xmin, y), (xmax, y), (0, 0, 0), 1)  # Horizontal lines
        Image = cv.addWeighted(overlay, alpha, Image, 1 - alpha, 0)

        #Table boundaries
        xmin, ymin, xmax, ymax = table['Table Boundaries']
        cv.rectangle(Image, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=4)
        
        #Header
        boxes = table['Header Boundaries']
        for xmin, ymin, xmax, ymax in boxes:
            cv.rectangle(Image, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=4)

    cv.imwrite(my_path, Image)
    print(f"Screenshot {filename} with matching tables saved on: {my_path}")


def IdentifyTables(image_name): 
    """
    Main Function to identify tables and their features. 
    INPUT: 
        my_path: Path to the image. 
    OUTPUT:
        DETECTED_TABLES: List of dictionaries with the tables detected and their features.
        Each entrance in the list is a different table.
        The dictionary describes the table as follows.
        numeric_features_vector has numeric qualities (adjusted to size) for table comparisson

        numeric_features_vector=np.array([average_row_height, average_column_width, number_columns, number_column_lines, number_row_lines, header_heigth_to_table_width, average_text_size])

        this_table_features = {
            "Table Boundaries": table,
            "Header Boundaries": this_table_header,
            "Row Boxes": this_table_row_boxes,
            "Column Boxes": this_table_columns_boxes,
            "Row Lines": this_table_row_lines,
            "Column Lines": this_table_column_lines,
            "Columns from Header": this_table_columns_from_header,
            "Text Columns": this_table_text_columns,
            "Text Rows": this_table_text_rows,
            "Image": myPicture,
            "Feature Vector": numeric_features_vector
        }

    """
   
    #Loads the image in BGR and Gray scale
    my_folder = os.getcwd() #gets main folder
    my_path=os.path.join(my_folder, image_name) #sets the path adding the image name
    my_Picture = cv.imread(my_path) #load image
    if my_Picture is None: #checks the image was loaded correctly
        print("Screenshot could not be loaded correctly. Please check path. \n")
        return
    myPicture_gray = cv.cvtColor(my_Picture, cv.COLOR_BGR2GRAY)

    # Perform EAST Text detection. Makes sure w and h are multiples of 32
    (h, w) = my_Picture.shape[:2]
    h = (h // 32) * 32
    w = (w // 32) * 32
    EAST_text_boxes = EAST(my_Picture, new_w=w, new_h=h)
    
    #Apply Threshold to remove pat of the background and isolate tables
    #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    low=250
    _, myPicture_thresh = cv.threshold(myPicture_gray, low, 255, cv.THRESH_BINARY)

    #Sobel helps separating vertical and horizontal line (better results than Canny Edges)
    # https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    SobelX = cv.Sobel(myPicture_thresh, cv.CV_64F, 1, 0, ksize=3)
    SobelY = cv.Sobel(myPicture_thresh, cv.CV_64F, 0, 1, ksize=3)
    SobelX_Gray_8u = cv.convertScaleAbs(SobelX)
    SobelY_Gray_8u = cv.convertScaleAbs(SobelY)

    #Applies a kernel to dilatate borders -> strengthen weak edges
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    SobelX_thick_edges = cv.dilate(SobelX_Gray_8u, kernel, iterations=2)
    SobelY_thick_edges = cv.dilate(SobelY_Gray_8u, kernel, iterations=2)

    #Applies a kernel for closing -> patch gaps differently for verticals and horizontal lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,5))
    SobelX_closed = cv.morphologyEx(SobelX_thick_edges, cv.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    SobelY_closed = cv.morphologyEx(SobelY_thick_edges, cv.MORPH_CLOSE, kernel, iterations=1)

    #Skeletonization -> reduce to 1-pixel centerlines
    SobelX_skeleton = skeletonize(SobelX_closed)
    SobelY_skeleton = skeletonize(SobelY_closed)

    #Perform line identification
    #Vertical Lines with Hough Probabilistic and filtered ober SobelX with thick edges from kernel
    #Hozitontal Lines with Hough Probabilistic and filtered ober SobelY with thick edges from kernel
    Vertical_LinesP = HoughProb_with_filter(Image=SobelX_skeleton, threshold=100, min_long_line=int(0.075*h))
    Horizontal_LinesP = HoughProb_with_filter(Image=SobelY_skeleton, threshold=100, min_long_line=int(0.075*h))
  
    #Filter possible line duplicates (Skeletonization should prevent this***)
    Vertical_LinesP = delete_deduplicate_lines(Vertical_LinesP, tol_dist=int(0.005*h))
    Horizontal_LinesP = delete_deduplicate_lines(Horizontal_LinesP, tol_dist=int(0.005*h))

    #Identify groups and boxes
    grouped_sequences_x,bounding_sequences_x = GetHorizontalSequencesAndBoxes(Horizontal_LinesP, rel_tolerance=0.075) 
    grouped_sequences_y,bounding_sequences_y = GetVerticalSequencesAndBoxes(Vertical_LinesP, rel_tolerance=0.075)

    #Extend boxes vertically to account for header if missed
    bounding_sequences_x=extend_boxes_vertically(bounding_sequences_x, factor_x=0.1, factor_y=0.1, image_height=h, image_width=w)
    bounding_sequences_y=extend_boxes_vertically(bounding_sequences_y, factor_x=0.1, factor_y=0.15, image_height=h, image_width=w)
    #draw_boxes(Image=my_Picture, boxes=bounding_sequences_x, title="Boxes Lines X", color=(0, 0, 255), thickness=2)
    #draw_boxes(Image=my_Picture, boxes=bounding_sequences_y, title="Boxes Lines Y", color=(0, 0, 255), thickness=2)

    #From all the text identified, single gruops of texts that may be columns or rows
    groups_col_text, boxes_columns_text, groups_col_text, boxes_rows_text = GetHorizontalVerticalSequencesAndBoxes_Text(EAST_text_boxes, tolerance_x=int(w*0.01),tolerance_y=int(w*0.01) )
    #draw_boxes(Image=my_Picture, boxes=boxes_columns_text, title="Text Columns", color=(0, 255, 0), thickness=2)
    #draw_boxes(Image=my_Picture, boxes=boxes_rows_text, title="Text Rows", color=(0, 255, 0), thickness=2)

    #Only care about text boxes that match with previous rows and column boxes
    boxes_1_filter = filter_bounding_boxes(bounding_sequences_x, boxes_columns_text)
    boxes_2_filter = filter_bounding_boxes(bounding_sequences_x, boxes_rows_text)
    boxes_3_filter = filter_bounding_boxes(bounding_sequences_y, boxes_columns_text)
    boxes_4_filter = filter_bounding_boxes(bounding_sequences_y, boxes_rows_text)
    possible_text_columns = boxes_1_filter+boxes_3_filter
    possible_text_rows = boxes_2_filter+boxes_4_filter
    #draw_boxes(Image=my_Picture, boxes=possible_text_columns, title="Filtered Text Columns", color=(0, 255, 0), thickness=2)
    #draw_boxes(Image=my_Picture, boxes=possible_text_rows, title="Filtered Text Rows", color=(0, 255, 0), thickness=2)

    # Identifies possible tables as the intersection of rows+columns+text
    intersections_rows_and_columns = merge_intersections(bounding_sequences_x,bounding_sequences_y)
    intersections_rows_and_text = merge_intersections(bounding_sequences_x, possible_text_columns)
    intersections_columns_and_text = merge_intersections(bounding_sequences_y, possible_text_columns)
    possible_table_boxes= filter_contained(intersections_rows_and_columns+intersections_rows_and_text+intersections_columns_and_text)
    #draw_boxes(Image=my_Picture, boxes=intersections_rows_and_columns, title="Intersections rows and columns", color=(255, 0, 0), thickness=2)
    #draw_boxes(Image=my_Picture, boxes=intersections_rows_and_text, title="Intersections rows and text", color=(255, 0, 0), thickness=2)
    #draw_boxes(Image=my_Picture, boxes=intersections_columns_and_text, title="Intersections columns and text", color=(255, 0, 0), thickness=2)
    #draw_boxes(Image=my_Picture, boxes=possible_table_boxes, title="Possible Tables", color=(255, 0, 0), thickness=2)

    #Identifies the headers
    possible_headers=get_possible_headers(possible_table_boxes, possible_text_rows)
    #draw_boxes(Image=my_Picture, boxes=possible_headers, title="Headers", color=(0, 255, 0), thickness=2)

    #Get column based on the header extedning down the table
    columns_from_header = extend_header_columns(possible_table_boxes, possible_headers, EAST_text_boxes)
    #draw_boxes(Image=my_Picture, boxes=columns_from_header, title="Header Columns", color=(0, 255, 0), thickness=2)

    #Get the row and column lines
    row_lines=get_lines_inside_table(possible_table_boxes, grouped_sequences_x, error_tol=3)
    #row_lines=get_lines_inside_table(possible_table_boxes, [Horizontal_LinesP], error_tol=3)
    #draw_lines(Image=my_Picture, lines=row_lines, title="Row separations", color=(0, 0, 255), thickness=2)
    column_lines=get_lines_inside_table(possible_table_boxes, grouped_sequences_y, error_tol=3)
    #column_lines=get_lines_inside_table(possible_table_boxes, [Vertical_LinesP], error_tol=3)
    #draw_lines(Image=my_Picture, lines=column_lines, title="Column separations", color=(0, 0, 255), thickness=2)

    #Get the possible columns and rows as boxes based on the row/column lines and text
    possible_row_boxes=lines_to_boxes(grouped_sequences_x, orientation='horizontal')
    possible_row_boxes=merge_boxes_with_priority(possible_table_boxes, possible_row_boxes, possible_text_rows, orientation='horizontal')
    #draw_boxes(Image=my_Picture, boxes=possible_row_boxes, title="Possible Rows", color=(0, 255, 255), thickness=1)
    possible_column_boxes=lines_to_boxes(grouped_sequences_y, orientation='vertical')
    possible_column_boxes=merge_boxes_with_priority(possible_table_boxes, possible_column_boxes, possible_text_columns, orientation='vertical')
    #possible_column_boxes=merge_boxes_with_priority(possible_table_boxes, possible_column_boxes, columns_from_header, orientation='vertical')
    #draw_boxes(Image=my_Picture, boxes=possible_column_boxes, title="Possible Columns", color=(0, 255, 255), thickness=1)

    DETECTED_TABLES=[]
    #Metrics/Features used for table comparisson
    for table in possible_table_boxes:
        
        table_width=abs(table[2]-table[0]) #used for normalization of some features
        table_height=abs(table[3]-table[1]) #used for normalization of some features

        #Importan part of the table
        this_table_header=[]

        this_table_row_boxes=[]
        this_table_columns_boxes=[]

        this_table_row_lines=[]
        this_table_column_lines=[]

        this_table_columns_from_header=[]
        this_table_text_columns=[]
        this_table_text_rows=[]


        #Row Average Height
        row_counter=0
        row_height=0
        average_row_height=0
        for row_box in possible_row_boxes:
            if boxes_intersect(table,row_box):
                row_counter += 1
                row_height += abs(row_box[3]-row_box[1])
                this_table_row_boxes.append(row_box)
        if(row_counter!=0):
            average_row_height=(row_height/row_counter)/table_width
        
        #Column Average Width
        column_counter=0
        column_width=0
        average_column_width=0
        for column_box in possible_column_boxes:
            if boxes_intersect(table, column_box):
                column_counter += 1
                column_width += abs(column_box[2]-column_box[0])
                this_table_columns_boxes.append(column_box)
        if(column_counter!=0):
            average_column_width=(column_width/column_counter)/table_width

        #Amount of columns detected
        header_text_counter=0
        for header_extended_box in columns_from_header:
            if boxes_intersect(table, header_extended_box):
                header_text_counter += 1       
                this_table_columns_from_header.append(header_extended_box)
        #number_columns=max(column_counter, header_text_counter) #tends to commit errors
        number_columns=column_counter

        #Amount of vertical lines detected
        this_table_column_lines = get_lines_inside_table([table], [column_lines])
        number_column_lines=len(this_table_column_lines)
        #Amount of horizontal lines detected
        this_table_row_lines = get_lines_inside_table([table], [row_lines])
        number_row_lines=len(this_table_row_lines)/table_height

        #header height relation to table width
        header_heigth=0
        for header in possible_headers:
            if boxes_intersect(table, header):
                header_heigth += abs(header[3]-header[1])
                this_table_header.append(header)
        header_heigth_to_table_width=header_heigth/table_width

        #Avarage text size
        horizontal_text_boxes_counter=0
        total_text_size=0
        average_text_size=0
        for text_box in possible_text_columns:
            if boxes_intersect(table, text_box):
                this_table_text_columns.append(text_box)
        for text_box in possible_text_rows:
            if boxes_intersect(table, text_box):
                this_table_text_rows.append(text_box)
                horizontal_text_boxes_counter += 1
                total_text_size += abs(text_box[3]-text_box[1])
        if(horizontal_text_boxes_counter!=0):
            average_text_size=(total_text_size/horizontal_text_boxes_counter)/table_width

        #Feature vector for comparisson
        numeric_features_vector=np.array([average_row_height, average_column_width, number_columns, number_column_lines, number_row_lines, header_heigth_to_table_width, average_text_size])

        #Full Dictionary
        this_table_features = {
            "Table Boundaries": table,
            "Header Boundaries": this_table_header,
            "Row Boxes": this_table_row_boxes,
            "Column Boxes": this_table_columns_boxes,
            "Row Lines": this_table_row_lines,
            "Column Lines": this_table_column_lines,
            "Columns from Header": this_table_columns_from_header,
            "Text Columns": this_table_text_columns,
            "Text Rows": this_table_text_rows,
            "Image": my_Picture,
            "Feature Vector": numeric_features_vector
        }

        DETECTED_TABLES.append(this_table_features) #List of dictionaries

    return DETECTED_TABLES




def IdentifySimilarTables(Table1, Tables2, similarity_threshold=0.1):
    """
    This function compared tables from two groups, and checks similarities. 
    Returns a table from group2 that matches the one from group1.
    Group1 is expected to have only one table. If multiple tables, it chooses the table with the most columns. 
    INPUTS:
        Table1: List of dictionaries. Each dictionary describes a table.
        Tables2: List of dictionaries. Each dictionary describes a table.
        similarity_threshold: Maximum euclidean distance to be considered matching tables (default 0.1)
    RETURN:
        MATCHING_TABLES: List of citionaries. Each dictionary describes a table from Tables2 that matches the table from Table1. 
    """

    #Checks tables were detected/images correctly loaded
    if Table1 is None:
        print("\n No tables detected in prototype. \n")
        return
    if Tables2 is None:
        print("\n No tables detected in the screenshot. \n")
        return

    #Checks if there is more than one table in Table1. If so, takes only the one with the most columns.
    if(len(Table1)>1):
        #print("Caution: more than 1 table detected in the prototype image. Taking the table with the most columns.")
        max_columns_table = max(Table1, key=lambda dict: dict["Feature Vector"][2])
        vector_proto = max_columns_table['Feature Vector']
    else:
        vector_proto = Table1[0]['Feature Vector']

    MATCHING_TABLES=[] #empty list

    #Compares the feature vector from each table from Tables2 to the vector from Table1 with normalized euclidean distance
    for table_dic in Tables2:
        vector_exp = table_dic['Feature Vector']
        #print((euclidean_distance(vector_proto, vector_exp)))
        if (euclidean_distance(vector_proto, vector_exp) < similarity_threshold): #similarity threshold
            MATCHING_TABLES.append(table_dic) #sabes the table
            
    print(f"\nIdentified {len(MATCHING_TABLES)} matching table(s).")
    return MATCHING_TABLES



if __name__ == "__main__":

    print("Computer Vision Challenge - Santiago Solano \n")

    Tables_from_proto=None
    while Tables_from_proto is None:
        proto_path = input("Enter the relative path to the PROTOTYPE/PARTIAL IMAGE inside the working directory: ")
        Tables_from_proto = IdentifyTables(proto_path) 

    Tables_from_ss=None
    while Tables_from_ss is None:
        ss_path = input("Enter the relative path to the SCREENSHOT IMAGE inside the working directory: ")
        Tables_from_ss = IdentifyTables(ss_path) 

    Matching_tables = IdentifySimilarTables(Tables_from_proto, Tables_from_ss)

    if Matching_tables:
        for t in Matching_tables:
            print(f"Table Boundaries [xmin, ymin, xmax, ymax] : {t['Table Boundaries']}")
            print(f"Header Boundaries [xmin, ymin, xmax, ymax] : {t['Header Boundaries']}")
        DrawTables(Matching_tables)

    input("\nPress ENTER to exit.")

    
