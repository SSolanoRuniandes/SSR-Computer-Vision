import numpy as np
from collections import defaultdict


"""Functions to perform line/box manipulation"""

def LineLength(line):
    """
    Function used to calculate the length of a line
    INPUTS:
        line: A single line with the format [x1, y1, x2, y2]
    OUTPUTS:
        returns the line length in pixels
    """
    return abs( line[2]-line[0] + line[3]-line[1] )

def LineOrientation(line, tolerance=5):
    """
    Functions used to identify the orientation of a line. Vertical, horizontal or neither
    INPUTS:
        line: A single line with the format [x1, y1, x2, y2]
        tolerance: A small tolerance of pixel errors to consider a line vertical or horizontal despite this deviation (default 5 pixels)
    OUTPUTS:
        returns: Integer, 1 (Vertical), 2 (Horizontal), 0 (Nither)
    """
    #Gets the change in x and y -axis
    dx=abs(line[2]-line[0])
    dy=abs(line[3]-line[1])

    if(dx<tolerance): #Vertical Line
        return 1
    elif(dy<tolerance): #Horizontal Line
        return 2
    else: #Neither
        return 0

def get_bounding_box(sequence):
    """
    Function that receives a sequence of lines and returns the bounding box of the sequence
    INPUTS:
        sequence: A list of lists. The main list is the sequence that contains lines that are similar (same orientation, length, starting point). 
        A single line has the format [x1, y1, x2, y2]
    OUTPUTS:
        Returns the bounding box for the sequence in the format [xmin, ymin, xmax, ymax]
    """

    xs, ys = [], [] #empty lists for coordinates
    for line in sequence: #will check all the lines in the sequence
        xs.extend([line[0], line[2]]) #adds multiple x coordinates
        ys.extend([line[1], line[3]]) #adds multiple y coordinates
    return [min(xs), min(ys), max(xs), max(ys)]  # boundig box coordinates [x_min, y_min, x_max, y_max]


def get_bounding_boxes_rows_and_columns_text(groups, tolerance_x=10, tolerance_y=10):
    """
    Function that receives grups of boxes and returns the bounding box for each group
    INPUTS:
        groups: A list of lists. The main list is the group that contains boxes that are alligened vertically or horizontally 
        A single box has the format [xmin, ymin, xmax, ymax]
        tolerance_x: Adds a tolerance of pixels horizontally to the bounding box (default 10 pixels)
        tolerance_y: Adds a tolerance of pixels vertically to the bounding box (default 10 pixels)
    OUTPUTS:
        bounding_boxes: List of lists. Returns the bounding box for each group in the format [xmin, ymin, xmax, ymax]
    """

    bounding_boxes = [] #empty list for saving the bounding boxes
    for group in groups: #will check every group
        arr = np.array(group) #creates an array with the lists from the group
        #get the maximum and minimum coordinates and adds tolerance
        xmin = np.min(arr[:,0])-tolerance_x
        ymin = np.min(arr[:,1])-tolerance_y
        xmax = np.max(arr[:,2])+tolerance_x
        ymax = np.max(arr[:,3])+tolerance_y
        bounding_boxes.append([xmin, ymin, xmax, ymax]) #saves bounding box
    return bounding_boxes



def delete_deduplicate_lines(lines, tol_dist=5, tol_angle=np.deg2rad(5)):
    """
    This function receives a list of lines, and will delete lines that are too similar or seem overlaping
    INPUTS:
        lines: List of lists. Each line has the format [x1, y1, x2, y2] and may contain duplicates
        tol_dist: Maximum distance in pixels that may be between lines to consider them equal
        tol_angle: Maximum angle in radians that may be formed between lines to consider them equal
    OUTPUTS:
        unique: List of lists. Each line has the format [x1, y1, x2, y2] and will not contain duplicates
    """
   
    unique = [] #list for filtered lines

    for l in lines:
        #for each line, distances and angles are calculated
        x1, y1, x2, y2 = l
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)
        #will check with the lines already in unique
        keep = True
        for u in unique:
            #for each line distances and angles are calculated
            ux1, uy1, ux2, uy2 = u
            udx = ux2 - ux1
            udy = uy2 - uy1
            uangle = np.arctan2(udy, udx)
            #if angle is similar enough, they are paralell
            if abs(angle - uangle) < tol_angle:
                #if the distance is very small, they are probably the same
                dist = abs((x1 + x2)/2 - (ux1 + ux2)/2) + abs((y1 + y2)/2 - (uy1 + uy2)/2)
                if dist < tol_dist:
                    keep = False #don't keep this line because it is to similar to another
                    break
        if keep:
            unique.append(l) #keep
    return unique


def filter_bounding_boxes(group1, group2, percentage_threshold=0.3):
    """
    This function receives two groups of bounding boxes. It will only keep the bounding boxes from group2 inside group1. 
    INPUTS:
       group1: List of lists. Each box has the format [xmin, ymin, xmax, ymax]
       group2: List of lists. Each box has the format [xmin, ymin, xmax, ymax] 
       percentage_threshold: A minimum porcentage of area from box2 that may be inside box1 to be considered (trimmed), otherwise fully deleted (default 30%)
    OUTPUTS:
        filtered: List of lists. Each box has the format [xmin, ymin, xmax, ymax] and only contains boxes grom group2 insdie the boundaries of group1
    """

    #Precheck: if either group is empty, return empty list
    if not group1 or not group2:
        return []

    filtered = [] #empty list
    #Cycles trhough all the boxes from each group
    for xmin1, ymin1, xmax1, ymax1 in group1:
        for xmin, ymin, xmax, ymax in group2:
            #It will cut parts of the boxes from group 2 outside group 1
            new_xmin = max(xmin, xmin1)
            new_ymin = max(ymin, ymin1)
            new_xmax = min(xmax, xmax1)
            new_ymax = min(ymax, ymax1)
            #Checks if it is a valid box
            if new_xmin < new_xmax and new_ymin < new_ymax:
                #Checks how much of box2 is actually inside box1
                area_box2 = (xmax - xmin) * (ymax - ymin)
                area_intersection = (new_xmax - new_xmin) * (new_ymax - new_ymin)
                if area_intersection >= percentage_threshold * area_box2:
                    #If a minimum area intersects, trims the box and adds it
                    filtered.append([new_xmin, new_ymin, new_xmax, new_ymax])
                
    return filtered


def extend_boxes_vertically(boxes, factor_x=0.1, factor_y=0.1, image_height=None, image_width=None):
    """
    Function used to expand boxes horizontally and vertically. 
    INPUTS:
        boxes: List of lists. Each box has the format [xmin, ymin, xmax, ymax]
        factor_x: Percentage of expansion in the x-direction on each side (default 10%)
        factor_y: Percentage of expansion in the x-direction on each side (default 15%)
        image_height: Maximum image height used for trimming. (default None)
        image_width: Maximum image width used for trimming. (default None)
    OUTPUTS:
        extended: List of lists. Each extended box has the format [xmin, ymin, xmax, ymax]
    """
    
    extended = [] #empty list
    for xmin, ymin, xmax, ymax in boxes:
        #Gets the original dimensions and calculated the delta for expansion
        h = ymax - ymin
        delta_y = int(h * factor_y)
        w = xmax - xmin
        delta_x = int(w * factor_x)
        #Expand dimensiones
        new_ymin = ymin - delta_y
        new_ymax = ymax + delta_y
        new_xmin = xmin - delta_x
        new_xmax = xmax + delta_x

        #Clip to fit in the original image if specified
        if image_height is not None:
            new_ymin = max(0, new_ymin)
            new_ymax = min(image_height - 1, new_ymax)
        if image_width is not None:
            new_xmin = max(0, new_xmin)
            new_xmax = min(image_width - 1, new_xmax)

        #Adds new expanded box
        extended.append([new_xmin, new_ymin, new_xmax, new_ymax]) 

    return extended


def boxes_intersect(box1, box2, small_tolerance=3):
        """
        This function receives two bounding boxes, and returnsa boolean wether they intersect or no. 
        INPUTS:
            box1: Bounding box with the format [xmin, ymin, xmax, ymax]
            box2: Bounding box with the format [xmin, ymin, xmax, ymax]
            small_tolerance: If the boxes are close enough with this tolerance, they also intersect (default 3 pixels)
        OUTPUTS:
            Boolean True or False. Intersect or not. 
        """

        #Get coordinates
        b1x1, b1y1, b1x2, b1y2 = box1
        b2x1, b2y1, b2x2, b2y2 = box2
        
        #Check order [xmin, ymin, xmax, ymax]
        if(b1x2<b1x1):
            b1x1, b1x2 = b1x2, b1x1
        if(b1y2<b1y1):
            b1y1, b1y2 = b1y2, b1y1
        if(b2x2<b2x1):
            b2x1, b2x2 = b2x2, b2x1
        if(b2y2<b2y1):
            b2y1, b2y2 = b2y2, b2y1

        #check for intersections between two boxes
        #If one is completely on the left
        if b1x2 < b2x1-small_tolerance or b2x2 < b1x1-small_tolerance:
            return False
        #If one is completely up
        if b1y2 < b2y1-small_tolerance or b2y2 < b1y1-small_tolerance:
            return False
        #else, they intersect
        return True


def merge_intersections(group1, group2):
    """
    Functions that returns a bigger bounding box that contains two bounding boxes if they intersect. 
    INPUTS:
        group1: List of lists. Each box has the format [xmin, ymin, xmax, ymax]
        group2: List of lists. Each box has the format [xmin, ymin, xmax, ymax]
    OUTPUTS:
        merged: List of lists. Each box has the format [xmin, ymin, xmax, ymax]. These boxes are bigger than the original and group the intersections. 
    """

    merged = [] #empty list

    def union_box(box1, box2):
        #Returns the bigger box that fits box1 and box2
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2]

    used2 = set() #for not repeating boxes

    #Cycle thorugh all the boxes
    for box1 in group1:
        merged_box = box1.copy()
        for i, box2 in enumerate(group2):
            if i in used2:
                continue
            if boxes_intersect(merged_box, box2): #checks intersection
                merged_box = union_box(merged_box, box2) #adds a new box containing both boxes because there was an intersection
                used2.add(i) #not take into account this box in the future
        merged.append(merged_box) 

    return merged


def filter_contained(boxes):
    """
    This function deletes boxes contained in others
    INPUTS:
        boxes: List of lists. Each box has the format [xmin, ymin, xmax, ymax]. Some boxes may be contained in others. 
    OUTPUTS:
        filtered: List of lists. Each box has the format [xmin, ymin, xmax, ymax]. The boxes will not be contained in others. 
    """

    filtered = []#empty list
    #Cycle through all the boxes
    for i, box1 in enumerate(boxes):
        contained = False
        for j, box2 in enumerate(boxes):
            if i == j:
                continue
            #Checks if box 1 is inside box 2
            if box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]:
                #If some boxes are identical, only deletes the one with higher index
                if box1 != box2 or i > j:
                    contained = True
                    break
        if not contained:
            filtered.append(box1) #Only adds if not contained
    return filtered



def get_possible_headers(group1, group2, width_ratio=0.8):
    """
    This Function searches fro boxes from group 2 inside group 1, and then chooses the one on top.
    If group1 is a list of possible tables, and group2 are text rows, the result is effectively a possible header. 
    INPUTS:
        group1: List of lists. Each box is defined as [xmin, yminx, xmax, ymax]
        group2: List of lists. Each box is defined as [xmin, yminx, xmax, ymax]
        width_ratio: A minimum width raton between a box from group2 to a box from group1 to be considered a possible header (default 80%)
    OUTPUTS:
        headers: List of lists. Each header is a box is defined as [xmin, yminx, xmax, ymax]

    """
    
    headers = []

    for box1 in group1: #for each possible table
        candidates = []
        w1=abs(box1[2] - box1[0]) #width of the table
        for box2 in group2: #check text boxes (rows)
            #firs verifies that it is inside the table
            if (box2[0] >= box1[0] and box2[1] >= box1[1] and box2[2] <= box1[2] and box2[3] <= box1[3]):
                #extra width condition for it to be a header
                w2=abs(box2[2] - box2[0])
                if w2 >= width_ratio * w1:
                    candidates.append(box2) #becomes a candidate
        if candidates:
            #chooses the one on top 
            top_box = min(candidates, key=lambda b: b[1])
            headers.append(top_box)

    return headers

def extend_header_columns(tables, headers, text_boxes):
    """
    Function that expand columns from the header of a table. 
    INPUTS:
        tables: List of lists. Each table is defined by a bounding box with the format [xmin, ymin, xmax, ymax]
        headers: List of lists. Each header is defined by a bounding box with the format [xmin, ymin, xmax, ymax]
        text_boxes: List of lists. Each text box has the format [xmin, ymin, xmax, ymax]
    OUTPUT:
        header_column_expanded: List of lists. Each columns is defined by a bounding box with the format [xmin, ymin, xmax, ymax]
    """

    header_column_expanded = [] #empty list
    for i, header in enumerate(headers): #Cycle all headers. The index i from the header should math the table index
        for box in text_boxes: #Cycle trhough all the text boxes
            if boxes_intersect(box, header): #If the text belongs to the header
                x1,y1,x2,y2 = box #Get the coordinates of the text box
                header_column_expanded.append([x1, tables[i][1], x2, tables[i][3]]) #expand the boxes vertically
    return header_column_expanded


def get_lines_inside_table(boxes, groups_of_lines, error_tol=5):
    """
    This functions will identify which lines (vertical and horizontal) are inside a group of boxes. 
    INPUTS:
        boxes: A list of lists. Each box is defined as [xmin, yminx, xmax, ymax]
        groups_of_lines: Groups with sequences of lines. Each line has the format [x1, y1, x2, y2]. The groups are spected to ve horizontal or vertical lines. 
        error_tol: Small difference that may exist between line limits to be considered vertical or horizontal, and possible difference between box and line limits (default 5 pixels)
    OUTPUTS:
        clipped: List of lists. Lines that fit inside the boxes, with the format [x1, y1, x2, y2]

    """

    clipped = [] #empty list

    for lines in groups_of_lines:
        #group of lines have seqences of consecutie lines with the same length and orientation
        for line in lines:
            #each line has the strucutre [x1,y1,x2,y2]
            x1, y1, x2, y2 = line
            #check order of coordinates
            if(y1>y2):
                y1, y2 = y2, y1
            if(x1>x2):
                x1, x2 = x2, x1
            #Cycle through boxes
            for bx1, by1, bx2, by2 in boxes:
                # vertical line
                if abs(x1-x2)<error_tol:
                    if bx1-error_tol <= x1 <= bx2+error_tol:
                        y_start = max(y1, by1)
                        y_end   = min(y2, by2)
                        if y_start < y_end:
                            clipped.append([x1, y_start, x2, y_end]) #vertical line that fits into a box
                # horizontal line
                elif abs(y1-y2)<error_tol:
                    if by1-error_tol <= y1 <= by2+error_tol:
                        x_start = max(x1, bx1)
                        x_end   = min(x2, bx2)
                        if x_start < x_end:
                            clipped.append([x_start, y1, x_end, y2]) #horizontal line that fits into a box
    return clipped



def lines_to_boxes(groups_of_lines, orientation="horizontal", error_tol=2):
    """
    This function receives sequences of horizontal or vertical lines and will return the boxes formed between them
    INPUTS:
        groups_of_lines: Groups of sequences of similar lines. Each line has the format [x1, y1, x2, y2]
        orientation: String. 'horizontal' or 'vertical'. Set if it will look for boxes between horizontal or vertical lines, rows or columns. (default 'horizontal')
        error_tol=Minimum distance between lines to form a box (default 2 pixels)
    OUTPUTS:
        boxes: List of lists. Each box has the format [xmin, ymin, xmax, ymax]
    """

    boxes = [] #empty list

    #groups of lines have sequences, that contain the lines
    for lines in groups_of_lines: 
        norm_lines = [] #will fix the lines 
        for line in lines:
            x1, y1, x2, y2 = line
            if orientation == "horizontal":
                if x1 > x2:
                    x1, x2 = x2, x1 # organize coordinates if inverted
                y = (y1 + y2) // 2  # in case not perfectly aligned
                norm_lines.append((x1, y, x2, y))
            elif orientation == "vertical":
                if y1 > y2:
                    y1, y2 = y2, y1 # organize coordinates if inverted
                x = (x1 + x2) // 2  # in case not perfectly aligned
                norm_lines.append([x, y1, x, y2])

        # sort lines by position
        if orientation == "horizontal":
            norm_lines.sort(key=lambda l: l[1])  # sort by y
        else:  # vertical
            norm_lines.sort(key=lambda l: l[0])  # sort by x

        # generate boxes between consecutive lines
        for i in range(len(norm_lines) - 1):
            if orientation == "horizontal":
                x1a, y1a, x2a, _ = norm_lines[i]     #boundaries first line
                x1b, y1b, x2b, _ = norm_lines[i + 1] #boundaries second line
                left = max(x1a, x1b)  #will respect the boundaries of the shorter line
                right = min(x2a, x2b)
                if right - left > error_tol:
                    top = y1a
                    bottom = y1b
                    boxes.append([left, top, right, bottom]) #adds the box
            else:  # vertical
                x1a, y1a, _, y2a = norm_lines[i]     #boundaries first line
                x1b, y1b, _, y2b = norm_lines[i + 1] #boundaries second line
                top = max(y1a, y1b)   #will respect the boundaries of the shorter line
                bottom = min(y2a, y2b)
                if bottom - top > error_tol:
                    left = x1a
                    right = x1b
                    boxes.append([left, top, right, bottom]) #adds the box

    return boxes





def merge_boxes_with_priority(tables, group1, group2, orientation="horizontal"):
    """
    This function will return boxes from group1 and those from group2 that do not share space with group 1
    INPUTS:
        tables:
        group1: List of lists. Each box is defined as [xmin, yminx, xmax, ymax]
        group2: List of lists. Each box is defined as [xmin, yminx, xmax, ymax]
        orientation: String. 'horizontal' or 'vertical'. Set if it will look for boxes between horizontal or vertical lines, rwos or columns. (default 'horizontal')
    OUTPUTS:
        results: List of lists. Each box is defined as [xmin, yminx, xmax, ymax]
    """

    result = [] #empty list

    #First, it will add the boxes from group 1
    for table in tables:
        h_table=abs(table[3]-table[1])
        w_table=abs(table[2]-table[0])
        for box1 in group1:
            if(orientation == "horizontal"):
                if(abs(box1[2]-box1[0]) > 0.8*w_table):
                    result.append(box1)
            else: #vertical
                if(abs(box1[3]-box1[1]) > 0.8*h_table):
                    result.append(box1)

    #Then checks boxes from group2, but prioritize previous boxes
    for table in tables:
        h_table=abs(table[3]-table[1])
        w_table=abs(table[2]-table[0])
        for box2 in group2:
            if not any(boxes_intersect(box2, box1) for box1 in result): #check intersection from all boxes from group1 and other boxes from group 2 already in resuls
                if(orientation == "horizontal"):
                    if(abs(box2[2]-box2[0]) > 0.8*w_table):
                        result.append(box2)
                else: #vertical
                    if(abs(box2[3]-box2[1]) > 0.8*h_table):
                        result.append(box2)

    return result



       

def GetHorizontalSequencesAndBoxes(AllLines, rel_tolerance=0.05):
    """
    Function that receives a list of lines, and will group similar HOZIRONTAL lines with the same length and starting point (possible table rows)
    INPUTS:
        AllLines: A list of lists. The lines have the format [x1, y1, x2, y2]
        rel_tolerance: A percentage of error admissible in line length to be considered equal (default: 5%)
    OUTPUTS:
        Returns a tuple (grouped_sequences_x,bounding_sequences_x) with the sequences of similar lines (list of lists) and its bounding boxes (list of lists) 
    """

    #If AllLines comes from SobelY, the lines whould already be horizontal.
    #If AllLines comes from CannyEdges, the lines can ve vertical or horizontal. 

    #Empty lists to store the lines and characteristics
    horizontal_lines, horizontal_lines_length, horizontal_lines_xo, horizontal_lines_y = [], [], [], []

    #Checks that the lines are horizontal
    for line in AllLines:
        l_or_h=LineOrientation(line) #Obtains orientation
        if(l_or_h==2): #2 means horizontal
            #Append the line, its length, its starting point xo, and its y-coordinate 
            horizontal_lines.append(line) 
            horizontal_lines_length.append(LineLength(line))
            horizontal_lines_xo.append(line[0])
            horizontal_lines_y.append(line[1])

    # Now, groups of horizontal lines are created
    # Criteria:
    # 1. The lines must start at the same xo
    # 2. The lines must have the same length
    # 3. There should not be a third line between two lines, unless the third line follows the previous two conditions

    
    """This code segment will group lines by their same initial x and length"""
    groups_x = defaultdict(list) #Dictionary for storage
    for i, line in enumerate(horizontal_lines):
        #Get the starting point and length of the line
        xo = horizontal_lines_xo[i]
        length = horizontal_lines_length[i]
        #Will look in the dictionary for a key for lines with the same starting point and length
        found_key = None
        for (kxo, klength) in groups_x.keys(): #checks keys in the groups
            if kxo == xo and abs(length - klength) <= rel_tolerance * klength: #gets the length, and checks if the new line satisfies same length with a relative error
                found_key = (kxo, klength) #found the key then breaks
                break
        if found_key:
            groups_x[found_key].append((horizontal_lines_y[i], line)) #will add the line to the same key, with its y-coordinate
        else:
            groups_x[(xo, length)].append((horizontal_lines_y[i], line)) #if not, creates a new group using this new line's length and starting point

    """This code segment will filter the groups by uninterrupted sequences checking the other coordinate"""
    grouped_sequences_x = [] #List for storage
    for key, lines in groups_x.items():
        lines.sort(key=lambda x:x[0]) #will sort by the y position coordinate
        seq = [lines[0][1]] # Start a new sequence with the first line in this group. lines are tuples like (y_coordinate, line)
        for idx in range(1, len(lines)): # Loop over the rest of the lines in this group, starting from the 2nd one
            # Checks if the line continues the sequence. Compares y-coordinate of the current line with previous line y-coordinate
            if lines[idx][0] > lines[idx-1][0]: #if it is greater
                seq.append(lines[idx][1]) #the line comes after, adds it to the sequence
            else: #if not, the sequence is broken
                grouped_sequences_x.append(seq) #add to the results the calculated sequence
                seq = [lines[idx][1]] #starts new sequence
        grouped_sequences_x.append(seq) #add last sequence

    """Filters and get bounding boxes of the line sequences"""
    grouped_sequences_x = [seq for seq in grouped_sequences_x if len(seq) >= 3] #Only cares about sequences of more than three lines
    bounding_sequences_x = [get_bounding_box(seq) for seq in grouped_sequences_x] #Also calculates bounding boxes for the indentified sequences


    return (grouped_sequences_x,bounding_sequences_x)


def GetVerticalSequencesAndBoxes(AllLines, rel_tolerance=0.05):
    """
    Function that receives a list of lines, and will group similar VERTICAL lines with the same length and starting point (possible table rows)
    INPUTS:
        AllLines: A list of lists. The lines have the format [x1, y1, x2, y2]
        rel_tolerance: A percentage of error admissible in line length to be considered equal (default: 5%)
    OUTPUTS:
        Returns a tuple (grouped_sequences_y,bounding_sequences_y) with the sequences of similar lines (list of lists) and its bounding boxes (list of lists) 
    """
    
    #If AllLines comes from SobelX, the lines whould already be horizontal.
    #If AllLines comes from CannyEdges, the lines can ve vertical or horizontal. 

    vertical_lines, vertical_lines_length, vertical_lines_yo, vertical_lines_x= [], [], [], []

    #Checks that the lines are vertical
    for line in AllLines:
        l_or_h=LineOrientation(line) #Obtains orientation
        if(l_or_h==1): #1 means vertical
            #Append the line, its length, its starting point yo, and its x-coordinate 
            vertical_lines.append(line)
            vertical_lines_length.append(LineLength(line))
            vertical_lines_yo.append(line[1])
            vertical_lines_x.append(line[0])

    # Now, groups of vertical lines are created
    # Criteria:
    # 1. The lines must start at the same yo
    # 2. The lines must have the same length
    # 3. There should not be a third line between two lines, unless the third line follows the previous two conditions

    
    """This code segment will group lines by their same initial y and length"""
    groups_y = defaultdict(list) #Dictionary for storage
    for i, line in enumerate(vertical_lines):
        #Get the starting point and length of the line
        yo = vertical_lines_yo[i]
        length = vertical_lines_length[i]
        #Will look in the dictionary for a key for lines with the same starting point and length
        found_key = None
        for (kyo, klength) in groups_y.keys(): #checks keys in the groups
            if kyo == yo and abs(length - klength) <= rel_tolerance * klength: #gets the length, and checks if the new line satisfies same length with a relative error
                found_key = (kyo, klength) #found the key then
                break
        if found_key:
            groups_y[found_key].append((vertical_lines_x[i], line)) #will add the line, with a slightly different length, to the same key
        else:
            groups_y[(yo, length)].append((vertical_lines_x[i], line)) #if not, creates a new group using this new line's length

    """This code segment will filter the groups by uninterrupted sequences checking the other coordinate"""
    grouped_sequences_y = [] #List for storage
    for key, lines in groups_y.items():
        lines.sort(key=lambda y:y[0]) #will sort by the y position coordinate
        seq = [lines[0][1]] # Start a new sequence with the first line in this group. lines are tuples like (x_coordinate, line)
        for idy in range(1, len(lines)): # Loop over the rest of the lines in this group, starting from the 2nd one
            # Checks if the line continues the sequence. Compares x-coordinate of the current line with previous line x-coordinate
            if lines[idy][0] > lines[idy-1][0]: #if it is greater
                seq.append(lines[idy][1]) #the line comes after, adds it to the sequence
            else: #if not, the sequence is broken
                grouped_sequences_y.append(seq) #add to the results the calculated sequence
                seq = [lines[idy][1]] #starts new sequence
        grouped_sequences_y.append(seq) #add last sequence

    """Filters and get bounding boxes of the line sequences"""
    grouped_sequences_y = [seq for seq in grouped_sequences_y if len(seq) >= 3] #Only cares about sequences of more than three lines
    bounding_sequences_y = [get_bounding_box(seq) for seq in grouped_sequences_y] #Also calculates bounding boxes for the indentified sequences

    return (grouped_sequences_y,bounding_sequences_y)




def GetHorizontalVerticalSequencesAndBoxes_Text(boxes, tolerance_x=10, tolerance_y=10):
    """
    Function that receives a list of boxes, and will group the boxes in columns and rows
    INPUTS:
        boxes: A list of lists. The boxes have the format [xmin, ymin, xmax, ymax]
        tolerance_x: A tolerance in pixels that a box can be deviated horizontally from another but still be considered in the same row or column (default: 10 pixels)
        tolerance_y: A tolerance in pixels that a box can be deviated vertically from another but still be considered in the same row or column (default: 10 pixels)
    OUTPUTS:
        Returns a tuple (groups_col_text, boxes_columns_text, groups_col_text, boxes_rows_text) with the sequeces of text rows/columns and their main bounding box
    """

    #Empty lists for storage
    groups_col_text, groups_row_text = [], []
    used_col = set()
    used_row = set()
    
    """This section looks for boxes alligned that may be columns"""
    for i, box in enumerate(boxes):
        if i in used_col: #if the box is already used in a column ignore
            continue
        #gets the boundaries and adds the box to a new column
        xmin, ymin, xmax, ymax = box
        group_col = [box]
        used_col.add(i)  
        #Search for other boxed alligned
        for j, other in enumerate(boxes):
            if j in used_col: #if the box is already used in a column ignore
                continue
            #gets the boundaries of the second box
            oxmin, oymin, oxmax, oymax = other
            if abs(xmin - oxmin) <= tolerance_x: # Check alignment on left side
                #If alligned, add the box to the column
                group_col.append(other)
                used_col.add(j)
        #Finish the column adding it to the groups
        groups_col_text.append(group_col)

    """Filters and get bounding boxes of the columns"""
    groups_col_text = [group_col for group_col in groups_col_text if len(group_col) >= 2] #Only keeps groups with at least 2 boxes
    boxes_columns_text = get_bounding_boxes_rows_and_columns_text(groups_col_text) #Also calculates bounding box for the columns


    """This section looks for boxes alligned that may be rows"""
    for i, box in enumerate(boxes): 
        if i in used_row: #if the box is already used in a row ignore
            continue
        #gets the boundaries and adds the box to a new row
        xmin, ymin, xmax, ymax = box
        group_row = [box]
        used_row.add(i)
        #Search for other boxed alligned
        for j, other in enumerate(boxes):
            if j in used_row: #if the box is already used in a row ignore
                continue
            #gets the boundaries of the second box
            oxmin, oymin, oxmax, oymax = other
            if abs(ymin - oymin) <= tolerance_y: # Check alignment on top side
                #If alligned, add the box to the row
                group_row.append(other)
                used_row.add(j)
        #Finish the row adding it to the groups
        groups_row_text.append(group_row)

    """Filters and get bounding boxes of the rows"""
    groups_row_text = [group_row for group_row in groups_row_text if len(group_row) >= 2] #Only keeps groups with at least 2 boxes
    boxes_rows_text = get_bounding_boxes_rows_and_columns_text(groups_row_text) #Also calculates bounding box for the rows

    return (groups_col_text, boxes_columns_text, groups_col_text, boxes_rows_text)


