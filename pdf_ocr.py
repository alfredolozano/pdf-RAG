### ULTIMA VERSION WIP FEB27

from paddleocr import PaddleOCR
import pdfplumber
from PIL import Image
import io

import torch
from ultralytics import YOLO

# Path to the PDF file
file_path = '/content/DOC-20230901-WA0017_230916_085403.pdf'

# Initialize the PaddleOCR model for Spanish language
ocr_model = PaddleOCR(lang="es", use_gpu=True)  # 'es' for Spanish

# load model
device: str = "mps" if torch.backends.mps.is_available() else "cpu"

table_model = YOLO('/content/drive/MyDrive/AFIRME - RAGente de Poliza/v4_mult/train/weights/last.pt')
table_model.to(device)

# # set model parameters
table_model.overrides['conf'] = 0.25  # NMS confidence threshold
table_model.overrides['iou'] = 0.30  # NMS IoU threshold
table_model.overrides['agnostic_nms'] = True  # NMS class-agnostic
table_model.overrides['max_det'] = 15 # maximum number of detections per image


def normalize_accents(text):
    # Replace alternative representations of accented characters
    replacements = {
        'ä': 'á',
        '\\u00e1': 'á',  # Unicode for á
        'é': 'é',
        '\\u00e9': 'é',  # Unicode for é
        'ó': 'ó',
        '\\u00f3': 'ó',  # Unicode for ó
        '6l': 'ó',
        'ü': 'ú',
        '\\u00fc': 'ú',  # Unicode for ü
        'ú': 'ú',
    }

    # Iterate through the replacements and apply them
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

def drop_divisions(text):
    return text.replace(": | ", ": ").replace(":  |", '')

# Function to perform OCR on a single page of the PDF using PaddleOCR
def ocr_page(page):
    image = page.to_image(resolution=300).original
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    image_data = image_bytes.read()
    result = ocr_model.ocr(image_data)
    return result

def ocr_image(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    image_data = image_bytes.read()
    result = ocr_model.ocr(image_data)
    return result


def ocr_to_markdown_advanced(ocr_section_input, y_threshold=30):
    def get_vertical_center(box):
        return int((box[0][1] + box[2][1]) / 2)

    def get_horizontal_center(box):
        return int((box[0][0] + box[1][0]) / 2)

    # Normalize data
    text_entries = [(get_vertical_center(box), get_horizontal_center(box), text) for box, (text, _) in ocr_section_input]

    # Group by rows based on y_threshold
    rows = {}
    for y_center, x_center, text in text_entries:
        found_row = False
        for row_key in sorted(rows.keys()):
            if abs(row_key - y_center) <= y_threshold:
                rows[row_key]['entries'].append((x_center, text))
                found_row = True
                break
        if not found_row:
            rows[y_center] = {'y_center': y_center, 'entries': [(x_center, text)]}

    # Sort rows and process for magnetic merging
    sorted_row_keys = sorted(rows.keys())

    for i, row_key in enumerate(sorted_row_keys):
        current_row = rows[row_key]
        # If the current row has only one entry, try to merge it with an adjacent row
        if len(current_row['entries']) == 1:
            x_center, single_text = current_row['entries'][0]

            if 'DATOS' in single_text:
                continue
            # Look for a row to merge with above or below
            for direction in (-1, 1):  # Check the previous row and the next row
                adjacent_row_key = sorted_row_keys[i + direction] if 0 <= i + direction < len(sorted_row_keys) else None
                if adjacent_row_key:
                    adjacent_row = rows[adjacent_row_key]

                    # If the adjacent row has more than one entry, merge the single entry with the closest cell
                    if len(adjacent_row['entries']) > 1:
                        # Find the closest cell in the adjacent row to merge with based on x_center
                        closest_cell = min(adjacent_row['entries'], key=lambda entry: abs(entry[0] - x_center))

                        # Merge the single entry with the closest cell
                        if direction == -1:  # If the single entry is above, prepend its text
                            merged_text =  closest_cell[1] + ' ' + single_text

                        else:  # If the single entry is below, append its text
                            merged_text = closest_cell[1] + ' ' + single_text
                        # Update the cell in the adjacent row
                        adjacent_row['entries'] = [(x_center, merged_text) if cell == closest_cell else cell for cell in adjacent_row['entries']]
                        # Remove the single entry row as it has been merged
                        del rows[row_key]
                        sorted_row_keys.remove(row_key)
                        break

    # Sort each row by x_center and create markdown
    markdown_text = ""
    for row_key in sorted(rows.keys()):
        sorted_entries = sorted(rows[row_key]['entries'], key=lambda entry: entry[0])
        row_texts = [text for _, text in sorted_entries]
        markdown_text += "| " + " | ".join(row_texts) + " |\n"
    
    return markdown_text

def process_reading_order(data):
    # Calculate the midpoint between min and max x-coordinates
    x_coordinates = [elem[1] for elem in data]
    x_midpoint = (min(x_coordinates) + max(x_coordinates)) / 2

    flattened_data = data

    lines = []
    current_line_left = []
    current_line_right = []
    current_y = None

    for y, x, text in flattened_data:
        if current_y is not None and y != current_y:
            # Combine left and right columns if both are present
            if current_line_left and current_line_right:
                combined_line = " ".join(current_line_left) + " | " + " ".join(current_line_right)
                lines.append(combined_line)
            elif current_line_left:
                lines.append(" ".join(current_line_left))
            elif current_line_right:
                lines.append(" ".join(current_line_right))

            current_line_left, current_line_right = [], []

        if x < x_midpoint:
            current_line_left.append(text)
        else:
            current_line_right.append(text)

        current_y = y

    # Add the last lines if they exist
    if current_line_left or current_line_right:
        if current_line_left and current_line_right:
            combined_line = " ".join(current_line_left) + " | " + " ".join(current_line_right)
            lines.append(combined_line)
        elif current_line_left:
            lines.append(" ".join(current_line_left))
        elif current_line_right:
            lines.append(" ".join(current_line_right))

    return " ".join(lines)

def detect_tables_and_content(image, table_model):
    results = table_model.predict(image)
    table_boxes = []
    content_boxes = []

    for result in results:
        for box in result.boxes:
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            if int(box.cls) != 3:
                content_boxes.append((cords, int(box.cls)))

    return content_boxes

def is_within_box(item, box):
    # Extract the coordinates for the item and the box
    x1, y1 = item[0][0]
    x2, y2 = item[0][2]

    box_x1, box_y1 = box[0][0], box[0][1]
    box_x2, box_y2 = box[0][2], box[0][3]

    # Check if the item is within the box
    if box_x1 <= x1 <= box_x2 and box_y1 <= y1 <= box_y2:
        return True
    if box_x1 <= x2 <= box_x2 and box_y1 <= y2 <= box_y2:
        return True
    if box_x1 <= x1 <= box_x2 and box_y1 <= y2 <= box_y2:
        return True
    if box_x1 <= x2 <= box_x2 and box_y1 <= y1 <= box_y2:
        return True

    return False

def get_ocr_data_in_box(data, box):
    y_tolerance = 50
    content_data = []

    for item in data[0]:
        if is_within_box(item, box):
            content_data.append(item)

    # Adjusted to use minimum x-coordinate for sorting and grouping
    flattened_data = [(rounded_avg_y_coordinate(coords), min_coordinate(coords, 0), text)
                      for coords, (text, _) in content_data]
    flattened_data.sort(key=lambda x: (x[0], x[1]))

    return flattened_data

def min_coordinate(coords, index):
    # Adjusted to return the minimum x or y coordinate from the OCR element
    return min(coord[index] for coord in coords)

def avg_coordinate(coords, index):
    return sum(coord[index] for coord in coords[:2]) / len(coords[:2])

def rounded_avg_y_coordinate(coords, y_tolerance = 50):
    return round(avg_coordinate(coords, 1) / y_tolerance) * y_tolerance


def process_pdf(file_path, table_model):
    text_results = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:

            chunks = []

            page_image = page.to_image(resolution=300).original
            ocr_result = ocr_page(page)

            content_boxes = detect_tables_and_content(page_image, table_model)
            content_boxes.sort(key=lambda x: (int((x[0][1])/50)*50, x[0][0]))


            for object_box in content_boxes:
                # check object detection class

                if object_box[1] in [0,1]:  # process table content
                    box = object_box[0]
                    ## crop image to box
                    cropped_image = page_image.copy().crop((box[0], box[1], box[2], box[3]))
                    ocr_section = ocr_image(cropped_image)

                    ## process table
                    chunks.append(ocr_to_markdown_advanced(ocr_section[0]))

                if object_box[1] in [2,4]:  # process text content
                    text_content = get_ocr_data_in_box(ocr_result, object_box)
                    if text_content: 
                        chunks.append(process_reading_order(text_content))
        
            # normalize and append content to page chunks
            chunks = list(map(normalize_accents,chunks))
            chunks = list(map(drop_divisions,chunks))

            if chunks not in text_results: # maneja esa página duplicada
                text_results.append(chunks)

    return text_results


# process_pdf('/content/DOC-20230901-WA0017_230916_085403.pdf', table_model)
