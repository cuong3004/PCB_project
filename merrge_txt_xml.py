import xml.etree.ElementTree as ET
import os

def yolo_to_voc_bbox(yolo_bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = yolo_bbox
    x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)
    return (xmin, ymin, xmax, ymax, class_id)

def merge_yolo_into_voc(yolo_file, voc_file, class_names):
    tree = ET.parse(voc_file)
    root = tree.getroot()

    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    # print(yolo_file)
    with open(yolo_file, 'r') as yf:
        for line in yf:
            # print(line)
            yolo_bbox = line.strip().split()
            class_id = int(yolo_bbox[0])
            class_name = class_names[class_id]
            if class_name == "IC":
                continue
            xmin, ymin, xmax, ymax, _ = yolo_to_voc_bbox(yolo_bbox, img_width, img_height)

            obj = ET.SubElement(root, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = class_name
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            xmin_elem = ET.SubElement(bndbox, 'xmin')
            xmin_elem.text = str(xmin)
            ymin_elem = ET.SubElement(bndbox, 'ymin')
            ymin_elem.text = str(ymin)
            xmax_elem = ET.SubElement(bndbox, 'xmax')
            xmax_elem.text = str(xmax)
            ymax_elem = ET.SubElement(bndbox, 'ymax')
            ymax_elem.text = str(ymax)

    tree.write(voc_file)

# Tên các lớp tương ứng với các ID trong tệp YOLO
class_names = ["IC", "SMD", "capacitor", "inductor", "other", "resistor", "transistor"]  # Thay thế bằng danh sách các lớp của bạn

import re
def extract_numbers(filename):
    # print(filename)
    filename_ = filename.split("/")[-1]
    match = re.match(r'pcb(\d+)_rec(\d+)\.xml', filename_)
    if match:
        # print(match) 
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')  # Đảm bảo các tệp không phù hợp sẽ được xếp cuối

# Sắp xếp các tệp dựa trên các số đã trích xuất

import os
# Đường dẫn tới tệp YOLO và tệp VOC
# from glob import glob  

dir_voc_files = "Annotations2/content/Annotations"

voc_file_names = os.listdir(dir_voc_files)
# voc_files.sort(reverse=True)

voc_file_names = sorted(voc_file_names, key=extract_numbers)
# print(voc_file_names[:20])

for voc_file_name in voc_file_names[150:200]:
    
    path_voc_file = os.path.join(dir_voc_files, voc_file_name)
    
    yolo_file = "output_txt/output_txt/" + voc_file_name.split(".")[0] + ".txt"
    if os.path.isfile(yolo_file):
        print(path_voc_file)
        merge_yolo_into_voc(yolo_file, path_voc_file, class_names)
