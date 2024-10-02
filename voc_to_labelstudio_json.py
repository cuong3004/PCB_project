import os
import xml.etree.ElementTree as ET
import json
import glob
import uuid

def voc_to_custom_json(voc_file, id, image_path, model_version="version 1", score=0.5):
    tree = ET.parse(voc_file)
    root = tree.getroot()

    filename = os.path.basename(image_path)
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    results = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        result = {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "x": xmin / width * 100,
                "y": ymin / height * 100,
                "width": (xmax - xmin) / width * 100,
                "height": (ymax - ymin) / height * 100,
                "rotation": 0,
                "rectanglelabels": [label]
            },
            "id": str(uuid.uuid4()),
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "manual"
        }
        results.append(result)

    custom_json = {
        "id": id,
        "data": {
            "image": f"/data/upload/1/"+filename
        },
        "predictions": [{
            "model_version": model_version,
            "score": score,
            "result": results
        }]
    }

    return custom_json


import re
def extract_numbers(filename):
    # print(filename)
    #  = filename.split("/")[-1]
    filename_ = os.path.basename(filename)
    filename_ = filename_[len("fa80fb8e-"):]
    match = re.match(r'pcb(\d+)_rec(\d+)\.jpg', filename_)
    if match:
        # print(match) 
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')  # Đảm bảo các tệp không phù hợp sẽ được xếp cuối

def main(args):
    os.makedirs(args.json_output_dir, exist_ok=True)
    img_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))
    img_files = sorted(img_files, key=extract_numbers)
    # print(img_files[:10])
    i = 1
    all_annotations = []
    for img_file in img_files[150:200]:
        if ".jpg" not in img_file:
            continue
        voc_name = os.path.basename(img_file).replace(".jpg", ".xml")
        voc_name_ = voc_name[len("fa80fb8e-"):]
        voc_path = os.path.join(args.voc_dir, voc_name_)
        
        
        
        # json_output_path = os.path.join(args.json_output_dir, image_name.replace(".jpg", ".json"))

        custom_json = voc_to_custom_json(voc_path, i, img_file)
        all_annotations.append(custom_json)
        
        # with open(json_output_path, 'w') as f:
        #     json.dump(custom_json, f, indent=2)
        
        i += 1

    output_file_path = os.path.join(args.json_output_dir, "all_annotations.json")
    with open(output_file_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)
        
if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--voc_dir", type=str, required=True, help="Directory containing Pascal VOC XML files")
    # parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the corresponding images")
    # parser.add_argument("--json_output_dir", type=str, required=True, help="Directory to save the converted JSON files")
    # args = parser.parse_args()
    
    args = argparse.Namespace()
    args.image_dir = r"D:\data\data_pcb\media\upload\1"
    args.voc_dir = r"Annotations2\content\Annotations"
    args.json_output_dir = "json_output_dir"
    main(args)
