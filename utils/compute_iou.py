import os
import json

def compute_iou(box1, box2):
    # Calculate the (x, y) coordinates of the intersection rectangle
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def main():
    input_dir = 'misc/outputs'
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for json_file in json_files:
        with open(os.path.join(input_dir, json_file), 'r') as f:
            data = json.load(f)
            box1 = data['box1']
            box2 = data['box2']
            iou = compute_iou(box1, box2)
            print(f'IoU for {json_file}: {iou}')

if __name__ == "__main__":
    main()