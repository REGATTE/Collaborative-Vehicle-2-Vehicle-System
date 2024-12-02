import json

# Load JSON files
with open('/data/UCR_student/sds/code/Collaborative-Vehicle-2-Vehicle-System/misc/output/det_bounding_boxes/det_bounding_boxes_1733036940701.json') as f:
    det_bboxes = json.load(f)

with open('/data/UCR_student/sds/code/Collaborative-Vehicle-2-Vehicle-System/misc/output/gt_bounding_boxes/gt_bounding_boxes_1733036940677.json') as f:
    gt_bboxes = json.load(f)

def calculate_volume(extent):
    """Calculate volume of a bounding box given its extent."""
    return extent["x"] * extent["y"] * extent["z"]

def calculate_iou(box1, box2):
    """Calculate IoU for two 3D bounding boxes."""
    # Determine the min and max points for both boxes
    box1_min = {dim: box1["location"][dim] - box1["extent"][dim] / 2 for dim in ["x", "y", "z"]}
    box1_max = {dim: box1["location"][dim] + box1["extent"][dim] / 2 for dim in ["x", "y", "z"]}

    box2_min = {dim: box2["location"][dim] - box2["extent"][dim] / 2 for dim in ["x", "y", "z"]}
    box2_max = {dim: box2["location"][dim] + box2["extent"][dim] / 2 for dim in ["x", "y", "z"]}

    # Calculate overlap in each dimension
    overlap_min = {dim: max(box1_min[dim], box2_min[dim]) for dim in ["x", "y", "z"]}
    overlap_max = {dim: min(box1_max[dim], box2_max[dim]) for dim in ["x", "y", "z"]}
    overlap_extent = {dim: max(0, overlap_max[dim] - overlap_min[dim]) for dim in ["x", "y", "z"]}

    # Calculate intersection and union volumes
    intersection_volume = calculate_volume(overlap_extent)
    volume1 = calculate_volume(box1["extent"])
    volume2 = calculate_volume(box2["extent"])
    union_volume = volume1 + volume2 - intersection_volume

    return intersection_volume / union_volume if union_volume > 0 else 0

# Compute IoU for each pair of det and gt bounding boxes
iou_scores = []
for det_box in det_bboxes:
    for gt_box in gt_bboxes:
        iou_score = calculate_iou(det_box, gt_box)
        iou_scores.append({"det_box": det_box, "gt_box": gt_box, "iou": iou_score})

# Print IoU scores
for score in iou_scores:
    print(f"Det Box: {score['det_box']['location']}, GT Box: {score['gt_box']['location']}, IoU: {score['iou']:.4f}")