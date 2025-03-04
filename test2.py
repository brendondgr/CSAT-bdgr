import numpy as np
from sklearn.metrics import auc
import torch

import numpy as np
from sklearn.metrics import auc

from loss.box_ops import box_cxcywh_to_xyxy

def calculate_iou(box1, box2):
	"""
	Calculate Intersection over Union (IoU) between two bounding boxes.
	Each box is in the format [x1, y1, x2, y2].
	"""
	box1[1] = 0.5
	box1[3] = 0.9
	box1 = box_cxcywh_to_xyxy(box1)
	box2 = box_cxcywh_to_xyxy(box2)

	x1 = max(box1[0], box2[0])
	y1 = max(box1[1], box2[1])
	x2 = min(box1[2], box2[2])
	y2 = min(box1[3], box2[3])
	
	# print(f"Box1: {box1}, Box2: {box2}")

	# assert box1[0] < box1[2] and box1[1] < box1[3], "Invalid bounding boxes: no intersection"

	inter_area = max(0, x2 - x1) * max(0, y2 - y1)
	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
	union_area = box1_area + box2_area - inter_area

	return inter_area / union_area if union_area > 0 else 0

def calculate_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.2,
                 opts=None, target=None, class_id=None, list_of_stuff=None):
	"""
	Calculate Average Precision (AP) for a single class.
	Args:
		gt_boxes: Ground truth boxes for the class, shape (N, 4).
		pred_boxes: Predicted boxes, shape (M, 4).
		pred_scores: Confidence scores for the predicted boxes, shape (M,).
		iou_threshold: IoU threshold to consider a prediction as a true positive.
	"""
	tp = np.zeros(len(pred_boxes))
	fp = np.zeros(len(pred_boxes))
	used_gt_boxes = set()

	for i, pred_box in enumerate(pred_boxes):
		best_iou = 0
		best_gt_idx = -1

		for j, gt_box in enumerate(gt_boxes):
			if j in used_gt_boxes:
				continue
			iou = calculate_iou(pred_box, gt_box)
			if opts != None and opts.debug: print(f'IoU: {iou}')
			if iou > best_iou:
				best_iou = iou
				best_gt_idx = j

		if opts != None and opts.debug:
			print(f"Best IoU: {best_iou}, Best GT Index: {best_gt_idx}")

		if best_iou >= iou_threshold:
			tp[i] = 1
			used_gt_boxes.add(best_gt_idx)
		else:
			fp[i] = 1
	
	# gt_boxes, gt_scores(targets), pred_boxes, pred_scores
	# Create Dictionary with above values.
	dictionary = {
		'class_id': class_id,
		'gt_boxes': gt_boxes.tolist() if isinstance(gt_boxes, torch.Tensor) else gt_boxes,
		'gt_scores': target.tolist() if isinstance(target, torch.Tensor) else target,
		'pred_boxes': pred_boxes.tolist() if isinstance(pred_boxes, torch.Tensor) else pred_boxes,
		'pred_scores': pred_scores.tolist() if isinstance(pred_scores, torch.Tensor) else pred_scores
	}
	# Append to List of stuff
	if list_of_stuff is not None: list_of_stuff.append(dictionary)
 
 
	# Cumulative TP and FP
	tp_cumsum = np.cumsum(tp)
	fp_cumsum = np.cumsum(fp)
	if opts != None:
		if opts.debug: print(tp_cumsum, fp_cumsum)
	return tp_cumsum, fp_cumsum


def get_ap(precision, recall, opts=None):
	# Ensure Precision and Recall are monotonic
	precision = np.maximum.accumulate(precision)
 
	# If self.debug
	if opts is not None and opts.debug:
		print(f"Precision shape: {precision.shape}, Recall shape: {recall}")
		if precision.size > 0:
			print(f"Precision: {precision}")

	try:
		# AP calculation (area under the Precision-Recall curve)
		ap = auc(recall, precision)
		return ap
	except:
		return 0.0

def compute_val_stats(outputs, objthres=0.4, phase='val', device='cpu', opts=None):
	outputs_class, outputs_coord = outputs
	outputs_class = outputs_class[0]
	outputs_coord = outputs_coord[0]

	# print(targets)
	# Calculate objectness score
	objectness = torch.softmax(outputs_class, dim=-1)
	# if opts is not None and opts.debug: print(f'Objectness: {objectness}')

	indices = torch.argmax(objectness, dim=-1)	
	# if opts is not None and opts.debug: print(f'Indices: {indices}')

	topobjectness  = torch.max(objectness, dim=-1)[0]	
	idx_array = torch.arange(len(objectness))# [N]
	indices = indices[topobjectness>objthres]						# class indices after filtering [M] [2]
	
	idx_array = idx_array.to(device)
	topobjectness = topobjectness.to(device)
	
	idx_array = idx_array[topobjectness>objthres]
	# if opts is not None and opts.debug:
	# 	print(f'Top Objectness (>{objthres}): {topobjectness[topobjectness>objthres]}')
	# 	print(f'Indices: {indices}')

	predicted_class = indices
	predicted_box = outputs_coord[idx_array]

	return predicted_class, predicted_box


def calculate_map(target, output, iou_threshold=0.3, opts=None):
	"""
	Calculate mean Average Precision (mAP) for all classes.
	Args:
		target: Ground truth boxes, shape (N, 5) where each row is [class_id, x1, y1, x2, y2].
		output: A tuple (output_class, output_boxes):
			- output_class: Confidence scores for each class, shape (M, 3).
			- output_boxes: Predicted boxes, shape (M, 4).
		iou_threshold: IoU threshold to consider a prediction as a true positive.
	"""
	output_class, output_boxes = output
	num_classes = 3 # Number of classes (3 in this case)
	tps = []
	fps = []
	list_of_stuff = []

	for class_id in range(1,num_classes):
		# Filter ground truth boxes for the current class
		gt_boxes = target[target[:, 0] == class_id][:, 1:]  # Shape (N_class, 4)
		
		# print('target boxes for this class', gt_boxes)

		# Filter predictions for the current class
		pred_scores = output_class[output_class==class_id]  # Confidence scores for the current class
		pred_boxes = output_boxes[output_class==class_id]  # All predicted boxes (M, 4)

		# if opts is not None and opts.debug:
		# 	# Print Scores and Boxes
		# 	print(f"Predicted Scores for Class {class_id}: {pred_scores}")
		# 	print(f"Predicted Boxes for Class {class_id}: {pred_boxes}")

		# print(pred_scores, pred_boxes)

		# Skip if there are no ground truth boxes for this class
		if len(gt_boxes) == 0:
			continue

		# Calculate AP for the current class
		tp, fp = calculate_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold,
                        opts=opts, target=target, class_id=class_id, list_of_stuff=list_of_stuff)
  
		tps.extend(tp)
		fps.extend(fp)
		# aps.append(ap)

	return tps, fps, list_of_stuff


if __name__=='__main__':
	op_class = torch.rand((256, 3))
	op_box = torch.rand((256, 4))
	target_class = torch.tensor([[1, 2, 1, 1, 1]]).reshape((-1, 1))
	
	target_box = torch.rand((5, 4))
	target = torch.concatenate([target_class, target_box], dim=-1)
	output = (op_class, op_box)
	
	tps, fps = calculate_map(target, compute_val_stats(output))
	gtl = len(target_box)

	precision = [tp/(tp+fp) for tp, fp in zip(tps, fps)]
	recall = [tp/gtl for tp in tps]

	ap = get_ap(precision, recall)
	print(ap)
	