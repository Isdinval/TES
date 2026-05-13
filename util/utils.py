# util/utils.py
# ── Corrections apportées ────────────────────────────────────────────────────
# 1. Suppression de `from openai import AzureOpenAI` (jamais utilisé, absent de requirements)
# 2. easyocr.Reader et PaddleOCR initialisés en lazy (pas au niveau module)
#    → évite le crash au démarrage si paddleocr absent, et respecte model_storage_directory
# 3. Correction du crash `zip(None, ocr_text)` quand ocr_bbox=None (get_som_labeled_img)
# 4. Suppression des imports dupliqués (os x3, json x2, time x2)
# ─────────────────────────────────────────────────────────────────────────────

import os
import io
import base64
import time
import ast
import sys
import re
import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import requests

from typing import Tuple, List, Union
from torchvision.ops import box_convert
from torchvision.transforms import ToPILImage
import torchvision.transforms as T
import supervision as sv

from util.box_annotator import BoxAnnotator


# ── Lazy initialization OCR ──────────────────────────────────────────────────
# Les lecteurs OCR sont initialisés à la première utilisation,
# pas au chargement du module — évite crashes et téléchargements intempestifs.

_easyocr_reader = None
_paddle_ocr      = None


def _get_easyocr_reader(model_storage_directory: str = None):
    """Retourne le reader EasyOCR, initialisé au premier appel."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        kwargs = {}
        if model_storage_directory:
            kwargs['model_storage_directory'] = model_storage_directory
        _easyocr_reader = easyocr.Reader(['en'], **kwargs)
    return _easyocr_reader


def _get_paddle_ocr():
    """Retourne l'instance PaddleOCR, initialisée au premier appel."""
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(
            lang='en',
            use_angle_cls=False,
            use_gpu=False,
            show_log=False,
            max_batch_size=1024,
            use_dilation=True,
            det_db_score_mode='slow',
            rec_batch_num=1024,
        )
    return _paddle_ocr


# ── Modèles de captioning ─────────────────────────────────────────────────────

def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, device_map=None, torch_dtype=torch.float32
            )
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, device_map=None, torch_dtype=torch.float16
            ).to(device)
    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
            ).to(device)
    return {'model': model.to(device), 'processor': processor}


def get_yolo_model(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model


# ── Captioning icônes ─────────────────────────────────────────────────────────

@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=128):
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except Exception:
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i:i+batch_size]
        if model.device.type == 'cuda':
            inputs = processor(
                images=batch, text=[prompt]*len(batch),
                return_tensors="pt", do_resize=False
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch, text=[prompt]*len(batch),
                return_tensors="pt"
            ).to(device=device)
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20, num_beams=1, do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs, max_length=100, num_beams=5,
                no_repeat_ngram_size=2, early_stopping=True,
                num_return_sequences=1,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts


def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs = {'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            inp = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(inp['input_ids'])
            inputs['attention_mask'].append(inp['attention_mask'])
            inputs['pixel_values'].append(inp['pixel_values'])
            inputs['image_sizes'].append(inp['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat(
                [processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1
            )
            inputs['attention_mask'][i] = torch.cat(
                [torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1
            )
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}
        generation_args = {"max_new_tokens": 25, "temperature": 0.01, "do_sample": False}
        generate_ids = model.generate(
            **inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
        )
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts


# ── Suppression des overlaps ──────────────────────────────────────────────────

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    for i, box1 in enumerate(boxes):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                if not any(
                    IoU(box1, box3) > iou_threshold and not is_inside(box1, box3)
                    for k, box3 in enumerate(ocr_bbox)
                ):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    """
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format:    [{'type': 'icon', 'bbox':[x,y], 'interactivity':True,  'content':None }, ...]
    """
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1):
                            try:
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except Exception:
                                continue
                        elif is_inside(box1, box3):
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({
                            'type': 'icon', 'bbox': box1_elem['bbox'],
                            'interactivity': True, 'content': ocr_labels,
                            'source': 'box_yolo_content_ocr',
                        })
                    else:
                        filtered_boxes.append({
                            'type': 'icon', 'bbox': box1_elem['bbox'],
                            'interactivity': True, 'content': None,
                            'source': 'box_yolo_content_yolo',
                        })
            else:
                filtered_boxes.append(box1)
    return filtered_boxes


# ── Utilitaires bbox ──────────────────────────────────────────────────────────

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    return int(x), int(y), int(w), int(h)


def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    return int(x), int(y), int(xp), int(yp)


def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    return int(x), int(y), int(w), int(h)


def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    return (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
    text_scale: float,
    text_padding=5,
    text_thickness=2,
    thickness=3,
) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]
    box_annotator = BoxAnnotator(
        text_scale=text_scale, text_padding=text_padding,
        text_thickness=text_thickness, thickness=thickness,
    )
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections,
        labels=labels, image_size=(w, h),
    )
    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


# ── Prédiction YOLO ───────────────────────────────────────────────────────────

def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    if scale_img:
        result = model.predict(source=image, conf=box_threshold, imgsz=imgsz, iou=iou_threshold)
    else:
        result = model.predict(source=image, conf=box_threshold, iou=iou_threshold)
    boxes   = result[0].boxes.xyxy
    conf    = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]
    return boxes, conf, phrases


def predict(model, image, caption, box_threshold, text_threshold):
    """Grounded DINO (non utilisé dans ce projet)."""
    model, processor = model['model'], model['processor']
    device = model.device
    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


# ── OCR ───────────────────────────────────────────────────────────────────────

def check_ocr_box(
    image_source: Union[str, Image.Image],
    display_img=True,
    output_bb_format='xywh',
    goal_filtering=None,
    easyocr_args=None,
    use_paddleocr=False,
    easyocr_model_dir: str = None,   # ← nouveau paramètre pour pointer vers weights/EasyOCR/
):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    w, h = image_source.size

    if use_paddleocr:
        text_threshold = (easyocr_args or {}).get('text_threshold', 0.5)
        result = _get_paddle_ocr().ocr(image_np, cls=False)[0]
        # FIX : result peut être None si aucun texte détecté
        if result is None:
            return ([], []), goal_filtering
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text  = [item[1][0] for item in result if item[1][1] > text_threshold]
    else:
        if easyocr_args is None:
            easyocr_args = {}
        result = _get_easyocr_reader(easyocr_model_dir).readtext(image_np, **easyocr_args)
        coord = [item[0] for item in result]
        text  = [item[1] for item in result]

    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
        else:
            bb = []

    return (text, bb), goal_filtering


# ── Pipeline principal ────────────────────────────────────────────────────────

def get_som_labeled_img(
    image_source: Union[str, Image.Image],
    model=None,
    BOX_TRESHOLD=0.01,
    output_coord_in_ratio=False,
    ocr_bbox=None,
    text_scale=0.4,
    text_padding=5,
    draw_bbox_config=None,
    caption_model_processor=None,
    ocr_text=[],
    use_local_semantics=True,
    iou_threshold=0.9,
    prompt=None,
    scale_img=False,
    imgsz=None,
    batch_size=128,
):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    image_source = image_source.convert("RGB")
    w, h = image_source.size
    if not imgsz:
        imgsz = (h, w)

    xyxy, logits, phrases = predict_yolo(
        model=model, image=image_source,
        box_threshold=BOX_TRESHOLD,
        imgsz=imgsz, scale_img=scale_img,
        iou_threshold=0.1,
    )
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # ── FIX : ocr_bbox peut être None ou vide ────────────────────────────────
    if ocr_bbox:
        ocr_bbox_tensor = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox_norm   = ocr_bbox_tensor.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox_norm = []   # liste vide au lieu de None → évite le crash zip()

    ocr_bbox_elem = [
        {'type': 'text', 'bbox': box, 'interactivity': False,
         'content': txt, 'source': 'box_ocr_content_ocr'}
        for box, txt in zip(ocr_bbox_norm, ocr_text)
        if int_box_area(box, w, h) > 0
    ]
    xyxy_elem = [
        {'type': 'icon', 'bbox': box, 'interactivity': True, 'content': None}
        for box in xyxy.tolist()
        if int_box_area(box, w, h) > 0
    ]

    filtered_boxes = remove_overlap_new(
        boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem
    )
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
    starting_idx = next(
        (i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1
    )
    filtered_boxes_tensor = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
    print('len(filtered_boxes):', len(filtered_boxes_tensor), starting_idx)

    time1 = time.time()
    if use_local_semantics:
        caption_model = caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type:
            parsed_content_icon = get_parsed_content_icon_phi3v(
                filtered_boxes_tensor, ocr_bbox_elem, image_source, caption_model_processor
            )
        else:
            parsed_content_icon = get_parsed_content_icon(
                filtered_boxes_tensor, starting_idx, image_source,
                caption_model_processor, prompt=prompt, batch_size=batch_size,
            )
        ocr_text_labeled = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text_labeled)
        parsed_content_icon_copy = list(parsed_content_icon)
        for i, box in enumerate(filtered_boxes_elem):
            if box['content'] is None:
                box['content'] = parsed_content_icon_copy.pop(0) if parsed_content_icon_copy else ''
        parsed_content_icon_ls = [
            f"Icon Box ID {str(i+icon_start)}: {txt}"
            for i, txt in enumerate(parsed_content_icon_copy)
        ]
        parsed_content_merged = ocr_text_labeled + parsed_content_icon_ls
    else:
        ocr_text_labeled  = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text_labeled

    print('time to get parsed content:', time.time()-time1)

    filtered_boxes_cxcywh = box_convert(
        boxes=filtered_boxes_tensor, in_fmt="xyxy", out_fmt="cxcywh"
    )
    phrases_int = [i for i in range(len(filtered_boxes_cxcywh))]

    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(
            image_source=image_source,
            boxes=filtered_boxes_cxcywh,
            logits=logits,
            phrases=phrases_int,
            **draw_bbox_config,
        )
    else:
        annotated_frame, label_coordinates = annotate(
            image_source=image_source,
            boxes=filtered_boxes_cxcywh,
            logits=logits,
            phrases=phrases_int,
            text_scale=text_scale,
            text_padding=text_padding,
        )

    pil_img  = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')

    if output_coord_in_ratio:
        label_coordinates = {
            k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h]
            for k, v in label_coordinates.items()
        }
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, filtered_boxes_elem