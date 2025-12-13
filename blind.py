import cv2
import numpy as np
import torch
from google.cloud import vision
from google.oauth2 import service_account
import base64
import io
from PIL import Image
from gtts import gTTS
import os
import time
from deep_translator import GoogleTranslator
import json

class SmartVisionSystem:
    def __init__(self, credentials_json_string): 
        credentials_dict = json.loads(credentials_json_string)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model.to(self.device)
        self.depth_model.eval()
        
        self.translator = GoogleTranslator(source='auto', target='ar')
        
        self.last_announcements = {}
        self.announcement_cooldown = 1.5
        self.detection_history = {}
        self.min_detection_frames = 3
        self.translation_cache = {}
        
        self.depth_only_objects = {
            'Street light', 'Traffic light', 'Traffic sign', 'Pole',
            'Building', 'Tree', 'Billboard', 'Sign', 'Streetlight', 'Lamppost',
        }
        
        self.object_sizes = {
            'Person': 50, 'Face': 18, 'Glasses': 14, 'Headphones': 18,
            'Phone': 16.8, 'Laptop': 35, 'Bottle': 16.51, 'Cup': 16.5,
            'Book': 15, 'Door': 90, 'Television': 100, 'Chair': 91,
            'Table': 150, 'Car': 180, 'Bicycle': 58, 'Hand': 10,
            'Head': 20, 'Bag': 30, 'Backpack': 35, 'Handbag': 25,
            'Clock': 25, 'Vase': 15, 'Plant': 20, 'Shoe': 25,
            'Window': 80, 'Desk': 70, 'Couch': 180, 'Bed': 200, 
        }
    
    def decode_base64_image(self, base64_string):
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(img_data))
        img_array = np.array(pil_image)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        return img_bgr
    
    def get_position_description(self, center_x, image_width):
        left_boundary = image_width * 0.33
        right_boundary = image_width * 0.67
        
        if center_x < left_boundary:
            return {'ar': 'على يمينك', 'position': 'right'}
        elif center_x > right_boundary:
            return {'ar': 'على يسارك', 'position': 'left'}
        else:
            return {'ar': 'أمامك مباشرة', 'position': 'center'}
    
    def calculate_priority(self, urgency, distance_meters):
        urgency_scores = {'DANGER': 0, 'WARNING': 100, 'INFO': 200}
        base = urgency_scores.get(urgency, 200)
        distance = int(distance_meters * 10)
        return base + distance
    
    def should_announce(self, obj_key):
        current_time = time.time()
        last_time = self.last_announcements.get(obj_key, 0)
        
        if current_time - last_time >= self.announcement_cooldown:
            self.last_announcements[obj_key] = current_time
            return True
        return False
    
    def update_detection_history(self, obj_key):
        current_time = time.time()
        
        keys_to_remove = [k for k, (count, last_seen) in self.detection_history.items()
                         if current_time - last_seen > 2.0]
        for k in keys_to_remove:
            del self.detection_history[k]
        
        if obj_key in self.detection_history:
            count, _ = self.detection_history[obj_key]
            self.detection_history[obj_key] = (count + 1, current_time)
        else:
            self.detection_history[obj_key] = (1, current_time)
        
        count, _ = self.detection_history[obj_key]
        return count >= self.min_detection_frames
    
    def detect_objects(self, image, conf_threshold=0.5):
        _, encoded = cv2.imencode('.jpg', image)
        content = encoded.tobytes()
        
        vision_image = vision.Image(content=content)
        objects = self.vision_client.object_localization(image=vision_image).localized_object_annotations

        results = {'labels': [], 'confidences': [], 'bboxes': []}
        h, w = image.shape[:2]
        
        for obj in objects:
            if obj.score < conf_threshold:
                continue    
                       
            results['labels'].append(obj.name)
            results['confidences'].append(float(obj.score))
            
            verts = obj.bounding_poly.normalized_vertices
            x_coords = [v.x * w for v in verts]
            y_coords = [v.y * h for v in verts]
            
            bbox = {
                'x_min': int(min(x_coords)),
                'y_min': int(min(y_coords)),
                'x_max': int(max(x_coords)),
                'y_max': int(max(y_coords))
            }
            results['bboxes'].append(bbox)
        
        return results
    
    def detect_text_ocr(self, image):
        _, encoded = cv2.imencode('.jpg', image)
        content = encoded.tobytes()
        
        vision_image = vision.Image(content=content)
        response = self.vision_client.text_detection(image=vision_image)
        texts = response.text_annotations
        
        if not texts:
            return []
        
        detected = []
        for text in texts[1:]:
            verts = text.bounding_poly.vertices
            cx = sum([v.x for v in verts]) / len(verts)
            cy = sum([v.y for v in verts]) / len(verts)
            
            detected.append({
                'text': text.description,
                'cx': int(cx),
                'cy': int(cy)
            })
        
        return detected
    
    def match_text_to_objects(self, objects_with_depth, texts):
        for obj in objects_with_depth:
            bbox = obj['bbox']
            obj['texts'] = []
            
            for t in texts:
                if (bbox['x_min'] <= t['cx'] <= bbox['x_max'] and
                    bbox['y_min'] <= t['cy'] <= bbox['y_max']):
                    obj['texts'].append(t['text'])
        
        return objects_with_depth
    
    def estimate_depth(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = self.depth_transform(rgb).to(self.device)
        
        with torch.no_grad():
            pred = self.depth_model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        
        depth = pred.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth
    
    def estimate_distance_from_size(self, label, bbox_width, bbox_height, image_width):
        if label in self.depth_only_objects:
            return None
        
        real_size_cm = self.object_sizes.get(label, None)
        
        if real_size_cm is None:
            return None
        
        if label == 'Person':
            focal_length = image_width * 1.3
            bbox_size = bbox_width
            
            if bbox_width > image_width * 0.4:
                focal_length = image_width * 1.0
        elif label in ['Phone', 'Bottle', 'Cup', 'Book', 'Glasses', 'Hand']:
            focal_length = image_width * 0.85
            bbox_size = max(bbox_width, bbox_height)
        elif label in ['Television', 'Door', 'Table', 'Car', 'Couch', 'Bed']:
            focal_length = image_width * 0.9
            bbox_size = max(bbox_width, bbox_height)
        else:
            focal_length = image_width * 0.8
            bbox_size = max(bbox_width, bbox_height)
        
        if bbox_size > 0:
            distance_cm = (real_size_cm * focal_length) / bbox_size
            distance_m = distance_cm / 100
            
            if label == 'Person':
                distance_m = max(0.3, min(distance_m, 10.0))
            elif label in ['Phone', 'Bottle', 'Cup', 'Book', 'Glasses']:
                distance_m = max(0.2, min(distance_m, 5.0))
            elif label in ['Television', 'Door', 'Car']:
                distance_m = max(0.5, min(distance_m, 20.0))
            else:
                distance_m = max(0.2, min(distance_m, 15.0))
            
            return distance_m
        
        return None
    
    def extract_depth_info(self, detection_results, depth_map, image_shape):
        objects = []
        h_img, w_img = image_shape[:2]
        
        for i, label in enumerate(detection_results['labels']):
            bbox = detection_results['bboxes'][i]
            conf = detection_results['confidences'][i]
            
            center_x = (bbox['x_min'] + bbox['x_max']) // 2
            center_y = (bbox['y_min'] + bbox['y_max']) // 2
            
            h, w = depth_map.shape
            x1 = max(0, bbox['x_min'])
            y1 = max(0, bbox['y_min'])
            x2 = min(w, bbox['x_max'])
            y2 = min(h, bbox['y_max'])
            
            roi = depth_map[y1:y2, x1:x2]
            
            if roi.size > 0:
                d_relative = float(np.percentile(roi, 10))
            else:
                d_relative = 0.5
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            distance_from_size = self.estimate_distance_from_size(
                label, bbox_width, bbox_height, w_img
            )
            
            if distance_from_size is not None:
                estimated_distance = distance_from_size
                estimation_method = 'size_based'
            else:
                if d_relative > 0.85:
                    estimated_distance = 0.2 + ((1.0 - d_relative) * 2.0)
                elif d_relative > 0.7:
                    estimated_distance = 0.5 + ((0.85 - d_relative) * 6.67)
                elif d_relative > 0.5:
                    estimated_distance = 1.5 + ((0.7 - d_relative) * 12.5)
                elif d_relative > 0.3:
                    estimated_distance = 4.0 + ((0.5 - d_relative) * 30.0)
                else:
                    estimated_distance = 10.0 + ((0.3 - d_relative) * 50.0)
                
                estimation_method = 'depth_based'
            
            position_info = self.get_position_description(center_x, w_img)
            
            if estimated_distance > 5.0:
                continue
            
            objects.append({
                'label': label,
                'confidence': conf,
                'bbox': bbox,
                'center_x': center_x,
                'center_y': center_y,
                'd_relative': d_relative,
                'estimated_distance': estimated_distance,
                'estimation_method': estimation_method,
                'bbox_size': (bbox_width, bbox_height),
                'position': position_info,
                'texts': []
            })
        
        return objects
    
    def filter_redundant_objects(self, objects):
        always_ignored = {
            'Wheel', 'Tire', 'Window', 'License plate', 'Headlight', 
            'Automotive lighting', 'Auto part', 'Vehicle door', 
            'Light fixture', 'Roof', 'Wall', 'Ceiling', 'Floor'
        }
        
        objects = [obj for obj in objects if obj['label'] not in always_ignored]
        
        object_priority = {
            'Person': 1, 'Car': 1, 'Bicycle': 1, 'Motorcycle': 1,
            'Bus': 1, 'Truck': 1, 'Animal': 1, 'Dog': 1, 'Cat': 1,
            'Door': 2, 'Stairs': 2, 'Building': 2, 'Obstacle': 2,
            'Table': 3, 'Chair': 3, 'Couch': 3, 'Bed': 3,
            'Glasses': 5, 'Shoe': 5, 'Bag': 4, 'Phone': 4, 'Book': 5,
            'Clothing': 8, 'Face': 9, 'Hand': 9, 'Head': 9
        }
        
        smart_relations = {
            'Car': {
                'always_part': ['Wheel', 'Tire', 'License plate', 'Window', 'Headlight'],
                'sometimes_part': ['Door', 'Bag']
            },
            'Building': {
                'always_part': ['Window', 'Wall', 'Roof', 'Ceiling'],
                'sometimes_part': ['Door']
            },
            'Person': {
                'always_part': ['Face', 'Head', 'Hand', 'Arm', 'Leg', 'Eye', 'Nose', 'Ear', 'Clothing', 'Shirt', 'Pants'],
                'sometimes_part': ['Glasses', 'Shoe', 'Footwear', 'Bag', 'Backpack', 'Phone', 'Book']
            },
            'Table': {
                'sometimes_part': ['Book', 'Phone', 'Cup', 'Glasses', 'Laptop']
            },
            'Bicycle': {
                'always_part': ['Wheel', 'Tire']
            },
            'Motorcycle': {
                'always_part': ['Wheel', 'Tire']
            }
        }
        
        filtered = []
        objects_sorted = sorted(objects, key=lambda x: object_priority.get(x['label'], 5))
        
        for obj in objects_sorted:
            should_keep = True
            
            for parent_obj in filtered:
                parent_label = parent_obj['label']
                
                if parent_label not in smart_relations:
                    continue
                
                relations = smart_relations[parent_label]
                
                if obj['label'] in relations.get('always_part', []):
                    distance_diff = abs(obj['estimated_distance'] - parent_obj['estimated_distance'])
                    if distance_diff < 2.0:
                        should_keep = False
                        break
                
                elif obj['label'] in relations.get('sometimes_part', []):
                    distance_diff = abs(obj['estimated_distance'] - parent_obj['estimated_distance'])
                    
                    if distance_diff < 0.8:
                        should_keep = False
                        break
                    elif distance_diff >= 0.8:
                        should_keep = True
            
            if should_keep:
                filtered.append(obj)
        
        return filtered
    
    def filter_text_content(self, text):
        if not text or len(text.strip()) == 0:
            return None
        
        text = text.strip()
        digit_count = sum(c.isdigit() for c in text)
        alpha_count = sum(c.isalpha() for c in text)
        total_chars = len(text)
        
        if total_chars < 15 and digit_count > total_chars * 0.6:
            return None
        
        if total_chars <= 10 and alpha_count <= 3 and digit_count > 0:
            return None
        
        return text
    
    def format_distance(self, distance_meters):
        cm = int(distance_meters * 100)
        
        if distance_meters < 0.5:
            ar = "أقل من نص متر - قريب جداً"
        
        elif distance_meters < 1.0:
            ar = "نص متر"
        
        elif distance_meters < 2.0:
            meters = int(distance_meters)
            remaining_cm = cm - (meters * 100)
            
            if remaining_cm < 5:
                ar = "متر واحد تقريباً"
            else:
                ar = f"متر و{remaining_cm} سنتيمتر"
        
        elif distance_meters < 5.0:
            meters = round(distance_meters, 1)
            
            if meters == int(meters):
                meters = int(meters)
                arabic_numbers = {
                    2: "مترين", 3: "ثلاثة أمتار", 4: "أربعة أمتار",
                    5: "خمسة أمتار"
                }
                ar = arabic_numbers.get(meters, f"{meters} أمتار")
            else:
                ar = f"{meters} متر"
                    
        return {
            'ar': ar,
            'en':None,
            'exact_meters': round(distance_meters, 2),
            'exact_cm': cm
        }
    
    def translate_with_cache(self, text):
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        try:
            translated = self.translator.translate(text)
            self.translation_cache[text] = translated
            return translated
        except Exception as e:
            return text
    
    def generate_message(self, obj):
        label = obj['label']
        distance_m = obj['estimated_distance']
        texts = obj.get('texts', [])
        position = obj['position']
        
        distance_format = self.format_distance(distance_m)
        distance_ar = distance_format['ar']
        position_ar = position['ar']
        
        label_ar = self.translate_with_cache(label)
        
        if distance_m < 0.5:
            urgency = "DANGER"
        elif distance_m < 1.5:
            urgency = "WARNING"
        else:
            urgency = "INFO"
        
        important_text_objects = ['Traffic sign', 'Sign', 'Billboard', 'Building', 
                                 'Door', 'Street light', 'Traffic light', 'Pole']
        
        filtered_texts = []
        if texts and label in important_text_objects:
            for text in texts[:3]:
                filtered = self.filter_text_content(text)
                if filtered:
                    filtered_texts.append(filtered)
        
        if filtered_texts:
            text_str = " ".join(filtered_texts[:2])
            text_ar = self.translate_with_cache(text_str) if not self._is_arabic(text_str) else text_str
            
            if urgency == "DANGER":
                msg_ar = f"احذر! {label_ar} {position_ar}، على بعد {distance_ar}، مكتوب: {text_ar}"
            elif urgency == "WARNING":
                msg_ar = f"انتبه، {label_ar} {position_ar}، على بعد {distance_ar}، مكتوب: {text_ar}"
            else:
                msg_ar = f"{label_ar} {position_ar}، على بعد {distance_ar}، مكتوب: {text_ar}"
        else:
            if urgency == "DANGER":
                msg_ar = f"احذر! {label_ar} {position_ar}، على بعد {distance_ar}"
            elif urgency == "WARNING":
                msg_ar = f"انتبه، {label_ar} {position_ar}، على بعد {distance_ar}"
            else:
                msg_ar = f"{label_ar} {position_ar}، على بعد {distance_ar}"
        
        return {
            'urgency': urgency,
            'message_ar': msg_ar,
            'distance_meters': distance_m,
            'distance_formatted': distance_format,
            'position': position,
            'has_text': len(filtered_texts) > 0,
            'texts': filtered_texts
        }
    
    def _is_arabic(self, text):
        arabic_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if '\u0600' <= char <= '\u06FF':
                    arabic_chars += 1
        
        return total_chars > 0 and arabic_chars / total_chars > 0.5
    
    def process_image_from_flutter(self, base64_image, conf_threshold=0.5, enable_ocr=True, force_announce=False):
        try:
            image = self.decode_base64_image(base64_image)
            detection = self.detect_objects(image, conf_threshold)
            
            if not detection['labels']:
                return None
            
            depth_map = self.estimate_depth(image)
            objects = self.extract_depth_info(detection, depth_map, image.shape)
            
            objects = self.filter_redundant_objects(objects)
            
            all_texts = []
            if enable_ocr:
                all_texts = self.detect_text_ocr(image)
                objects = self.match_text_to_objects(objects, all_texts)
            
            announcements = []
            
            for obj in objects:
                guidance = self.generate_message(obj)
                priority = self.calculate_priority(guidance['urgency'], guidance['distance_meters'])
                
                obj_key = f"{obj['label']}_{int(guidance['distance_meters']*10)}_{obj['position']['position']}"
                
                if force_announce:
                    should_announce = True
                else:
                    is_consistent = self.update_detection_history(obj_key)
                    should_announce = is_consistent and self.should_announce(obj_key)
                
                if should_announce:
                    announcements.append({
                        'priority': priority,
                        'message_ar': guidance['message_ar']
                    })
            
            if not announcements:
                return None
            
            announcements.sort(key=lambda x: x['priority'])
            
            unique_announcements = []
            seen_combinations = set()
            
            for item in announcements:
                message = item['message_ar']
                
                key_parts = []
                
                if 'احذر!' in message:
                    obj_type = message.split('احذر!')[1].strip().split()[0]
                elif 'انتبه،' in message:
                    obj_type = message.split('انتبه،')[1].strip().split()[0]
                else:
                    obj_type = message.strip().split()[0]
                
                key_parts.append(obj_type)
                
                if 'أمامك' in message:
                    key_parts.append('أمامك')
                elif 'على يمينك' in message:
                    key_parts.append('يمينك')
                elif 'على يسارك' in message:
                    key_parts.append('يسارك')
                
                unique_key = '_'.join(key_parts)
                
                if unique_key not in seen_combinations:
                    seen_combinations.add(unique_key)
                    unique_announcements.append(item)
            
            top_messages = [item['message_ar'] for item in unique_announcements[:3]]
            combined_text = ". ".join(top_messages)
            
            audio_base64 = self.generate_audio(combined_text, language='ar')
            
            result = {
                'text': combined_text,
                'audio_base64': audio_base64,
                'messages': top_messages
            }
            
            return result
            
        except Exception as e:
            return None
    
    def generate_audio(self, text, language='ar'):
        try:
            timestamp = int(time.time() * 1000)
            filename = f"audio_{timestamp}.mp3"
            
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(filename)
            
            with open(filename, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            os.remove(filename)
            
            return audio_base64
        except Exception as e:
            return None
