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
import hashlib

class SmartVisionSystem:
    def __init__(self, credentials_json_string): 
        if os.path.isfile(credentials_json_string):
            credentials = service_account.Credentials.from_service_account_file(credentials_json_string)
        else:
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
        self.announcement_cooldown = 2.0
        self.detection_history = {}
        self.min_detection_frames = 2
        self.translation_cache = {}
        
        self.last_api_call_time = 0
        self.min_api_interval = 1.2
        
        self.prev_frame_hash = None
        self.scene_cache = {} 
        self.cache_duration = 5.0 
        
        self.api_call_count = {'object': 0, 'ocr': 0}
        self.session_start_time = time.time()
        
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
    
    def get_api_usage_stats(self):
        elapsed = time.time() - self.session_start_time
        minutes = elapsed / 60
        return {
            'object_detection_calls': self.api_call_count['object'],
            'ocr_calls': self.api_call_count['ocr'],
            'total_calls': sum(self.api_call_count.values()),
            'elapsed_minutes': round(minutes, 2)
        }
    
    def calculate_frame_hash(self, image, sample_ratio=0.1):
        h, w = image.shape[:2]
        sample = cv2.resize(image, (int(w * sample_ratio), int(h * sample_ratio)))
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray.tobytes()).hexdigest()
    
    def has_significant_change(self, image):
        current_hash = self.calculate_frame_hash(image)
        if self.prev_frame_hash is None:
            self.prev_frame_hash = current_hash
            return True
        if current_hash == self.prev_frame_hash:
            return False
        self.prev_frame_hash = current_hash
        return True
    
    def check_scene_cache(self, frame_hash):
        current_time = time.time()
        expired_keys = [k for k, (data, timestamp) in self.scene_cache.items() 
                        if current_time - timestamp > self.cache_duration]
        for k in expired_keys: 
            del self.scene_cache[k]
        
        if frame_hash in self.scene_cache:
            cached_data, timestamp = self.scene_cache[frame_hash]
            if current_time - timestamp < self.cache_duration:
                return cached_data
        return None
    
    def cache_scene_result(self, frame_hash, result):
        self.scene_cache[frame_hash] = (result, time.time())
    
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
                         if current_time - last_seen > 3.5]
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
        self.api_call_count['object'] += 1
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
        self.api_call_count['ocr'] += 1
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
        focal_length = image_width * 0.8
        if label == 'Person': 
            focal_length = image_width * 1.3
        bbox_size = max(bbox_width, bbox_height)
        if bbox_size > 0:
            distance_cm = (real_size_cm * focal_length) / bbox_size
            return distance_cm / 100
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
            d_relative = float(np.percentile(roi, 10)) if roi.size > 0 else 0.5
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            distance_from_size = self.estimate_distance_from_size(
                label, bbox_width, bbox_height, w_img
            )
            if distance_from_size is not None:
                estimated_distance = distance_from_size
                estimation_method = 'size_based'
            else:
                estimated_distance = 1.5 + ((0.7 - d_relative) * 12.5)
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
            'Person': 1, 'Car': 1, 'Bicycle': 1, 'Motorcycle': 1, 'Bus': 1,
            'Door': 2, 'Stairs': 2, 'Table': 3, 'Chair': 3, 'Phone': 4
        }
        main_entities = {
            'Person', 'Car', 'Bicycle', 'Motorcycle', 'Building', 
            'Table', 'Desk', 'Chair', 'Couch', 'Bed'
        }
        filtered = []
        objects_sorted = sorted(objects, key=lambda x: object_priority.get(x['label'], 5))
        
        for obj in objects_sorted:
            should_keep = True
            for parent_obj in filtered:
                if parent_obj['label'] not in main_entities: 
                    continue
                bbox_overlap = self._calculate_bbox_overlap(obj['bbox'], parent_obj['bbox'])
                if bbox_overlap > 0.7:
                    should_keep = False
                    break
            if should_keep: 
                filtered.append(obj)
        return filtered
    
    def _calculate_bbox_overlap(self, bbox1, bbox2):
        x1_min, y1_min = bbox1['x_min'], bbox1['y_min']
        x1_max, y1_max = bbox1['x_max'], bbox1['y_max']
        x2_min, y2_min = bbox2['x_min'], bbox2['y_min']
        x2_max, y2_max = bbox2['x_max'], bbox2['y_max']
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min: 
            return 0.0
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        if bbox1_area == 0: 
            return 0.0
        return inter_area / bbox1_area
    
    def filter_text_content(self, text):
        if not text or len(text.strip()) == 0: 
            return None
        text = text.strip()
        digit_count = sum(c.isdigit() for c in text)
        total_chars = len(text)
        if total_chars < 15 and digit_count > total_chars * 0.6: 
            return None
        return text
    
    def format_distance(self, distance_meters):
        cm = int(distance_meters * 100)
        if distance_meters < 0.5: 
            ar = "أقل من نص متر - قريب جداً"
        elif distance_meters < 1.0: 
            ar = "نص متر"
        elif distance_meters < 5.0: 
            ar = f"{round(distance_meters, 1)} متر"
        else: 
            ar = f"{round(distance_meters, 1)} متر"
        return {
            'ar': ar, 
            'en': None, 
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
        except: 
            return text
    
    def generate_message(self, obj):
        label = obj['label']
        distance_m = obj['estimated_distance']
        texts = obj.get('texts', [])
        position = obj['position']
        distance_format = self.format_distance(distance_m)
        label_ar = self.translate_with_cache(label)
        
        urgency = "DANGER" if distance_m < 0.5 else ("WARNING" if distance_m < 1.5 else "INFO")
        
        important_text_objects = ['Traffic sign', 'Sign', 'Billboard', 'Building', 'Door']
        filtered_texts = []
        if texts and label in important_text_objects:
            for text in texts[:2]:
                filtered = self.filter_text_content(text)
                if filtered: 
                    filtered_texts.append(filtered)
        
        if filtered_texts:
            text_str = " ".join(filtered_texts)
            text_ar = self.translate_with_cache(text_str) if not self._is_arabic(text_str) else text_str
            msg_ar = f"{label_ar} {position['ar']}، مكتوب: {text_ar}"
        else:
            msg_ar = f"{label_ar} {position['ar']}، على بعد {distance_format['ar']}"
            
        return {
            'urgency': urgency, 
            'message_ar': msg_ar, 
            'distance_meters': distance_m, 
            'position': position
        }
    
    def _is_arabic(self, text):
        arabic_chars = 0
        total_chars = 0
        for char in text:
            if char.isalpha():
                total_chars += 1
                if '\u0600' <= char <= '\u06FF': 
                    arabic_chars += 1
        return total_chars > 0 and (arabic_chars / total_chars) > 0.5

    def process_image_from_flutter(self, base64_image, conf_threshold=0.5, enable_ocr=True, force_announce=False):

        current_time = time.time()
        
        if not force_announce and (current_time - self.last_api_call_time < self.min_api_interval):
            return None 

        try:
            image = self.decode_base64_image(base64_image)
            
            if not force_announce and not self.has_significant_change(image):
                return None

            frame_hash = self.calculate_frame_hash(image)
            cached_result = self.check_scene_cache(frame_hash)
            if cached_result and not force_announce:
                return cached_result

            self.last_api_call_time = current_time
            
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
                obj_key = f"{obj['label']}_{int(guidance['distance_meters']*10)}"
                
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
                msg = item['message_ar']
                if msg not in seen_combinations:
                    seen_combinations.add(msg)
                    unique_announcements.append(item)
            
            top_messages = [item['message_ar'] for item in unique_announcements[:3]]
            combined_text = ". ".join(top_messages)
            
            audio_base64 = self.generate_audio(combined_text, language='ar')
            
            final_result = {
                'success': True,
                'speech_text': combined_text,         
                'audio_file': audio_base64,            
                'messages': top_messages,               
                'has_audio': audio_base64 is not None,  
                'stats': self.get_api_usage_stats()
            }
            
            self.cache_scene_result(frame_hash, final_result)
            return final_result
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'speech_text': None,
                'audio_file': None,
                'has_audio': False
            }

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
            print(f"Error generating audio: {e}")
            return None
