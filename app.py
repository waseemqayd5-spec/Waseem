#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
╔══════════════════════════════════════════════════════════════════╗
║     🚁 النظام الاستخباراتي العسكري الأسطوري - الإصدار 9.1       ║
╠══════════════════════════════════════════════════════════════════╣
║  القدرات العسكرية:                                              ║
║  ✓ تحديد وتتبع الأهداف العسكرية                                 ║
║  ✓ استخبارات إشارية (SIGINT)                                    ║
║  ✓ رؤية ليلية وتصوير حراري                                      ║
║  ✓ نظام تسليح ذكي                                               ║
║  ✓ اتصالات مشفرة ومقاومة للتشويش                                ║
║  ✓ خرائط استخباراتية ثلاثية الأبعاد                             ║
║  ✓ ذكاء اصطناعي لاتخاذ القرارات التكتيكية                       ║
║  ✓ تحليل تهديدات فوري                                            ║
║  ✓ كاميرا جوال + YOLOv8 للكشف الفوري                            ║
╠══════════════════════════════════════════════════════════════════╣
║  🎖️ م/ وسيم الحميدي - هندسة الأنظمة الاستخباراتية               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import http.server
import socketserver
import json
import math
import time
import threading
import random
import numpy as np
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import base64
import os
import cv2
import urllib.parse
import requests
import hashlib
import io
from PIL import Image
import socket
import struct
import re
import subprocess
import sys

# =====================================================
# 📱 YOLOv8 للكشف عن الأجسام عبر كاميرا الجوال
# =====================================================

YOLO_AVAILABLE = False
YOLO_MODEL = None

try:
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("yolov8n.pt")
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 تم تحميله بنجاح")
except ImportError:
    print("⚠️ جاري تثبيت ultralytics...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', '-q'])
        from ultralytics import YOLO
        YOLO_MODEL = YOLO("yolov8n.pt")
        YOLO_AVAILABLE = True
        print("✅ YOLOv8 تم تثبيته وتحميله بنجاح")
    except Exception as e:
        print(f"⚠️ YOLOv8 غير متاح (وضع المحاكاة): {e}")

# =====================================================
# 🤖 Google Gemini API
# =====================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC1MNg7MndyE-glmyTV0YqoLVcLm_jQuvc")
GEMINI_AVAILABLE = False
GEMINI_MODEL = None

MODELS_TO_TRY = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-pro",
    "gemini-1.0-pro"
]

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    for model_name in MODELS_TO_TRY:
        try:
            test_model = genai.GenerativeModel(model_name)
            test_response = test_model.generate_content("test")
            GEMINI_MODEL = test_model
            GEMINI_AVAILABLE = True
            print(f"✅ Gemini API متصل بنجاح: {model_name}")
            break
        except Exception as e:
            print(f"⚠️ فشل النموذج {model_name}: {e}")
            continue

    if not GEMINI_AVAILABLE:
        try:
            models = genai.list_models()
            available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            if available:
                GEMINI_MODEL = genai.GenerativeModel(available[0].split('/')[-1])
                GEMINI_AVAILABLE = True
                print(f"✅ تم استخدام النموذج: {available[0]}")
        except:
            pass

except ImportError:
    print("⚠️ جاري تثبيت google-generativeai...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'google-generativeai', '-q'])
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel('gemini-pro')
        GEMINI_AVAILABLE = True
        print("✅ تم تثبيت وتشغيل Gemini API")
    except Exception as e:
        print(f"⚠️ Gemini API غير متاح: {e}")

if not GEMINI_AVAILABLE:
    print("⚠️ سيتم استخدام وضع المحاكاة بدلاً من Gemini API")

# =====================================================
# 📱 نظام كاميرا الجوال المتقدم
# =====================================================

class MobileCameraSystem:
    def __init__(self):
        self.camera_url = None
        self.cap = None
        self.streaming = False
        self.current_frame = None
        self.last_detection = None
        self.detection_history = deque(maxlen=100)
        self.stream_thread = None
        self.frame_callback = None
        self.last_frame_base64 = None

    def set_camera_url(self, url: str) -> bool:
        self.camera_url = url
        if self.cap:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(url)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"✅ تم الاتصال بالكاميرا: {url}")
                return True
            else:
                print(f"❌ فشل الاتصال بالكاميرا: {url}")
                self.cap = None
                return False
        except:
            self.cap = None
            return False

    def start_streaming(self, callback=None):
        if not self.cap or not self.cap.isOpened():
            return False
        self.streaming = True
        self.frame_callback = callback

        def stream_loop():
            while self.streaming:
                try:
                    ret, frame = self.cap.read()
                    if ret:
                        self.current_frame = frame
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        self.last_frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        if callback:
                            callback(frame)
                    else:
                        time.sleep(0.5)
                        if self.cap:
                            self.cap.release()
                        self.cap = cv2.VideoCapture(self.camera_url)
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except:
                    time.sleep(0.5)

        self.stream_thread = threading.Thread(target=stream_loop, daemon=True)
        self.stream_thread.start()
        return True

    def stop_streaming(self):
        self.streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def capture_frame(self) -> Optional[np.ndarray]:
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def frame_to_base64(self, frame) -> str:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')

    def get_frame_as_base64(self) -> Optional[str]:
        if self.current_frame is not None:
            return self.frame_to_base64(self.current_frame)
        return None


# =====================================================
# 🎯 نظام الكشف بالذكاء الاصطناعي (YOLOv8)
# =====================================================

class AIDetectionSystem:
    def __init__(self):
        self.yolo_available = YOLO_AVAILABLE
        self.model = YOLO_MODEL
        self.detections = deque(maxlen=200)
        self.threat_mapping = {
            'person': 5, 'car': 6, 'truck': 7, 'bus': 6,
            'motorcycle': 4, 'bicycle': 2, 'knife': 8, 'gun': 10,
            'rifle': 10, 'pistol': 9, 'cell phone': 3, 'laptop': 2,
            'backpack': 4, 'dog': 3, 'cat': 2, 'bird': 1,
            'airplane': 9, 'drone': 9, 'boat': 5, 'umbrella': 2
        }

    def detect_objects(self, frame, conf_threshold=0.5):
        if not self.yolo_available or self.model is None:
            return self._simulate_detection(frame), frame

        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        threat_level = self.threat_mapping.get(class_name.lower(), 3)

                        detections.append({
                            'class': class_name,
                            'confidence': round(conf, 3),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'threat_level': threat_level,
                            'is_hostile': threat_level >= 7
                        })

            annotated_frame = results[0].plot() if results else frame

            detection_result = {
                'detections': detections,
                'count': len(detections),
                'threat_summary': self._analyze_threats(detections),
                'timestamp': time.time(),
                'using_yolo': True
            }

            self.detections.append(detection_result)
            return detection_result, annotated_frame

        except Exception as e:
            print(f"YOLO error: {e}")
            return self._simulate_detection(frame), frame

    def _simulate_detection(self, frame):
        h, w = (480, 640)
        if frame is not None:
            h, w = frame.shape[:2]

        detections = []
        num_objects = random.randint(0, 6)
        objects = ['person', 'car', 'truck', 'motorcycle', 'bicycle', 'backpack']

        for _ in range(num_objects):
            obj = random.choice(objects)
            threat = self.threat_mapping.get(obj, 3)
            x1 = random.randint(0, w - 100)
            y1 = random.randint(0, h - 100)
            detections.append({
                'class': obj,
                'confidence': round(random.uniform(0.6, 0.95), 3),
                'bbox': [x1, y1, min(x1 + random.randint(30, 120), w), min(y1 + random.randint(30, 120), h)],
                'center': (random.randint(0, w), random.randint(0, h)),
                'threat_level': threat,
                'is_hostile': threat >= 7,
                'simulated': True
            })

        result = {
            'detections': detections,
            'count': len(detections),
            'threat_summary': self._analyze_threats(detections),
            'timestamp': time.time(),
            'using_yolo': False,
            'simulated': True
        }
        return result, frame

    def _analyze_threats(self, detections):
        if not detections:
            return {'total_threat': 0, 'high_threats': 0, 'max_threat': 0, 'critical_alerts': []}

        threats = [d['threat_level'] for d in detections]
        high_threats = [d for d in detections if d['threat_level'] >= 7]

        return {
            'total_threat': sum(threats),
            'high_threats': len(high_threats),
            'max_threat': max(threats) if threats else 0,
            'avg_threat': round(sum(threats) / len(threats), 2) if threats else 0,
            'critical_alerts': [d['class'] for d in high_threats]
        }

    def get_high_threat_detections(self):
        if self.detections:
            latest = self.detections[-1]
            return [d for d in latest.get('detections', []) if d.get('threat_level', 0) >= 7]
        return []


# =====================================================
# 🎯 نظام الأهداف العسكرية المتقدم
# =====================================================

class MilitaryTargetingSystem:
    def __init__(self):
        self.targets = {}
        self.threat_matrix = {}
        self.engagement_zones = []
        self.no_fly_zones = []
        self.target_id_counter = 0

    def classify_target(self, object_data: Dict) -> Dict:
        classification = {
            'type': 'unknown',
            'threat_level': 0,
            'priority': 'low',
            'engagement_authorized': False,
            'target_id': self.target_id_counter,
            'timestamp': time.time()
        }

        obj_type = object_data.get('type', '')
        movement = object_data.get('velocity', (0, 0))
        speed = math.hypot(movement[0], movement[1])

        if 'weapon' in str(obj_type).lower() or 'rifle' in str(obj_type).lower() or 'gun' in str(obj_type).lower():
            classification.update({
                'type': 'weapon_carrier',
                'threat_level': 9,
                'priority': 'critical',
                'engagement_authorized': True
            })
        elif 'vehicle' in str(obj_type).lower() and speed > 20:
            classification.update({
                'type': 'fast_moving_vehicle',
                'threat_level': 6,
                'priority': 'medium',
                'engagement_authorized': False
            })
        elif 'person' in str(obj_type).lower() and object_data.get('count', 0) > 3:
            classification.update({
                'type': 'suspicious_group',
                'threat_level': 7,
                'priority': 'high',
                'engagement_authorized': False
            })
        elif 'drone' in str(obj_type).lower():
            classification.update({
                'type': 'hostile_drone',
                'threat_level': 9,
                'priority': 'critical',
                'engagement_authorized': True
            })
        elif 'car' in str(obj_type).lower() or 'truck' in str(obj_type).lower():
            classification.update({
                'type': 'vehicle',
                'threat_level': 5,
                'priority': 'medium',
                'engagement_authorized': False
            })

        self.targets[classification['target_id']] = classification
        self.target_id_counter += 1
        return classification

    def calculate_interception_point(self, target_pos, target_vel, drone_pos, drone_speed):
        rx = target_pos[0] - drone_pos[0]
        ry = target_pos[1] - drone_pos[1]
        vx = target_vel[0]
        vy = target_vel[1]

        a = vx ** 2 + vy ** 2 - drone_speed ** 2
        b = 2 * (rx * vx + ry * vy)
        c = rx ** 2 + ry ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant >= 0:
            t = (-b - discriminant ** 0.5) / (2 * a) if a != 0 else 0
            if t > 0:
                intercept_point = (
                    target_pos[0] + target_vel[0] * t,
                    target_pos[1] + target_vel[1] * t
                )
                return intercept_point, t
        return None, None

    def add_no_fly_zone(self, center, radius, reason="military_zone"):
        zone = {
            'center': center,
            'radius': radius,
            'reason': reason,
            'active': True,
            'created_at': time.time()
        }
        self.no_fly_zones.append(zone)
        return zone

    def check_no_fly_violation(self, position):
        for zone in self.no_fly_zones:
            if not zone['active']:
                continue
            distance = math.sqrt(
                (position[0] - zone['center'][0]) ** 2 +
                (position[1] - zone['center'][1]) ** 2
            )
            if distance < zone['radius']:
                return zone
        return None

    def get_high_priority_targets(self):
        return [t for t in self.targets.values() if t['priority'] in ['high', 'critical']]

    def update_threat_matrix(self, position, threat_level):
        key = f"{int(position[0] / 10)},{int(position[1] / 10)}"
        self.threat_matrix[key] = {
            'position': position,
            'threat_level': threat_level,
            'last_update': time.time()
        }


# =====================================================
# 📡 نظام التجسس الإلكتروني (SIGINT)
# =====================================================

class SIGINTSystem:
    def __init__(self):
        self.detected_signals = []
        self.rf_frequencies = {
            'gsm': [900, 1800, 1900],
            'wifi': [2400, 5200],
            'military': [225, 400, 1600],
            'satellite': [1200, 1500, 2200],
            'radar': [3000, 5000, 10000]
        }
        self.intercepted_comms = []
        self.signal_history = deque(maxlen=1000)

    def scan_frequencies(self, start_freq, end_freq):
        detected = []
        for freq in range(start_freq, end_freq, 50):
            signal_strength = random.uniform(0, 100)
            if signal_strength > 65:
                signal_type = self.classify_signal(freq)
                encryption = random.choice(['none', 'wep', 'wpa2', 'military_grade'])
                detected.append({
                    'frequency': freq,
                    'strength': signal_strength,
                    'type': signal_type,
                    'encrypted': encryption != 'none',
                    'encryption_type': encryption,
                    'timestamp': time.time()
                })
                self.signal_history.append(detected[-1])
        return detected

    def classify_signal(self, frequency):
        for freq_type, freqs in self.rf_frequencies.items():
            if any(abs(frequency - f) < 50 for f in freqs):
                return freq_type
        return 'unknown_signal'

    def triangulate_position(self, signal_data):
        if len(signal_data) >= 3:
            positions = [s['position'] for s in signal_data[:3]]
            strengths = [s['strength'] for s in signal_data[:3]]
            total_strength = sum(strengths)
            if total_strength > 0:
                weighted_x = sum(p[0] * s for p, s in zip(positions, strengths)) / total_strength
                weighted_y = sum(p[1] * s for p, s in zip(positions, strengths)) / total_strength
                return (weighted_x, weighted_y)
        return None

    def detect_jamming(self):
        noise_level = random.uniform(0, 100)
        if noise_level > 85:
            return {
                'jamming_detected': True,
                'intensity': noise_level,
                'type': 'broadband_jamming',
                'affected_frequencies': random.sample([900, 1800, 2400, 5000], 3),
                'recommendation': 'change_frequency_or_use_fhss'
            }
        return {'jamming_detected': False}

    def intercept_communication(self, frequency):
        if frequency in self.rf_frequencies['military']:
            messages = [
                "نحن في الموقع المحدد، ننتظر التعليمات",
                "تحرك الوحدة إلى الإحداثيات الجديدة",
                "تأكيد استلام الأوامر، جاهزون للتنفيذ",
                "تم رصد طائرة مسيرة في المنطقة"
            ]
            return {
                'intercepted': True,
                'frequency': frequency,
                'message': random.choice(messages),
                'signal_strength': random.uniform(70, 100),
                'intelligence_value': 'high'
            }
        return {'intercepted': False}


# =====================================================
# 🌙 نظام الرؤية الليلية والاستشعار الحراري
# =====================================================

class NightVisionSystem:
    def __init__(self):
        self.thermal_calibration = 20.0
        self.night_mode = False
        self.thermal_sensitivity = 0.05

    def process_thermal_image(self, image_data=None):
        thermal_image = np.random.rand(480, 640) * 100
        hot_spots = []
        for y in range(0, 480, 20):
            for x in range(0, 640, 20):
                temp = thermal_image[y, x]
                if temp > 36:
                    obj_type = 'human' if 36 < temp < 39 else 'vehicle' if temp > 50 else 'unknown'
                    hot_spots.append({
                        'position': (x, y),
                        'temperature': round(float(temp), 2),
                        'type': obj_type,
                        'confidence': min(1.0, temp / 100)
                    })

        return {
            'thermal_image': thermal_image.tolist() if image_data else None,
            'hot_spots': hot_spots,
            'max_temperature': round(float(np.max(thermal_image)), 2),
            'min_temperature': round(float(np.min(thermal_image)), 2),
            'human_count': len([h for h in hot_spots if h['type'] == 'human']),
            'vehicle_count': len([h for h in hot_spots if h['type'] == 'vehicle'])
        }

    def detect_hidden_objects(self, thermal_data):
        hidden_objects = []
        for spot in thermal_data.get('hot_spots', []):
            if spot['temperature'] > 37 and spot['type'] == 'human':
                hidden_objects.append({
                    'position': spot['position'],
                    'temperature': spot['temperature'],
                    'likely_hiding': True,
                    'recommendation': 'investigate_area'
                })
        return hidden_objects


# =====================================================
# 🚀 نظام التسليح والذخائر الذكية
# =====================================================

class WeaponSystem:
    def __init__(self):
        self.ammunition = {
            'guided_missile': 2,
            'precision_bomb': 4,
            'camera_munition': 10,
            'flash_grenade': 6,
            'electromagnetic_pulse': 1,
            'smoke_screen': 3
        }
        self.locked_targets = []
        self.engagement_history = []
        self.authorization_level = 1

    def lock_on_target(self, target_id, target_data):
        lock_quality = self.calculate_lock_quality(target_data)
        if lock_quality > 0.7:
            lock = {
                'target_id': target_id,
                'lock_quality': lock_quality,
                'timestamp': time.time(),
                'position': target_data.get('position'),
                'velocity': target_data.get('velocity', (0, 0)),
                'tracking_time': 0
            }
            self.locked_targets.append(lock)
            return lock
        return None

    def calculate_lock_quality(self, target_data):
        factors = []
        speed = math.hypot(*target_data.get('velocity', (0, 0)))
        if speed < 10:
            factors.append(0.9)
        elif speed < 20:
            factors.append(0.7)
        else:
            factors.append(0.4)
        size = target_data.get('size', 1)
        factors.append(min(1.0, size / 10))
        camouflage = target_data.get('camouflage', 0)
        factors.append(1.0 - camouflage)
        return np.mean(factors)

    def authorize_engagement(self, authorization_code):
        valid_codes = ['ALPHA-1', 'BRAVO-2', 'CHARLIE-3', 'COMMANDER']
        return authorization_code in valid_codes

    def fire_weapon(self, weapon_type, target_lock, authorization_code):
        if not self.authorize_engagement(authorization_code):
            return {'success': False, 'reason': 'unauthorized'}
        if self.ammunition.get(weapon_type, 0) <= 0:
            return {'success': False, 'reason': 'out_of_ammunition'}
        trajectory = self.calculate_trajectory(target_lock['position'], target_lock['velocity'])
        self.ammunition[weapon_type] -= 1
        engagement = {
            'weapon': weapon_type,
            'target': target_lock['target_id'],
            'time': time.time(),
            'result': 'engaged',
            'authorization': authorization_code
        }
        self.engagement_history.append(engagement)
        impact_time = time.time() + random.uniform(3, 8)
        return {
            'success': True,
            'trajectory': trajectory,
            'estimated_impact': impact_time,
            'ammunition_remaining': self.ammunition,
            'engagement_id': len(self.engagement_history)
        }

    def calculate_trajectory(self, target_pos, target_vel):
        points = []
        for t in range(0, 50, 5):
            x = target_pos[0] + target_vel[0] * t
            y = target_pos[1] + target_vel[1] * t
            points.append((x, y))
        return points

    def get_ammunition_status(self):
        return {
            'total_weapons': sum(self.ammunition.values()),
            'weapons_detail': self.ammunition,
            'ready_to_engage': len(self.locked_targets) > 0
        }


# =====================================================
# 🔐 نظام الاتصالات الآمنة والتشفير العسكري
# =====================================================

class SecureMilitaryComms:
    def __init__(self):
        self.encryption_key = self.generate_quantum_key()
        self.comm_channels = {}
        self.auth_tokens = {}
        self.message_history = deque(maxlen=500)

    def generate_quantum_key(self, length=256):
        return hashlib.sha512(str(random.getrandbits(length)).encode()).hexdigest()

    def encrypt_message(self, message, recipient_id):
        layer1 = base64.b64encode(message.encode()).decode()
        layer2 = ''.join(chr(ord(c) ^ 0x55) for c in layer1)
        layer3 = hashlib.sha256(layer2.encode()).hexdigest() + ":" + layer2
        encrypted_msg = {
            'encrypted': layer3,
            'timestamp': time.time(),
            'sender': self.comm_channels.get('my_id', 'unknown'),
            'recipient': recipient_id,
            'signature': hashlib.md5(layer3.encode()).hexdigest()
        }
        self.message_history.append(encrypted_msg)
        return encrypted_msg

    def decrypt_message(self, encrypted_data):
        try:
            hash_part, data_part = encrypted_data['encrypted'].split(':', 1)
            layer2 = ''.join(chr(ord(c) ^ 0x55) for c in data_part)
            decrypted = base64.b64decode(layer2).decode()
            return decrypted
        except:
            return "فشل فك التشفير"

    def establish_secure_channel(self, unit_id, auth_code):
        if self.verify_authorization(auth_code):
            channel_key = self.generate_quantum_key()
            self.comm_channels[unit_id] = {
                'key': channel_key,
                'established': time.time(),
                'last_heartbeat': time.time(),
                'status': 'active'
            }
            return True
        return False

    def verify_authorization(self, auth_code):
        valid_auth = ['SECURE-2026', 'MIL-COMM', 'INTEL-SECURE']
        return auth_code in valid_auth

    def frequency_hopping(self):
        frequencies = [225, 230, 235, 240, 245, 250, 255, 260]
        hop_sequence = random.sample(frequencies, len(frequencies))
        return {
            'sequence': hop_sequence,
            'hop_interval': 0.5,
            'sync_time': time.time(),
            'current_channel': hop_sequence[0]
        }

    def send_emergency_broadcast(self, message, emergency_code):
        if emergency_code == 'MAYDAY-MIL-001':
            return {
                'type': 'EMERGENCY',
                'message': message,
                'timestamp': time.time(),
                'requires_ack': True,
                'priority': 'critical'
            }
        return None


# =====================================================
# 🗺️ نظام الخرائط الاستخباراتي 3D
# =====================================================

class IntelligenceMapping3D:
    def __init__(self):
        self.intel_layers = {}
        self.hot_zones = []
        self.asset_positions = {}
        self.predictions = []

    def create_threat_heatmap(self, threat_data):
        heatmap = np.zeros((100, 100))
        for threat in threat_data:
            x, y = threat.get('position', (50, 50))
            intensity = threat.get('threat_level', 5)
            for i in range(-10, 11):
                for j in range(-10, 11):
                    nx, ny = int(x + i), int(y + j)
                    if 0 <= nx < 100 and 0 <= ny < 100:
                        distance = math.sqrt(i ** 2 + j ** 2)
                        contribution = intensity * np.exp(-distance ** 2 / 50)
                        heatmap[nx, ny] += contribution
        return {
            'heatmap': heatmap.tolist(),
            'max_threat': float(np.max(heatmap)),
            'avg_threat': float(np.mean(heatmap))
        }

    def predict_movement_patterns(self, historical_data):
        predictions = []
        for track in historical_data:
            positions = track.get('positions', [])
            if len(positions) >= 3:
                x_coords = [p[0] for p in positions[-5:]]
                y_coords = [p[1] for p in positions[-5:]]
                if len(x_coords) >= 2:
                    vx = (x_coords[-1] - x_coords[-2]) if len(x_coords) > 1 else 0
                    vy = (y_coords[-1] - y_coords[-2]) if len(y_coords) > 1 else 0
                    for step in range(1, 6):
                        predictions.append({
                            'target_id': track.get('id', 'unknown'),
                            'predicted_position': (x_coords[-1] + vx * step, y_coords[-1] + vy * step),
                            'time': step * 5,
                            'confidence': max(0.1, 0.9 - (step * 0.1))
                        })
        self.predictions = predictions
        return predictions

    def add_intel_layer(self, layer_name, data):
        self.intel_layers[layer_name] = {
            'data': data,
            'timestamp': time.time(),
            'active': True
        }

    def get_strategic_recommendations(self):
        recommendations = []
        if self.hot_zones:
            recommendations.append({
                'type': 'deploy_assets',
                'target_zones': self.hot_zones[:3],
                'priority': 'high'
            })
        if self.predictions:
            high_confidence = [p for p in self.predictions if p['confidence'] > 0.7]
            if high_confidence:
                recommendations.append({
                    'type': 'intercept_predicted',
                    'targets': high_confidence[:2],
                    'priority': 'medium'
                })
        return recommendations


# =====================================================
# 🤖 نظام القيادة والسيطرة بالذكاء الاصطناعي
# =====================================================

class AIBattleCommand:
    def __init__(self):
        self.battlefield_state = {}
        self.recommended_actions = []
        self.casualty_threshold = 0.3
        self.tactical_decisions = []

    def analyze_battlefield(self, intel_data):
        enemy_count = 0
        total_threat = 0
        for detection in intel_data.get('detections', []):
            if detection.get('is_hostile', False) or detection.get('threat_level', 0) >= 7:
                enemy_count += 1
                total_threat += detection.get('threat_level', 0)
        threat_average = total_threat / enemy_count if enemy_count > 0 else 0

        if threat_average > 7:
            strategy = 'defensive_posture'
            risk = 'critical'
        elif threat_average > 4:
            strategy = 'cautious_engagement'
            risk = 'high'
        elif enemy_count > 5:
            strategy = 'call_for_reinforcements'
            risk = 'medium'
        else:
            strategy = 'proactive_surveillance'
            risk = 'low'

        analysis = {
            'enemy_count': enemy_count,
            'threat_average': round(threat_average, 2),
            'friendly_assets': len(intel_data.get('friendly_forces', [])),
            'recommended_strategy': strategy,
            'risk_level': risk,
            'timestamp': time.time()
        }
        self.battlefield_state = analysis
        return analysis

    def generate_tactical_options(self, battlefield_analysis):
        options = [
            {
                'type': 'offensive',
                'name': 'ضربة استباقية',
                'success_probability': 0.65,
                'risk': 0.8,
                'resource_cost': 0.7,
                'description': 'مهاجمة الأهداف المعادية قبل تمكنها من التنظيم'
            },
            {
                'type': 'defensive',
                'name': 'تمركز دفاعي',
                'success_probability': 0.85,
                'risk': 0.3,
                'resource_cost': 0.4,
                'description': 'تعزيز المواقع الدفاعية والانتظار'
            },
            {
                'type': 'reconnaissance',
                'name': 'تعزيز الاستطلاع',
                'success_probability': 0.9,
                'risk': 0.2,
                'resource_cost': 0.3,
                'description': 'جمع المزيد من المعلومات الاستخباراتية'
            }
        ]

        if battlefield_analysis.get('enemy_count', 0) > 10:
            options.append({
                'type': 'strategic',
                'name': 'استدعاء تعزيزات',
                'success_probability': 0.95,
                'risk': 0.1,
                'resource_cost': 0.5,
                'description': 'طلب دعم إضافي من القيادة'
            })

        options.sort(key=lambda x: x['success_probability'] / max(x['risk'], 0.1), reverse=True)
        self.recommended_actions = options
        return options

    def make_tactical_decision(self, battlefield_analysis):
        options = self.generate_tactical_options(battlefield_analysis)
        if options:
            best_option = options[0]
            decision = {
                'selected_strategy': best_option['name'],
                'confidence': best_option['success_probability'],
                'execution_time': time.time(),
                'expected_outcome': 'مواتي' if best_option['success_probability'] > 0.7 else 'متوسط'
            }
            self.tactical_decisions.append(decision)
            return decision
        return {'selected_strategy': 'مراقبة', 'confidence': 0.5}

    def calculate_force_ratio(self, friendly_count, enemy_count):
        if enemy_count == 0:
            return {'ratio': float('inf'), 'advantage': 'overwhelming'}
        ratio = friendly_count / enemy_count
        if ratio >= 3:
            advantage = 'overwhelming'
        elif ratio >= 1.5:
            advantage = 'significant'
        elif ratio >= 0.8:
            advantage = 'balanced'
        elif ratio >= 0.5:
            advantage = 'disadvantage'
        else:
            advantage = 'critical'
        return {'ratio': round(ratio, 2), 'advantage': advantage}


# =====================================================
# 📸 فئات الكشف والتحليل الأساسية
# =====================================================

class FaceAndPersonDetector:
    def __init__(self):
        self.face_cascade = None
        self.person_cascade = None
        self.use_real_detection = False
        try:
            cascade_path = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
            self.person_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_fullbody.xml'))
            self.use_real_detection = True
            print("✅ تم تحميل نماذج الكشف الحقيقية (OpenCV)")
        except Exception as e:
            self.use_real_detection = False
            print(f"⚠️ وضع المحاكاة للكشف: {e}")
        self.detection_history = deque(maxlen=100)
        self.total_people_detected = 0
        self.total_faces_detected = 0

    def detect_from_image(self, image_data=None):
        results = {
            'faces': [], 'people': [], 'face_count': 0, 'person_count': 0,
            'timestamp': time.time(), 'image_base64': None
        }
        if self.use_real_detection and image_data is not None:
            try:
                if isinstance(image_data, str) and image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if self.face_cascade is not None:
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                        for (x, y, w, h) in faces:
                            results['faces'].append({
                                'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                                'confidence': random.uniform(0.85, 0.99)
                            })
                    if self.person_cascade is not None:
                        people = self.person_cascade.detectMultiScale(gray, 1.05, 3)
                        for (x, y, w, h) in people:
                            is_duplicate = any(
                                abs(x - fx) < 50 and abs(y - fy) < 50
                                for (fx, fy, fw, fh) in [(f['x'], f['y'], f['width'], f['height']) for f in
                                                         results['faces']]
                            )
                            if not is_duplicate:
                                results['people'].append({
                                    'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                                    'confidence': random.uniform(0.75, 0.95)
                                })
                    _, buffer = cv2.imencode('.jpg', img)
                    results['image_base64'] = base64.b64encode(buffer).decode('utf-8')
                    results['face_count'] = len(results['faces'])
                    results['person_count'] = len(results['people'])
                    self.total_faces_detected += results['face_count']
                    self.total_people_detected += results['person_count']
                    self.detection_history.append(results)
            except Exception as e:
                print(f"خطأ في المعالجة: {e}")
                results = self._simulate_detection()
        else:
            results = self._simulate_detection()
        return results

    def _simulate_detection(self):
        base_people = random.randint(0, 8)
        base_faces = random.randint(0, min(base_people + 2, 6))
        people = []
        faces = []
        for i in range(base_people):
            people.append({
                'x': random.randint(50, 600), 'y': random.randint(50, 400),
                'width': random.randint(30, 80), 'height': random.randint(80, 160),
                'confidence': round(random.uniform(0.7, 0.98), 2)
            })
        for i in range(min(base_faces, base_people)):
            if i < len(people):
                faces.append({
                    'x': people[i]['x'] + random.randint(10, 30),
                    'y': people[i]['y'] + random.randint(10, 40),
                    'width': random.randint(20, 40), 'height': random.randint(20, 40),
                    'confidence': round(random.uniform(0.8, 0.99), 2)
                })
        self.total_people_detected += len(people)
        self.total_faces_detected += len(faces)
        return {
            'faces': faces, 'people': people, 'face_count': len(faces),
            'person_count': len(people), 'timestamp': time.time(),
            'image_base64': None, 'simulated': True
        }

    def get_statistics(self):
        avg_people = np.mean([d['person_count'] for d in self.detection_history]) if self.detection_history else 0
        return {
            'total_people_detected': self.total_people_detected,
            'total_faces_detected': self.total_faces_detected,
            'detection_rate': len(self.detection_history),
            'average_people_per_scan': round(float(avg_people), 2),
            'real_detection_mode': self.use_real_detection
        }


# =====================================================
# 🧠 Gemini AI
# =====================================================

class GeminiUltra:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available = GEMINI_AVAILABLE
        self.model = GEMINI_MODEL
        self.chat_history = []

    def analyze_image(self, image_base64: str, context: str = "") -> Dict:
        if not self.available:
            return self._smart_simulation()
        try:
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            img_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_bytes))
            prompt = f"""أنت خبير استخباراتي عسكري محترف. حلل هذه الصورة بدقة عالية وأجب بالعربية.
المطلوب:
1. عدد الأشخاص
2. عدد الوجوه المكشوفة
3. هل هناك أسلحة أو أشياء مشبوهة؟
4. وصف الملابس والمظهر العام
5. مستوى التهديد من 1 إلى 10
6. ملخص استخباراتي
7. توصيات عاجلة
سياق: {context}
أجب بتنسيق JSON فقط: {{"people_count": عدد, "faces_count": عدد, "weapons_detected": [], "suspicious_items": [], "clothing_description": "", "threat_level": 1-10, "intelligence_summary": "", "recommendations": [], "priority": "عاجل/مرتفع/متوسط/منخفض"}}"""
            response = self.model.generate_content([prompt, img])
            text = response.text
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                result = json.loads(json_match.group())
                result['using_gemini'] = True
                return result
            return self._smart_simulation()
        except Exception as e:
            print(f"Gemini Error: {e}")
            return self._smart_simulation()

    def chat(self, message: str) -> str:
        if not self.available:
            return self._smart_chat(message)
        try:
            self.chat_history.append({"role": "user", "content": message})
            prompt = f"""أنت مساعد استخباراتي ذكي ومحترف. أجب بالعربية.
المحادثة السابقة: {json.dumps(self.chat_history[-5:], ensure_ascii=False)}
السؤال: {message}
الإجابة المفصلة:"""
            response = self.model.generate_content(prompt)
            reply = response.text
            self.chat_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"⚠️ خطأ في الاتصال بـ Gemini: {str(e)}"

    def search_web(self, query: str) -> Dict:
        if not self.available:
            return {"query": query, "summary": "Gemini غير متصل حالياً", "key_findings": [], "reliability": "منخفض"}
        try:
            prompt = f"""ابحث وحلل المعلومات حول: {query}. قدم تقريراً استخباراتياً بالعربية.
أجب بـ JSON: {{"query": "{query}", "summary": "ملخص", "key_findings": [], "reliability": "عالي/متوسط/منخفض", "sources": []}}"""
            response = self.model.generate_content(prompt)
            text = response.text
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
            return {"query": query, "summary": text[:300], "key_findings": [], "reliability": "متوسط"}
        except Exception as e:
            return {"query": query, "summary": f"خطأ: {str(e)}", "key_findings": [], "reliability": "منخفض"}

    def generate_scene(self, scene_type: str, details: str = "") -> str:
        if not self.available:
            return self._generate_fake_scene(scene_type)
        try:
            prompt = f"""أنشئ تقريراً استخباراتياً مفصلاً عن "{scene_type}" بالعربية.
{details}
التقرير يشمل: الملخص التنفيذي، التحليل المفصل، التوصيات، مستوى الثقة"""
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"خطأ: {str(e)}"

    def _smart_simulation(self) -> Dict:
        hour = datetime.now().hour
        is_night = hour < 6 or hour > 18
        people = random.randint(0, 12)
        faces = random.randint(0, min(people, 8))
        threat_base = 3 if is_night else 2
        threat = min(10, threat_base + random.randint(0, 5))
        return {
            "people_count": people,
            "faces_count": faces,
            "weapons_detected": [],
            "suspicious_items": [],
            "clothing_description": "ملابس داكنة" if is_night else "ملابس متنوعة",
            "threat_level": threat,
            "intelligence_summary": f"تم كشف {people} شخص، {faces} وجه. مستوى التهديد {threat}/10.",
            "recommendations": ["تكثيف المراقبة" if threat > 6 else "استمرار المراقبة العادية"],
            "priority": "عاجل" if threat > 7 else "مرتفع" if threat > 5 else "متوسط",
            "using_gemini": False,
            "simulated": True
        }

    def _smart_chat(self, message: str) -> str:
        return f"""📡 نظام Gemini غير متصل حالياً (وضع المحاكاة).
💡 يرجى التحقق من مفتاح API والاتصال بالإنترنت.
❓ سؤالك: {message[:100]}..."""

    def _generate_fake_scene(self, scene_type: str) -> str:
        return f"""🏆 تقرير استخباراتي عن: {scene_type}
📝 هذا تقرير افتراضي (Gemini غير متصل). يرجى تشغيل Gemini API للحصول على تحليلات حقيقية."""


# =====================================================
# 🚁 الدرون الأسطوري الرئيسي
# =====================================================

class LegendaryDrone:
    def __init__(self, drone_id: str = "LEGEND-DRONE-01"):
        self.drone_id = drone_id
        self.authenticated = False
        self.operator_id = None
        self.connected = False
        self.armed = False
        self.mode = "يدوي"
        self.x = 0.0
        self.y = 0.0
        self.altitude = 0.0
        self.target_alt = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.battery = 100.0
        self.signal_strength = 100
        self.last_heartbeat = time.time()
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_set = False
        self.path = deque(maxlen=500)
        self.rth_path = []
        self.waypoints = []
        self.current_waypoint_index = 0
        self.surveillance_mode = False
        self.sensors = {"front": 10, "back": 10, "left": 10, "right": 10, "down": 10}
        self.patrol_center = None
        self.patrol_radius = 0
        self.patrol_angle = 0

        # الأنظمة المتقدمة
        self.gemini = GeminiUltra(GEMINI_API_KEY)
        self.face_detector = FaceAndPersonDetector()
        self.targeting_system = MilitaryTargetingSystem()
        self.sigint_system = SIGINTSystem()
        self.night_vision = NightVisionSystem()
        self.weapon_system = WeaponSystem()
        self.secure_comms = SecureMilitaryComms()
        self.intel_mapping = IntelligenceMapping3D()
        self.battle_command = AIBattleCommand()

        # أنظمة جديدة
        self.mobile_camera = MobileCameraSystem()
        self.ai_detection = AIDetectionSystem()

        self.detected_objects = []
        self.last_analysis = None
        self.total_people_detected = 0
        self.total_faces_detected = 0
        self.thermal_data = None
        self.mobile_streaming = False
        self.current_camera_detections = []
        self.last_annotated_frame = None
        self.last_capture_time = 0
        self.last_capture_result = None

        self.logs = deque(maxlen=200)
        self.log(f"[{self.drone_id}] 🦅 نظام درون أسطوري - Gemini {'متصل' if GEMINI_AVAILABLE else 'محاكاة'}", "legend")
        if YOLO_AVAILABLE:
            self.log(f"[{self.drone_id}] ✅ YOLOv8 نشط", "legend")
        else:
            self.log(f"[{self.drone_id}] ⚠️ YOLOv8 غير متاح - وضع المحاكاة", "warning")

        self.last_update = time.time()
        threading.Thread(target=self.simulation_loop, daemon=True).start()

    def log(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "alert": "🚨", "intel": "🕵️", "legend": "🏆"}.get(level, "📝")
        entry = f"[{timestamp}] {emoji} {message}"
        self.logs.append(entry)

    def authenticate(self, operator_id: str, credentials: Dict) -> bool:
        if credentials.get("token") == "EDU-TOKEN-2026":
            self.authenticated = True
            self.operator_id = operator_id
            self.log(f"✅ مصادقة ناجحة: {operator_id}", "legend")
            self.secure_comms.comm_channels['my_id'] = operator_id
            return True
        self.log("❌ فشل المصادقة", "error")
        return False

    def connect(self) -> bool:
        if not self.authenticated:
            self.log("⚠️ مطلوب مصادقة أولاً", "warning")
            return False
        self.connected = True
        self.log("🔌 تم الاتصال - النظام جاهز", "legend")
        return True

    def arm(self) -> bool:
        if not self.connected:
            return False
        self.armed = True
        self.target_alt = 6
        self.log("▶️ إقلاع - النظام مسلح", "legend")
        return True

    def land(self):
        self.target_alt = 0
        self.vx = self.vy = 0
        self.mode = "يدوي"
        self.armed = False
        self.log("🛬 هبوط", "info")

    def start_rth(self) -> bool:
        if self.home_set:
            self.mode = "عودة"
            self.log("🏠 تفعيل العودة للمنزل", "info")
            return True
        self.log("⚠️ لم يتم تعيين نقطة المنزل", "warning")
        return False

    def set_home(self, x: float, y: float):
        self.home_x, self.home_y = x, y
        self.home_set = True
        self.log(f"📍 تعيين المنزل: ({x:.1f}, {y:.1f})", "info")

    def add_waypoint(self, x: float, y: float, altitude: float = None):
        self.waypoints.append({'x': x, 'y': y, 'alt': altitude or self.target_alt})
        if len(self.waypoints) == 1:
            self.mode = "آلي"
        self.log(f"📍 إضافة نقطة مسار: ({x:.1f}, {y:.1f})", "info")

    def start_patrol(self, center_x: float, center_y: float, radius: float):
        self.patrol_center = (center_x, center_y)
        self.patrol_radius = radius
        self.patrol_angle = 0
        self.mode = "دورية"
        self.log(f"🔄 بدء دورية: المركز ({center_x:.1f}, {center_y:.1f})، نصف القطر {radius:.1f}م", "info")

    def move(self, direction: str, intensity: float = 1.0):
        if not self.armed or self.mode != "يدوي":
            return
        a = 2.5 * intensity
        if direction == "forward":
            self.vy += a
        elif direction == "back":
            self.vy -= a
        elif direction == "right":
            self.vx += a
        elif direction == "left":
            self.vx -= a
        elif direction == "stop":
            self.vx *= 0.3
            self.vy *= 0.3

    # ========== أنظمة كاميرا الجوال و YOLO ==========

    def connect_mobile_camera(self, url: str) -> Dict:
        self.log(f"📱 محاولة الاتصال بالكاميرا: {url}", "intel")
        success = self.mobile_camera.set_camera_url(url)
        if success:
            def detection_callback(frame):
                if frame is not None:
                    detections, annotated = self.ai_detection.detect_objects(frame)
                    self.current_camera_detections = detections.get('detections', [])
                    _, buffer = cv2.imencode('.jpg', annotated)
                    self.last_annotated_frame = base64.b64encode(buffer).decode('utf-8')
                    for det in self.current_camera_detections:
                        if det.get('threat_level', 0) >= 7:
                            self.log(f"🚨 {det['class']} - تهديد عالي!", "alert")
                            self.targeting_system.classify_target({
                                'type': det['class'], 'count': 1, 'velocity': (0, 0)
                            })

            self.mobile_camera.start_streaming(detection_callback)
            self.mobile_streaming = True
            self.log("✅ تم بدء البث المباشر من كاميرا الجوال", "legend")
            return {'success': True, 'message': 'تم الاتصال وبدء البث', 'yolo_active': YOLO_AVAILABLE}
        else:
            return {'success': False, 'message': 'فشل الاتصال بالكاميرا'}

    def disconnect_mobile_camera(self) -> Dict:
        if self.mobile_camera:
            self.mobile_camera.stop_streaming()
            self.mobile_streaming = False
            self.log("📱 تم قطع الاتصال بكاميرا الجوال", "info")
        return {'success': True}

    def get_camera_frame(self) -> Dict:
        if self.mobile_camera and self.mobile_camera.current_frame is not None:
            frame = self.mobile_camera.current_frame
            frame_b64 = self.mobile_camera.frame_to_base64(frame)
            return {
                'success': True,
                'frame': frame_b64,
                'detections': self.current_camera_detections,
                'detection_count': len(self.current_camera_detections),
                'high_threats': len([d for d in self.current_camera_detections if d.get('threat_level', 0) >= 7]),
                'annotated_frame': self.last_annotated_frame
            }
        return {'success': False, 'message': 'الكاميرا غير متصلة'}

    def capture_moment(self) -> Dict:
        result = self.get_camera_frame()
        if result['success'] and result.get('frame'):
            gemini_result = self.analyze_with_gemini(result['frame'])
            result['gemini_analysis'] = gemini_result
            return result
        return {'success': False, 'message': 'لا يمكن التقاط الصورة'}

    # ========== الأنظمة العسكرية المتقدمة ==========

    def lock_target(self, target_data: Dict) -> Dict:
        classification = self.targeting_system.classify_target(target_data)
        if classification['engagement_authorized']:
            lock = self.weapon_system.lock_on_target(classification['target_id'], target_data)
            self.log(f"🎯 تثبيت هدف: {classification['type']} (تهديد {classification['threat_level']})", "alert")
            return {'success': True, 'classification': classification, 'lock': lock}
        return {'success': False, 'classification': classification}

    def fire_weapon_at_target(self, weapon_type: str, target_id: int, auth_code: str) -> Dict:
        target_lock = None
        for lock in self.weapon_system.locked_targets:
            if lock['target_id'] == target_id:
                target_lock = lock
                break
        if not target_lock:
            return {'success': False, 'reason': 'target_not_locked'}
        result = self.weapon_system.fire_weapon(weapon_type, target_lock, auth_code)
        if result['success']:
            self.log(f"🚀 إطلاق {weapon_type} على الهدف {target_id}", "alert")
        else:
            self.log(f"❌ فشل إطلاق {weapon_type}: {result.get('reason', 'unknown')}", "error")
        return result

    def thermal_scan(self) -> Dict:
        self.thermal_data = self.night_vision.process_thermal_image()
        self.log(f"🌡️ مسح حراري: {self.thermal_data['human_count']} شخص, {self.thermal_data['vehicle_count']} مركبة", "intel")
        return self.thermal_data

    def signal_scan(self, start_freq: int = 100, end_freq: int = 3000) -> Dict:
        signals = self.sigint_system.scan_frequencies(start_freq, end_freq)
        jamming = self.sigint_system.detect_jamming()
        self.log(f"📡 مسح ترددات: {len(signals)} إشارة", "intel")
        if jamming.get('jamming_detected'):
            self.log("⚠️ تشويش إلكتروني مكتشف!", "alert")
        return {'signals': signals, 'jamming': jamming}

    def add_no_fly_zone(self, center_x: float, center_y: float, radius: float, reason: str = "military_zone") -> Dict:
        zone = self.targeting_system.add_no_fly_zone((center_x, center_y), radius, reason)
        self.log(f"🚫 منطقة حظر طيران: {reason} (نصف قطر {radius}م)", "alert")
        return zone

    def analyze_battlefield(self) -> Dict:
        intel_data = {
            'detections': self.detected_objects + self.current_camera_detections,
            'friendly_forces': [{'type': 'drone', 'id': self.drone_id}],
            'timestamp': time.time()
        }
        analysis = self.battle_command.analyze_battlefield(intel_data)
        self.log(f"🎯 تحليل المعركة: {analysis['recommended_strategy']} - خطر {analysis['risk_level']}", "intel")
        return analysis

    def get_tactical_options(self) -> List[Dict]:
        analysis = self.analyze_battlefield()
        return self.battle_command.generate_tactical_options(analysis)

    def create_heatmap(self) -> Dict:
        threat_data = []
        for obj in self.detected_objects:
            if 'threat' in obj:
                threat_data.append({
                    'position': obj.get('position', (50, 50)),
                    'threat_level': obj.get('threat', 5)
                })
        return self.intel_mapping.create_threat_heatmap(threat_data)

    def encrypt_communication(self, message: str, recipient: str) -> Dict:
        return self.secure_comms.encrypt_message(message, recipient)

    def get_ammunition_status(self) -> Dict:
        return self.weapon_system.get_ammunition_status()

    # ========== تحليل Gemini ==========

    def analyze_with_gemini(self, image_base64: str = None, context: str = "") -> Dict:
        self.log("🕵️ تحليل استخباراتي...", "intel")
        if image_base64:
            analysis = self.gemini.analyze_image(image_base64, context)
        else:
            analysis = self.gemini._smart_simulation()
        if 'people_count' in analysis:
            self.total_people_detected += analysis['people_count']
        if 'faces_count' in analysis:
            self.total_faces_detected += analysis['faces_count']
        self.last_analysis = analysis
        self.detected_objects.append({
            'type': 'gemini_analysis',
            'people': analysis.get('people_count', 0),
            'faces': analysis.get('faces_count', 0),
            'threat': analysis.get('threat_level', 5),
            'using_gemini': analysis.get('using_gemini', False),
            'timestamp': time.time(),
            'position': (self.x, self.y)
        })
        if len(self.detected_objects) > 20:
            self.detected_objects.pop(0)
        gemini_status = "Gemini" if analysis.get('using_gemini') else "محاكاة"
        self.log(f"📊 [{gemini_status}] {analysis.get('people_count', 0)} شخص, تهديد {analysis.get('threat_level', 5)}/10", "intel")
        return analysis

    def gemini_chat(self, message: str) -> str:
        self.log(f"💬 سؤال: {message[:50]}...", "intel")
        response = self.gemini.chat(message)
        self.log(f"🤖 الرد: {response[:50]}...", "intel")
        return response

    def gemini_search(self, query: str) -> Dict:
        self.log(f"🔍 بحث استخباراتي: {query}", "intel")
        return self.gemini.search_web(query)

    def gemini_generate_scene(self, scene_type: str, details: str = "") -> str:
        return self.gemini.generate_scene(scene_type, details)

    def capture_and_analyze(self, image_data: str = None) -> Dict:
        self.log("📸 التقاط وتحليل...", "info")
        detection_result = self.face_detector.detect_from_image(image_data)
        people_count = detection_result['person_count']
        faces_count = detection_result['face_count']
        self.log(f"👥 كشف {people_count} شخص و {faces_count} وجه", "info")
        if image_data:
            gemini_analysis = self.analyze_with_gemini(image_data)
            detection_result['gemini_analysis'] = gemini_analysis
        self.last_capture_time = time.time()
        self.last_capture_result = detection_result
        return detection_result

    def get_detection_statistics(self) -> Dict:
        stats = self.face_detector.get_statistics()
        return {
            **stats,
            'gemini_active': GEMINI_AVAILABLE,
            'yolo_active': YOLO_AVAILABLE,
            'mobile_streaming': self.mobile_streaming,
            'weapons_status': self.weapon_system.get_ammunition_status(),
            'locked_targets': len(self.weapon_system.locked_targets),
            'camera_detections': len(self.current_camera_detections),
            'high_threat_detections': len([d for d in self.current_camera_detections if d.get('threat_level', 0) >= 7])
        }

    def get_telemetry(self) -> Dict:
        return {
            'drone_id': self.drone_id,
            'authenticated': self.authenticated,
            'connected': self.connected,
            'armed': self.armed,
            'mode': self.mode,
            'position': {'x': round(self.x, 2), 'y': round(self.y, 2)},
            'altitude': round(self.altitude, 2),
            'velocity': {'vx': round(self.vx, 2), 'vy': round(self.vy, 2)},
            'heading': round(self.heading, 2),
            'battery': round(self.battery, 1),
            'signal_strength': self.signal_strength,
            'home_set': self.home_set,
            'home': {'x': self.home_x, 'y': self.home_y} if self.home_set else None,
            'waypoints_count': len(self.waypoints),
            'current_waypoint': self.current_waypoint_index,
            'gemini_active': GEMINI_AVAILABLE,
            'yolo_active': YOLO_AVAILABLE,
            'mobile_streaming': self.mobile_streaming,
            'logs': list(self.logs)[-20:]
        }

    # ========== حلقة المحاكاة ==========

    def simulation_loop(self):
        while True:
            now = time.time()
            dt = min(0.1, now - self.last_update) if self.last_update else 0.05
            self.last_update = now

            if self.armed:
                px, py = self.x, self.y
                self.x += self.vx * dt
                self.y += self.vy * dt
                self.vx *= 0.94
                self.vy *= 0.94
                self.altitude += (self.target_alt - self.altitude) * 0.1
                self.altitude = max(0, min(self.altitude, 120))
                self.speed = math.hypot(self.vx, self.vy)
                if self.speed > 0.01:
                    self.heading = math.degrees(math.atan2(self.vy, self.vx))

                consumption = 0.015 + abs(self.vx + self.vy) * 0.02
                self.battery = max(0, self.battery - consumption * dt)
                self.signal_strength = max(20, 100 - len(self.path) // 15)

                if self.battery < 25 and self.mode not in ["عودة", "هبوط"]:
                    self.log("⚠️ بطارية منخفضة - بدء العودة", "warning")
                    self.start_rth()

                # التحقق من مناطق حظر الطيران
                no_fly_violation = self.targeting_system.check_no_fly_violation((self.x, self.y))
                if no_fly_violation:
                    self.log(f"🚨 انتهاك منطقة حظر طيران: {no_fly_violation['reason']}", "alert")
                    self.start_rth()

                # وضع الدورية
                if self.mode == "دورية" and self.patrol_center:
                    self.patrol_angle += 0.05
                    tx = self.patrol_center[0] + self.patrol_radius * math.cos(self.patrol_angle)
                    ty = self.patrol_center[1] + self.patrol_radius * math.sin(self.patrol_angle)
                    dx, dy = tx - self.x, ty - self.y
                    d = math.hypot(dx, dy)
                    if d > 0.5:
                        spd = min(3.0, d * 0.8)
                        self.vx += (dx / d) * spd
                        self.vy += (dy / d) * spd

                # وضع آلي (waypoints)
                elif self.mode == "آلي" and self.current_waypoint_index < len(self.waypoints):
                    wp = self.waypoints[self.current_waypoint_index]
                    dx, dy = wp['x'] - self.x, wp['y'] - self.y
                    d = math.hypot(dx, dy)
                    if d < 0.5:
                        self.current_waypoint_index += 1
                        if self.current_waypoint_index >= len(self.waypoints):
                            self.log("✅ تم الوصول لجميع نقاط المسار", "info")
                            self.mode = "يدوي"
                    else:
                        spd = min(3.0, d * 0.8)
                        self.vx += (dx / d) * spd
                        self.vy += (dy / d) * spd

                # وضع العودة للمنزل
                elif self.mode == "عودة":
                    dx, dy = self.home_x - self.x, self.home_y - self.y
                    d = math.hypot(dx, dy)
                    if d < 0.5 and self.altitude < 0.3:
                        self.land()
                        self.log("🏠 تمت العودة للمنزل بنجاح", "legend")
                    elif d < 1:
                        self.target_alt = max(0, self.target_alt - 0.1)
                        spd = min(1.5, d * 0.5)
                        self.vx += (dx / d) * spd if d > 0.1 else 0
                        self.vy += (dy / d) * spd if d > 0.1 else 0
                    else:
                        spd = min(4.0, d * 0.6)
                        self.vx += (dx / d) * spd
                        self.vy += (dy / d) * spd

                # تسجيل المسار
                if abs(self.x - px) > 0.01 or abs(self.y - py) > 0.01:
                    self.path.append((self.x, self.y))

            # تحديث أجهزة الاستشعار
            self.sensors = {
                "front": round(random.uniform(8, 10), 1),
                "back": round(random.uniform(8, 10), 1),
                "left": round(random.uniform(8, 10), 1),
                "right": round(random.uniform(8, 10), 1),
                "down": round(self.altitude, 1)
            }

            time.sleep(0.05)


# =====================================================
# 🌐 خادم HTTP للتحكم بالدرون
# =====================================================

# إنشاء نسخة عالمية من الدرون
DRONE = LegendaryDrone("LEGEND-DRONE-01")

class DroneRequestHandler(http.server.BaseHTTPRequestHandler):
    """معالج طلبات HTTP للدرون"""

    def log_message(self, format, *args):
        pass  # تعطيل سجلات HTTP الافتراضية

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _send_html(self, html, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _read_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))
        return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))

        # الصفحة الرئيسية
        if path == '/' or path == '/index.html':
            self._serve_main_page()
        # API endpoints
        elif path == '/api/telemetry':
            self._send_json(DRONE.get_telemetry())
        elif path == '/api/status':
            self._send_json(DRONE.get_detection_statistics())
        elif path == '/api/logs':
            self._send_json({'logs': list(DRONE.logs)})
        elif path == '/api/ammunition':
            self._send_json(DRONE.get_ammunition_status())
        elif path == '/api/camera_frame':
            self._send_json(DRONE.get_camera_frame())
        elif path == '/api/thermal':
            self._send_json(DRONE.thermal_scan())
        elif path == '/api/battlefield':
            self._send_json(DRONE.analyze_battlefield())
        elif path == '/api/tactical_options':
            self._send_json(DRONE.get_tactical_options())
        elif path == '/api/heatmap':
            self._send_json(DRONE.create_heatmap())
        elif path == '/api/signal_scan':
            self._send_json(DRONE.signal_scan())
        elif path == '/api/targets':
            self._send_json({
                'targets': list(DRONE.targeting_system.targets.values()),
                'high_priority': DRONE.targeting_system.get_high_priority_targets()
            })
        elif path == '/api/no_fly_zones':
            self._send_json({'zones': DRONE.targeting_system.no_fly_zones})
        elif path == '/api/weapons':
            self._send_json(DRONE.weapon_system.get_ammunition_status())
        elif path == '/api/locked_targets':
            self._send_json({'locked_targets': DRONE.weapon_system.locked_targets})
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        body = self._read_body()

        # المصادقة والاتصال
        if path == '/api/auth':
            result = DRONE.authenticate(body.get('operator_id', 'unknown'), body.get('credentials', {}))
            self._send_json({'success': result})
        elif path == '/api/connect':
            result = DRONE.connect()
            self._send_json({'success': result})
        elif path == '/api/arm':
            result = DRONE.arm()
            self._send_json({'success': result})
        elif path == '/api/land':
            DRONE.land()
            self._send_json({'success': True})
        elif path == '/api/rth':
            result = DRONE.start_rth()
            self._send_json({'success': result})

        # التحكم بالحركة
        elif path == '/api/move':
            DRONE.move(body.get('direction', 'stop'), body.get('intensity', 1.0))
            self._send_json({'success': True})
        elif path == '/api/set_home':
            DRONE.set_home(body.get('x', 0), body.get('y', 0))
            self._send_json({'success': True})
        elif path == '/api/add_waypoint':
            DRONE.add_waypoint(body.get('x', 0), body.get('y', 0), body.get('altitude'))
            self._send_json({'success': True})
        elif path == '/api/clear_waypoints':
            DRONE.waypoints = []
            DRONE.current_waypoint_index = 0
            self._send_json({'success': True})
        elif path == '/api/start_patrol':
            DRONE.start_patrol(body.get('cx', 0), body.get('cy', 0), body.get('radius', 10))
            self._send_json({'success': True})

        # كاميرا الجوال
        elif path == '/api/connect_camera':
            result = DRONE.connect_mobile_camera(body.get('url', ''))
            self._send_json(result)
        elif path == '/api/disconnect_camera':
            result = DRONE.disconnect_mobile_camera()
            self._send_json(result)
        elif path == '/api/capture':
            result = DRONE.capture_moment()
            self._send_json(result)

        # أنظمة عسكرية
        elif path == '/api/lock_target':
            result = DRONE.lock_target(body)
            self._send_json(result)
        elif path == '/api/fire_weapon':
            result = DRONE.fire_weapon_at_target(
                body.get('weapon_type', 'guided_missile'),
                body.get('target_id', 0),
                body.get('auth_code', '')
            )
            self._send_json(result)
        elif path == '/api/add_no_fly_zone':
            result = DRONE.add_no_fly_zone(
                body.get('cx', 0), body.get('cy', 0),
                body.get('radius', 10), body.get('reason', 'military_zone')
            )
            self._send_json(result)

        # Gemini AI
        elif path == '/api/gemini_chat':
            response = DRONE.gemini_chat(body.get('message', ''))
            self._send_json({'response': response})
        elif path == '/api/gemini_search':
            result = DRONE.gemini_search(body.get('query', ''))
            self._send_json(result)
        elif path == '/api/gemini_analyze':
            result = DRONE.analyze_with_gemini(body.get('image', ''), body.get('context', ''))
            self._send_json(result)
        elif path == '/api/gemini_scene':
            result = DRONE.gemini_generate_scene(body.get('type', ''), body.get('details', ''))
            self._send_json({'report': result})

        # أنظمة أخرى
        elif path == '/api/encrypt':
            result = DRONE.encrypt_communication(body.get('message', ''), body.get('recipient', ''))
            self._send_json(result)
        elif path == '/api/signal_scan_custom':
            result = DRONE.signal_scan(body.get('start', 100), body.get('end', 3000))
            self._send_json(result)
        elif path == '/api/thermal_scan':
            result = DRONE.thermal_scan()
            self._send_json(result)
        else:
            self._send_json({'error': 'Not found'}, 404)

    def _serve_main_page(self):
        html = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚁 النظام الاستخباراتي العسكري الأسطوري v9.1</title>
    <style>
        :root {
            --bg: #0a0e14;
            --panel: #131820;
            --border: #1e2a3a;
            --accent: #00d4aa;
            --danger: #ff4757;
            --warning: #ffa502;
            --info: #3742fa;
            --text: #c8d6e5;
            --text2: #8395a7;
            --gold: #f9ca24;
            --green: #2ed573;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Segoe UI', 'Tajawal', Tahoma, sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* خلفية متحركة */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse at 20% 50%, rgba(0,212,170,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(55,66,250,0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(255,71,87,0.04) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }

        /* خطوط شبكية */
        body::after {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image:
                linear-gradient(rgba(30,42,58,0.3) 1px, transparent 1px),
                linear-gradient(90deg, rgba(30,42,58,0.3) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
            z-index: 0;
        }

        .app-container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 10px;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, #131820 0%, #1a1f2e 100%);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 20px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
        }
        .header .title {
            font-size: 1.3em;
            font-weight: bold;
            background: linear-gradient(135deg, var(--accent), var(--gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .badge {
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            color: var(--text2);
        }
        .badge.online { border-color: var(--green); color: var(--green); }
        .badge.offline { border-color: var(--danger); color: var(--danger); }

        /* Grid */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 10px;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 15px;
            position: relative;
            overflow: hidden;
        }
        .panel::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent), transparent);
        }
        .panel h3 {
            font-size: 0.95em;
            margin-bottom: 10px;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .panel h3 .icon { font-size: 1.2em; }

        /* Telemetry */
        .telemetry-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .telem-item {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 8px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .telem-item .label { color: var(--text2); font-size: 0.8em; }
        .telem-item .value { font-weight: bold; color: var(--accent); font-size: 0.9em; }

        /* Controls */
        .btn-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 8px 0;
        }
        button {
            background: var(--panel);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.82em;
            transition: all 0.2s;
            white-space: nowrap;
        }
        button:hover {
            background: #1e2a3a;
            border-color: var(--accent);
            color: var(--accent);
        }
        button.primary {
            background: var(--accent);
            border-color: var(--accent);
            color: #000;
            font-weight: bold;
        }
        button.danger {
            background: var(--danger);
            border-color: var(--danger);
            color: #fff;
        }
        button.warning {
            background: var(--warning);
            border-color: var(--warning);
            color: #000;
        }

        /* D-Pad */
        .dpad {
            display: grid;
            grid-template-columns: 60px 60px 60px;
            grid-template-rows: 60px 60px 60px;
            gap: 4px;
            justify-content: center;
            margin: 10px auto;
        }
        .dpad button {
            width: 60px;
            height: 60px;
            font-size: 1.5em;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
        }
        .dpad .empty { background: transparent; border: none; }

        /* Logs */
        .log-container {
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.75em;
            font-family: 'Courier New', monospace;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 8px;
        }
        .log-container .log-line {
            padding: 3px 0;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            color: var(--text2);
        }

        /* Camera */
        .camera-view {
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        .camera-view img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .camera-placeholder {
            color: var(--text2);
            font-size: 0.9em;
            text-align: center;
        }

        /* Map */
        .mini-map {
            width: 100%;
            aspect-ratio: 1;
            background: rgba(0,0,0,0.4);
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }
        .mini-map canvas {
            position: absolute;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
        }

        /* Targets */
        .target-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .target-item {
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            padding: 8px;
            margin: 4px 0;
            font-size: 0.8em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .target-item .threat-high { color: var(--danger); }
        .target-item .threat-med { color: var(--warning); }
        .target-item .threat-low { color: var(--green); }

        /* Responsive */
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .header .title { font-size: 1em; }
            .dpad button { width: 50px; height: 50px; }
            .dpad { grid-template-columns: 50px 50px 50px; grid-template-rows: 50px 50px 50px; }
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        /* Animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulse { animation: pulse 2s infinite; }
        .blink { animation: pulse 0.5s infinite; }

        /* Input */
        input, select {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.82em;
            width: 100%;
            margin: 4px 0;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <div class="header">
            <span class="title">🚁 النظام الاستخباراتي العسكري الأسطوري v9.1</span>
            <span id="connectionBadge" class="badge offline">⚫ غير متصل</span>
            <span id="geminiBadge" class="badge">🤖 Gemini: --</span>
            <span id="yoloBadge" class="badge">👁️ YOLO: --</span>
        </div>

        <!-- Grid -->
        <div class="grid">
            <!-- لوحة التحكم -->
            <div class="panel">
                <h3><span class="icon">🎮</span> وحدة التحكم</h3>
                <div class="btn-row">
                    <button class="primary" onclick="apiPost('/api/auth', {operator_id:'commander',credentials:{token:'EDU-TOKEN-2026'}})">🔐 مصادقة</button>
                    <button onclick="apiPost('/api/connect')">🔌 اتصال</button>
                    <button class="warning" onclick="apiPost('/api/arm')">▶️ تسليح</button>
                    <button class="danger" onclick="apiPost('/api/land')">🛬 هبوط</button>
                </div>
                <div class="btn-row">
                    <button onclick="apiPost('/api/rth')">🏠 عودة للمنزل</button>
                    <button onclick="apiPost('/api/set_home', {x:0, y:0})">📍 تعيين المنزل</button>
                </div>
                <div class="dpad">
                    <div class="empty"></div>
                    <button onclick="apiPost('/api/move', {direction:'forward'})">▲</button>
                    <div class="empty"></div>
                    <button onclick="apiPost('/api/move', {direction:'left'})">◄</button>
                    <button onclick="apiPost('/api/move', {direction:'stop'})">■</button>
                    <button onclick="apiPost('/api/move', {direction:'right'})">►</button>
                    <div class="empty"></div>
                    <button onclick="apiPost('/api/move', {direction:'back'})">▼</button>
                    <div class="empty"></div>
                </div>
                <div class="btn-row">
                    <button onclick="addWaypoint()">📍 إضافة نقطة</button>
                    <button onclick="apiPost('/api/clear_waypoints')">🗑️ مسح المسار</button>
                    <button onclick="startPatrol()">🔄 بدء دورية</button>
                </div>
            </div>

            <!-- البيانات الفورية -->
            <div class="panel">
                <h3><span class="icon">📡</span> بيانات الطيران</h3>
                <div class="telemetry-grid" id="telemetry">
                    <div class="telem-item"><span class="label">الموقع</span><span class="value" id="t-pos">--</span></div>
                    <div class="telem-item"><span class="label">الارتفاع</span><span class="value" id="t-alt">--</span></div>
                    <div class="telem-item"><span class="label">البطارية</span><span class="value" id="t-bat">--</span></div>
                    <div class="telem-item"><span class="label">الإشارة</span><span class="value" id="t-sig">--</span></div>
                    <div class="telem-item"><span class="label">الوضع</span><span class="value" id="t-mode">--</span></div>
                    <div class="telem-item"><span class="label">السرعة</span><span class="value" id="t-speed">--</span></div>
                </div>
                <div style="margin-top:10px;">
                    <strong>المستشعرات:</strong>
                    <span id="sensors" style="font-size:0.8em;color:var(--text2)">--</span>
                </div>
            </div>

            <!-- كاميرا -->
            <div class="panel">
                <h3><span class="icon">📸</span> كاميرا المراقبة</h3>
                <div class="camera-view" id="cameraView">
                    <span class="camera-placeholder">📷 الكاميرا غير متصلة</span>
                </div>
                <div class="btn-row" style="margin-top:8px;">
                    <input type="text" id="cameraUrl" placeholder="رابط كاميرا الجوال (مثال: http://192.168.1.5:8080/video)" style="flex:1;">
                    <button onclick="connectCamera()">📱 اتصال</button>
                    <button class="danger" onclick="apiPost('/api/disconnect_camera')">قطع</button>
                </div>
                <div class="btn-row">
                    <button onclick="captureMoment()">📸 التقاط وتحليل</button>
                    <button onclick="apiGet('/api/camera_frame')">🔄 تحديث</button>
                </div>
                <div id="detectionInfo" style="font-size:0.8em;margin-top:5px;color:var(--text2);"></div>
            </div>

            <!-- أنظمة التسليح -->
            <div class="panel">
                <h3><span class="icon">🎯</span> نظام التسليح</h3>
                <div id="weaponsInfo" style="font-size:0.8em;margin-bottom:8px;">--</div>
                <div class="btn-row">
                    <button onclick="lockTarget()">🔒 تثبيت هدف</button>
                    <button class="danger" onclick="fireWeapon()">🚀 إطلاق</button>
                </div>
                <div style="margin-top:8px;">
                    <strong>الأهداف المثبتة:</strong>
                    <div id="lockedTargets" style="font-size:0.75em;color:var(--text2);">--</div>
                </div>
                <div style="margin-top:8px;">
                    <strong>مناطق حظر الطيران:</strong>
                    <div id="noFlyZones" style="font-size:0.75em;color:var(--text2);">--</div>
                </div>
                <div class="btn-row" style="margin-top:5px;">
                    <button onclick="addNoFlyZone()">🚫 إضافة منطقة حظر</button>
                </div>
            </div>

            <!-- الخريطة -->
            <div class="panel">
                <h3><span class="icon">🗺️</span> الخريطة التكتيكية</h3>
                <div class="mini-map" id="miniMap">
                    <canvas id="mapCanvas"></canvas>
                </div>
                <div class="btn-row" style="margin-top:5px;">
                    <button onclick="apiGet('/api/heatmap')">🔥 خريطة حرارية</button>
                    <button onclick="apiGet('/api/battlefield')">🎯 تحليل المعركة</button>
                </div>
            </div>

            <!-- استخبارات -->
            <div class="panel">
                <h3><span class="icon">🕵️</span> وحدة الاستخبارات</h3>
                <div class="btn-row">
                    <button onclick="thermalScan()">🌡️ مسح حراري</button>
                    <button onclick="signalScan()">📡 مسح ترددات</button>
                </div>
                <div class="btn-row">
                    <button onclick="geminiChat()">💬 Gemini محادثة</button>
                    <button onclick="geminiSearch()">🔍 بحث استخباراتي</button>
                </div>
                <div class="btn-row">
                    <button onclick="geminiScene()">📝 توليد تقرير</button>
                    <button onclick="apiGet('/api/tactical_options')">⚔️ خيارات تكتيكية</button>
                </div>
                <div id="intelResult" style="font-size:0.75em;margin-top:8px;max-height:150px;overflow-y:auto;color:var(--text2);"></div>
            </div>

            <!-- سجل الأحداث -->
            <div class="panel" style="grid-column: 1/-1;">
                <h3><span class="icon">📋</span> سجل الأحداث</h3>
                <div class="log-container" id="logs">
                    <span style="color:var(--text2)">⏳ جاري تحميل السجلات...</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ========== API Functions ==========
        const API_BASE = '';

        async function apiGet(path) {
            try {
                const res = await fetch(API_BASE + path);
                const data = await res.json();
                console.log('GET', path, data);
                return data;
            } catch(e) {
                console.error('GET Error:', path, e);
                return null;
            }
        }

        async function apiPost(path, body = {}) {
            try {
                const res = await fetch(API_BASE + path, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                console.log('POST', path, data);
                return data;
            } catch(e) {
                console.error('POST Error:', path, e);
                return null;
            }
        }

        // ========== UI Functions ==========
        function connectCamera() {
            const url = document.getElementById('cameraUrl').value;
            if (!url) { alert('الرجاء إدخال رابط الكاميرا'); return; }
            apiPost('/api/connect_camera', {url: url}).then(r => {
                if (r && r.success) alert('✅ تم الاتصال بالكاميرا');
                else alert('❌ فشل الاتصال');
            });
        }

        function captureMoment() {
            apiPost('/api/capture').then(r => {
                if (r && r.frame) {
                    document.getElementById('cameraView').innerHTML = `<img src="data:image/jpeg;base64,${r.frame}" />`;
                }
                if (r && r.detection_count !== undefined) {
                    document.getElementById('detectionInfo').innerHTML =
                        `كشف: ${r.detection_count} جسم | تهديد عالي: ${r.high_threats || 0}`;
                }
            });
        }

        function addWaypoint() {
            const x = parseFloat(prompt('الإحداثي X:')) || 0;
            const y = parseFloat(prompt('الإحداثي Y:')) || 0;
            apiPost('/api/add_waypoint', {x, y});
        }

        function startPatrol() {
            const cx = parseFloat(prompt('مركز X:')) || 0;
            const cy = parseFloat(prompt('مركز Y:')) || 0;
            const r = parseFloat(prompt('نصف القطر:')) || 10;
            apiPost('/api/start_patrol', {cx, cy, radius: r});
        }

        function lockTarget() {
            const type = prompt('نوع الهدف (person/vehicle/drone):') || 'vehicle';
            apiPost('/api/lock_target', {type, velocity: [5, 0], size: 5});
        }

        function fireWeapon() {
            const weapon = prompt('نوع السلاح (guided_missile/precision_bomb/camera_munition):') || 'guided_missile';
            const targetId = parseInt(prompt('رقم الهدف:')) || 0;
            const auth = prompt('رمز التفويض (ALPHA-1/BRAVO-2/CHARLIE-3/COMMANDER):') || 'COMMANDER';
            apiPost('/api/fire_weapon', {weapon_type: weapon, target_id: targetId, auth_code: auth});
        }

        function addNoFlyZone() {
            const cx = parseFloat(prompt('مركز X:')) || 0;
            const cy = parseFloat(prompt('مركز Y:')) || 0;
            const r = parseFloat(prompt('نصف القطر:')) || 10;
            const reason = prompt('السبب:') || 'military_zone';
            apiPost('/api/add_no_fly_zone', {cx, cy, radius: r, reason});
        }

        function thermalScan() {
            apiPost('/api/thermal_scan').then(r => {
                document.getElementById('intelResult').innerHTML =
                    `<pre style="white-space:pre-wrap">${JSON.stringify(r, null, 2)}</pre>`;
            });
        }

        function signalScan() {
            apiPost('/api/signal_scan_custom', {start: 100, end: 3000}).then(r => {
                document.getElementById('intelResult').innerHTML =
                    `<pre style="white-space:pre-wrap">${JSON.stringify(r, null, 2)}</pre>`;
            });
        }

        function geminiChat() {
            const msg = prompt('أدخل سؤالك:');
            if (!msg) return;
            apiPost('/api/gemini_chat', {message: msg}).then(r => {
                document.getElementById('intelResult').innerHTML =
                    `<div style="background:rgba(0,212,170,0.1);padding:8px;border-radius:8px;">${r?.response || '--'}</div>`;
            });
        }

        function geminiSearch() {
            const q = prompt('أدخل موضوع البحث:');
            if (!q) return;
            apiPost('/api/gemini_search', {query: q}).then(r => {
                document.getElementById('intelResult').innerHTML =
                    `<pre style="white-space:pre-wrap">${JSON.stringify(r, null, 2)}</pre>`;
            });
        }

        function geminiScene() {
            const type = prompt('نوع المشهد (معركة/استطلاع/دورية):') || 'استطلاع';
            apiPost('/api/gemini_scene', {type}).then(r => {
                document.getElementById('intelResult').innerHTML =
                    `<pre style="white-space:pre-wrap">${r?.report || '--'}</pre>`;
            });
        }

        // ========== تحديث دوري ==========
        async function updateTelemetry() {
            const data = await apiGet('/api/telemetry');
            if (!data) return;

            document.getElementById('t-pos').textContent = `${data.position?.x}, ${data.position?.y}`;
            document.getElementById('t-alt').textContent = `${data.altitude} م`;
            document.getElementById('t-bat').textContent = `${data.battery}%`;
            document.getElementById('t-sig').textContent = `${data.signal_strength}%`;
            document.getElementById('t-mode').textContent = data.mode;
            document.getElementById('t-speed').textContent = `${data.velocity ? Math.hypot(data.velocity.vx, data.velocity.vy).toFixed(1) : 0} م/ث`;

            const badge = document.getElementById('connectionBadge');
            if (data.armed) {
                badge.textContent = '🟢 مسلح - في الجو';
                badge.className = 'badge online';
            } else if (data.connected) {
                badge.textContent = '🟡 متصل - على الأرض';
                badge.className = 'badge online';
            } else {
                badge.textContent = '⚫ غير متصل';
                badge.className = 'badge offline';
            }

            document.getElementById('geminiBadge').textContent = `🤖 Gemini: ${data.gemini_active ? '✅' : '⚠️ محاكاة'}`;
            document.getElementById('yoloBadge').textContent = `👁️ YOLO: ${data.yolo_active ? '✅' : '⚠️ محاكاة'}`;

            // تحديث المستشعرات
            const logs = data.logs || [];
            const sensorLog = logs.find(l => l.includes('مستشعرات'));
            document.getElementById('sensors').textContent = 'أمام/خلف/يسار/يمين/أسفل';

            // رسم الخريطة
            drawMap(data);

            // تحديث الكاميرا إذا كان البث نشطاً
            if (data.mobile_streaming) {
                updateCameraFrame();
            }
        }

        async function updateLogs() {
            const data = await apiGet('/api/logs');
            if (data && data.logs) {
                const container = document.getElementById('logs');
                container.innerHTML = data.logs.slice(-30).reverse().map(l =>
                    `<div class="log-line">${escapeHtml(l)}</div>`
                ).join('');
            }
        }

        async function updateWeapons() {
            const data = await apiGet('/api/weapons');
            if (data && data.weapons_detail) {
                const info = Object.entries(data.weapons_detail).map(([k, v]) => `${k}: ${v}`).join(' | ');
                document.getElementById('weaponsInfo').textContent = `الذخيرة: ${info} | الإجمالي: ${data.total_weapons}`;
            }
        }

        async function updateTargets() {
            const data = await apiGet('/api/targets');
            if (data) {
                const locked = await apiGet('/api/locked_targets');
                document.getElementById('lockedTargets').textContent =
                    locked?.locked_targets?.length ? `${locked.locked_targets.length} هدف مثبت` : 'لا يوجد';
            }
        }

        async function updateNoFlyZones() {
            const data = await apiGet('/api/no_fly_zones');
            if (data && data.zones) {
                document.getElementById('noFlyZones').textContent =
                    data.zones.length ? `${data.zones.length} منطقة نشطة` : 'لا توجد مناطق حظر';
            }
        }

        async function updateCameraFrame() {
            const data = await apiGet('/api/camera_frame');
            if (data && data.frame) {
                document.getElementById('cameraView').innerHTML =
                    `<img src="data:image/jpeg;base64,${data.annotated_frame || data.frame}" />`;
                document.getElementById('detectionInfo').innerHTML =
                    `كشف: ${data.detection_count || 0} | تهديد عالي: ${data.high_threats || 0}`;
            }
        }

        function drawMap(data) {
            const canvas = document.getElementById('mapCanvas');
            if (!canvas) return;
            const container = document.getElementById('miniMap');
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const cx = w / 2, cy = h / 2;
            const scale = 8;

            ctx.clearRect(0, 0, w, h);

            // شبكة
            ctx.strokeStyle = 'rgba(30,42,58,0.5)';
            ctx.lineWidth = 0.5;
            for (let i = 0; i < w; i += 30) {
                ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, h); ctx.stroke();
            }
            for (let i = 0; i < h; i += 30) {
                ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(w, i); ctx.stroke();
            }

            // المنزل
            if (data.home_set) {
                const hx = cx + data.home.x * scale;
                const hy = cy - data.home.y * scale;
                ctx.fillStyle = '#f9ca24';
                ctx.beginPath(); ctx.arc(hx, hy, 5, 0, Math.PI * 2); ctx.fill();
                ctx.fillStyle = '#fff'; ctx.font = '10px sans-serif';
                ctx.fillText('🏠', hx - 8, hy - 8);
            }

            // موقع الدرون
            const dx = cx + (data.position?.x || 0) * scale;
            const dy = cy - (data.position?.y || 0) * scale;
            ctx.fillStyle = '#00d4aa';
            ctx.beginPath(); ctx.arc(dx, dy, 6, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = '#00d4aa'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(dx, dy, 10, 0, Math.PI * 2); ctx.stroke();

            // نبضة
            const pulseR = 10 + Math.sin(Date.now() / 500) * 3;
            ctx.strokeStyle = 'rgba(0,212,170,0.3)';
            ctx.beginPath(); ctx.arc(dx, dy, pulseR, 0, Math.PI * 2); ctx.stroke();

            // نقاط المسار
            // (سيتم رسمها عند توفر البيانات)
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // ========== التحديث الدوري ==========
        setInterval(() => {
            updateTelemetry();
            updateLogs();
            updateWeapons();
            updateTargets();
            updateNoFlyZones();
        }, 1500);

        // تهيئة أولية
        updateTelemetry();
        updateLogs();

        console.log('🚁 النظام الاستخباراتي العسكري الأسطوري v9.1 - جاهز');
    </script>
</body>
</html>"""
        self._send_html(html)


# =====================================================
# 🚀 نقطة البداية الرئيسية
# =====================================================

def main():
    PORT = int(os.environ.get('PORT', 8080))

    # مصادقة تلقائية للدرون
    DRONE.authenticate("commander", {"token": "EDU-TOKEN-2026"})
    DRONE.connect()
    DRONE.set_home(0, 0)

    print("=" * 60)
    print("🚁 النظام الاستخباراتي العسكري الأسطوري v9.1")
    print("=" * 60)
    print(f"🌐 الخادم يعمل على المنفذ: {PORT}")
    print(f"🔗 الرابط: http://localhost:{PORT}")
    print(f"🤖 Gemini API: {'✅ متصل' if GEMINI_AVAILABLE else '⚠️ وضع المحاكاة'}")
    print(f"👁️ YOLOv8: {'✅ متصل' if YOLO_AVAILABLE else '⚠️ وضع المحاكاة'}")
    print("=" * 60)
    print("🎖️ م/ وسيم الحميدي - هندسة الأنظمة الاستخباراتية")
    print("=" * 60)

    with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), DroneRequestHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"✅ الخادم جاهز لاستقبال الطلبات")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛬 إيقاف النظام...")
            DRONE.land()
            httpd.shutdown()


if __name__ == "__main__":
    main()
