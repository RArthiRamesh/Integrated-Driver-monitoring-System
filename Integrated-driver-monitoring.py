import streamlit as st
import cv2
import numpy as np
import torch
import dlib
import mediapipe as mp
from ultralytics import YOLO
import pygame
import pyttsx3
import threading
from scipy.spatial import distance
import os
import time
import google.generativeai as genai
from datetime import datetime
import PIL.Image
import tempfile
from pymongo import MongoClient
from bson import ObjectId
from geopy.geocoders import Nominatim
from geopy.location import Location
from geopy.exc import GeocoderTimedOut
import geocoder

class LocationTracker:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="driver_safety_app")
        self.current_location = None
        self.location_history = []
        self.last_update_time = 0
        self.update_interval = 5  # Update location every 5 seconds

    def get_current_location(self):
        try:
            g = geocoder.ip('me')
            if g.ok:
                self.current_location = {
                    'latitude': g.lat,
                    'longitude': g.lng,
                    'address': g.address,
                    'timestamp': datetime.now()
                }
                self.location_history.append(self.current_location)
                return self.current_location
            return None
        except Exception as e:
            print(f"Error getting location: {str(e)}")
            return None

    def should_update_location(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def get_location_info(self, lat, lon):
        try:
            location = self.geolocator.reverse((lat, lon))
            return location.address if location else "Unknown location"
        except GeocoderTimedOut:
            return "Location lookup timed out"
        except Exception as e:
            return f"Error getting location info: {str(e)}"

class IntegratedDriverSafetySystem:
    def __init__(self):
        # Internal API key configuration
        self.GEMINI_API_KEY = "AIzaSyDlIRYdegfbkLXq83OBIotfavhoXHqdHvE"
        
        # Initialize MongoDB connection
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017/")
            self.db = self.mongo_client["cscorner"]
            self.reports_collection = self.db["safety_reports"]
            print("MongoDB connection successful")
        except Exception as e:
            st.error(f"MongoDB connection error: {str(e)}")
            self.mongo_client = None
            self.db = None
            self.reports_collection = None

        # Initialize face detection and landmark prediction
        self.face_detector = dlib.get_frontal_face_detector()
        
        try:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                self.landmark_predictor = dlib.shape_predictor(predictor_path)
            else:
                st.warning("Face landmark predictor file not found. Download it from dlib's website.")
                self.landmark_predictor = None
        except Exception as e:
            st.error(f"Error loading face landmark predictor: {str(e)}")
            self.landmark_predictor = None
        
        # Initialize pose detection
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize YOLO model
        try:
            self.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            st.error(f"Error loading YOLO model: {str(e)}")
            self.yolo_model = None
        
        # Initialize Gemini AI
        genai.configure(api_key=self.GEMINI_API_KEY)
        self.ai_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize audio with multiple alert sounds
        try:
            pygame.mixer.init()
            self.alert_sounds = {}
            if os.path.exists("distraction.mp3"):
                self.alert_sounds["Distraction"] = pygame.mixer.Sound("distraction.mp3")
            else:
                st.warning("Distraction alert sound file not found.")
            if os.path.exists("drowsy.mp3"):
                self.alert_sounds["Drowsiness"] = pygame.mixer.Sound("drowsy.mp3")
            else:
                st.warning("Drowsiness alert sound file not found.")
        except Exception as e:
            st.warning(f"Audio system initialization failed: {str(e)}")
            self.alert_sounds = {}
        
        # Initialize location tracker
        self.location_tracker = LocationTracker()
        
        # Initialize metrics and parameters
        self.reset_metrics()
        
        # Drowsiness detection parameters
        self.drowsiness_threshold = 0.25
        self.drowsiness_frames = 0
        self.alert_cooldown = 0
        self.last_alert_time = time.time()
        self.recording = False
        self.record_frames = []
        self.start_time = time.time()

    def reset_metrics(self):
        """Reset all safety metrics"""
        self.metrics = {
            'total_frames': 0,
            'drowsiness_count': 0,
            'distraction_count': 0,
            'posture_violations': 0,
            'phone_usage': 0,
            'looking_away': 0,
            'lane_deviation_count': 0,
            'alerts': [],
            'session_duration': 0,
            'location_history': [],
            'current_location': None,
            'location_updates': 0
        }

    def calculate_ear(self, eye_points):
        """Calculate eye aspect ratio"""
        try:
            vertical_dist1 = distance.euclidean(eye_points[1], eye_points[5])
            vertical_dist2 = distance.euclidean(eye_points[2], eye_points[4])
            horizontal_dist = distance.euclidean(eye_points[0], eye_points[3])
            ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
            return ear
        except:
            return 0.3

    def play_alert(self, alert_type):
        """Play specific alert sound based on alert type with cooldown"""
        current_time = time.time()
        if current_time - self.last_alert_time > 3.0:  # 3 seconds cooldown
            if alert_type in self.alert_sounds:
                self.alert_sounds[alert_type].play()
            self.metrics['alerts'].append(f"{alert_type} - {datetime.now().strftime('%H:%M:%S')}")
            self.last_alert_time = current_time

    def detect_phone(self, frame, objects):
        """Detect phone usage in frame"""
        phone_detected = False
        for obj in objects:
            if obj['name'] in ['cell phone', 'mobile phone'] and obj['confidence'] > 0.6:
                phone_detected = True
                self.metrics['phone_usage'] += 1
                cv2.putText(frame, "PHONE DETECTED!", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.play_alert("Distraction")  # Use distraction alert for phone usage
        return frame

    def detect_driver_distraction(self, frame):
        """Detect driver distraction using pose estimation"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb_frame)
        
        distraction_detected = False
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            
            left_dist = np.sqrt((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)
            right_dist = np.sqrt((right_wrist.x - nose.x)**2 + (right_wrist.y - nose.y)**2)
            
            if left_dist < 0.2 or right_dist < 0.2:
                distraction_detected = True
                self.metrics['distraction_count'] += 1
                cv2.putText(frame, "DISTRACTION DETECTED!", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.play_alert("Distraction")
        
        return frame, distraction_detected

    def process_frame(self, frame, is_video_analysis=False):
        """Process a single frame and update location data"""
        if frame is None:
            return frame

        # Update location only if not in video analysis mode
        if not is_video_analysis and self.location_tracker.should_update_location():
            location_data = self.location_tracker.get_current_location()
            if location_data:
                self.metrics['current_location'] = location_data
                self.metrics['location_history'].append(location_data)
                self.metrics['location_updates'] += 1
        
        # Update session duration
        self.metrics['session_duration'] = int(time.time() - self.start_time)
        self.metrics['total_frames'] += 1
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        # Face detection
        faces = self.face_detector(frame_rgb)
        face_detected = False
        
        for face in faces:
            face_detected = True
            if self.landmark_predictor:
                landmarks = self.landmark_predictor(frame_rgb, face)
                
                # Get eye landmarks
                left_eye = []
                right_eye = []
                for n in range(36, 42):
                    left_eye.append((landmarks.part(n).x, landmarks.part(n).y))
                for n in range(42, 48):
                    right_eye.append((landmarks.part(n).x, landmarks.part(n).y))
                
                # Calculate EAR
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                if ear < self.drowsiness_threshold:
                    self.drowsiness_frames += 1
                    if self.drowsiness_frames > 5:
                        self.metrics['drowsiness_count'] += 1
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.play_alert("Drowsiness")
                else:
                    self.drowsiness_frames = 0
                
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if not face_detected:
            self.metrics['looking_away'] += 1
            if self.metrics['looking_away'] > 30:
                cv2.putText(frame, "ATTENTION: EYES ON ROAD!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.play_alert("Distraction")  # Use distraction alert for looking away
        else:
            self.metrics['looking_away'] = 0
        
        # Object detection
        detected_objects = []
        if self.yolo_model:
            results = self.yolo_model(frame_rgb)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    name = result.names[cls]
                    
                    if conf > 0.5:
                        detected_objects.append({
                            'name': name,
                            'confidence': conf,
                            'box': (int(x1), int(y1), int(x2), int(y2))
                        })
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Detect phone usage
        frame = self.detect_phone(frame, detected_objects)
        
        # Detect driver distraction
        frame, _ = self.detect_driver_distraction(frame)
        
        # Add location overlay only if not in video analysis mode
        if not is_video_analysis and self.metrics['current_location']:
            location_text = f"Location: {self.metrics['current_location']['address']}"
            cv2.putText(frame, location_text[:60] + ('...' if len(location_text) > 60 else ''),
                      (10, frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add session duration
        cv2.putText(frame, f"Session: {self.metrics['session_duration']}s", (frame_width - 150, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def analyze_traffic_scene(self, image):
        """AI-powered traffic scene analysis"""
        analysis_prompt = """
        Analyze this traffic scene in detail:
        1. Identify potential safety risks
        2. Detect traffic rule violations
        3. Provide specific safety recommendations
        4. Assess overall traffic situation
        """
        
        try:
            response = self.ai_model.generate_content([analysis_prompt, image])
            return response.text
        except Exception as e:
            return f"Analysis error: {str(e)}"

    def save_report_to_mongodb(self, report_data):
        """Save safety report to MongoDB"""
        if self.reports_collection is not None:
            try:
                report_data['timestamp'] = datetime.now()
                result = self.reports_collection.insert_one(report_data)
                return str(result.inserted_id)
            except Exception as e:
                st.error(f"Error saving report to MongoDB: {str(e)}")
                return None
        return None

    def get_report_by_id(self, report_id):
        """Fetch a specific report by its ID from MongoDB"""
        if self.reports_collection is not None:
            try:
                object_id = ObjectId(report_id)
                report = self.reports_collection.find_one({"_id": object_id})
                if report:
                    report['_id'] = str(report['_id'])
                    return report
                else:
                    st.warning("No report found with the provided ID.")
                    return None
            except Exception as e:
                st.error(f"Error fetching report: {str(e)}")
                return None
        return None

    def generate_safety_report(self, report_id=None, is_video_analysis=False):
        """Generate a comprehensive safety report with location data"""
        if report_id:
            report_data = self.get_report_by_id(report_id)
            if report_data:
                return report_data.get('text', "Report content not found.")
            return None

        session_minutes = self.metrics['session_duration'] // 60
        alert_rate = (self.metrics['drowsiness_count'] + self.metrics['distraction_count']) / max(1, session_minutes)
        
        report = f"""
        🚗 Driver Safety Report
        
        📊 Session Statistics:
        - Duration: {session_minutes} minutes
        - Total Frames Analyzed: {self.metrics['total_frames']}
        - Alert Rate: {alert_rate:.1f} alerts/minute
        
        ⚠️ Incidents:
        - Drowsiness Detected: {self.metrics['drowsiness_count']} times
        - Distraction Events: {self.metrics['distraction_count']} times
        - Phone Usage Detected: {self.metrics['phone_usage']} times
        - Looking Away: {self.metrics['looking_away']} incidents
        - Lane Deviations: {self.metrics['lane_deviation_count']} times
        """
        
        # Add location information only if not in video analysis mode
        if not is_video_analysis:
            report += "\n📍 Location Information:\n"
            if self.metrics['location_history']:
                unique_locations = set(loc['address'] for loc in self.metrics['location_history'])
                report += f"- Total Location Updates: {self.metrics['location_updates']}\n"
                report += f"- Unique Locations Visited: {len(unique_locations)}\n\n"
                report += "Recent Locations:\n"
                for loc in self.metrics['location_history'][-5:]:
                    report += f"- {loc['address']} at {loc['timestamp'].strftime('%H:%M:%S')}\n"
            else:
                report += "No location data available for this session.\n"
        
        report += f"""
        🔔 Recent Alerts:
        {chr(10).join(['- ' + alert for alert in self.metrics['alerts'][-5:]])}
        
        💡 Safety Recommendations:
        1. Take a break every 2 hours or when feeling tired
        2. Maintain proper posture and keep eyes on the road
        3. Avoid phone usage while driving
        4. Ensure adequate rest before long drives
        5. Stay focused and minimize distractions
        
        ⚠️ Risk Level: {'High' if alert_rate > 2 else 'Moderate' if alert_rate > 1 else 'Low'}
        """
        
        # Save report to MongoDB
        report_data = {
            'text': report,
            'metrics': self.metrics,
            'location_history': [] if is_video_analysis else self.metrics['location_history'],
            'session_minutes': session_minutes,
            'alert_rate': alert_rate,
            'risk_level': 'High' if alert_rate > 2 else 'Moderate' if alert_rate > 1 else 'Low'
        }
        report_id = self.save_report_to_mongodb(report_data)
        if report_id:
            report += f"\n\nReport ID: {report_id}"
        
        return report

def main():
    st.set_page_config(page_title="Integrated Driver Safety System", layout="wide")
    st.title("Integrated Driver Safety Monitoring System")
    
    # Initialize safety system
    try:
        safety_system = IntegratedDriverSafetySystem()
        st.success("System initialized successfully with internal API configuration")
    except Exception as e:
        st.error(f"Error initializing safety system: {str(e)}")
        return
    
    # System status indicators
    st.sidebar.header("System Status")
    st.sidebar.info("✓ Core System: Active")
    st.sidebar.success("✓ API Configuration: Internal")
    
    if safety_system.landmark_predictor:
        st.sidebar.success("✓ Face Detection: Active")
    else:
        st.sidebar.warning("⚠ Face Detection: Disabled")
        
    if safety_system.yolo_model:
        st.sidebar.success("✓ Object Detection: Active")
    else:
        st.sidebar.warning("⚠ Object Detection: Disabled")
    
    # Feature Selection
    st.sidebar.header("Analysis Mode")
    feature_mode = st.sidebar.selectbox("Select Mode", [
        "Live Monitoring",
        "Video Analysis",
        "Traffic Scene Analysis",
        "Safety Report"
    ])
    
    # Settings
    st.sidebar.header("Settings")
    safety_system.drowsiness_threshold = st.sidebar.slider(
        "Drowsiness Threshold",
        min_value=0.15,
        max_value=0.35,
        value=0.25,
        step=0.01
    )
    
    if feature_mode == "Live Monitoring":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Live Camera Feed")
            camera_placeholder = st.empty()
        
        with col2:
            st.header("Safety Metrics")
            metrics_placeholder = st.empty()
            
            if st.button("Generate Safety Report"):
                report = safety_system.generate_safety_report()
                st.markdown(report)
            
            if st.button("Reset Metrics"):
                safety_system.reset_metrics()
                st.success("Metrics reset successfully")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                processed_frame = safety_system.process_frame(frame)
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
                
                # Update metrics
                metrics_placeholder.markdown(f"""
                📊 Current Statistics:
                - ⏱️ Session Duration: {safety_system.metrics['session_duration']} seconds
                - 😴 Drowsiness Alerts: {safety_system.metrics['drowsiness_count']}
                - 📱 Phone Usage Detected: {safety_system.metrics['phone_usage']}
                - 👀 Looking Away Incidents: {safety_system.metrics['looking_away']}
                - 🚗 Distraction Events: {safety_system.metrics['distraction_count']}
                
                📍 Current Location:
                {safety_system.metrics['current_location']['address'] if safety_system.metrics['current_location'] else 'Acquiring location...'}
                
                Recent Alerts:
                {chr(10).join(['- ' + alert for alert in safety_system.metrics['alerts'][-3:]])}
                """)
                
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"Error during video processing: {str(e)}")
        
        finally:
            cap.release()

    elif feature_mode == "Video Analysis":
        st.header("Video Analysis")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file and st.button("Analyze Video"):
            # Create temporary file
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(uploaded_file.read())
                
                # Display uploaded video"
                st.video(uploaded_file)
                
                with st.spinner("Analyzing video..."):
                    cap = cv2.VideoCapture(temp_file_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0)
                    
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Pass is_video_analysis=True to process_frame
                        safety_system.process_frame(frame, is_video_analysis=True)
                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames)
                    
                    cap.release()
                    
                    st.success("Video analysis complete!")
                    # Pass is_video_analysis=True to generate_safety_report
                    report = safety_system.generate_safety_report(is_video_analysis=True)
                    st.markdown(report)
                    
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception:
                        pass

    elif feature_mode == "Traffic Scene Analysis":
        st.header("Traffic Scene Analysis")
        uploaded_image = st.file_uploader("Upload Traffic Scene Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_image:
            image = PIL.Image.open(uploaded_image)
            st.image(image, caption="Uploaded Traffic Scene", use_container_width=True)
            
            if st.button("Analyze Scene"):
                with st.spinner("Analyzing traffic scene..."):
                    analysis = safety_system.analyze_traffic_scene(image)
                    st.subheader("Analysis Results")
                    st.markdown(analysis)

    else:  # Safety Report
        st.header("Safety Performance Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Session Duration", f"{safety_system.metrics['session_duration']} seconds")
            st.metric("Drowsiness Incidents", safety_system.metrics['drowsiness_count'])
            st.metric("Phone Usage Detected", safety_system.metrics['phone_usage'])
        
        with col2:
            st.metric("Distraction Events", safety_system.metrics['distraction_count'])
            st.metric("Looking Away Incidents", safety_system.metrics['looking_away'])
            st.metric("Total Frames Analyzed", safety_system.metrics['total_frames'])
        
        # Input for fetching specific report
        report_id = st.text_input("Enter Report ID to fetch specific report")
        
        if st.button("Generate Detailed Report"):
            with st.spinner("Generating report..."):
                report = safety_system.generate_safety_report(report_id if report_id else None)
                if report:
                    st.markdown(report)
                    
                    # Download button for report
                    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"safety_report_{report_time}.txt",
                        mime="text/plain"
                    )
def load_custom_css():
    st.markdown("""
    <style>
    /* Hide Streamlit elements */
    .stApp > header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    #MainMenu {visibility: hidden;}
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Custom card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(145deg, #2d5aa0, #1e4080);
        border-radius: 12px;
        padding: 15px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #64ffda;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #b0bec5;
        margin-top: 5px;
    }
    
    /* Status indicator styling */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        backdrop-filter: blur(5px);
    }
    
    .status-active {
        border-left: 4px solid #4caf50;
    }
    
    .status-warning {
        border-left: 4px solid #ff9800;
    }
    
    .status-error {
        border-left: 4px solid #f44336;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(145deg, #64ffda, #26a69a);
        color: #1e3c72;
        border: none;
        border-radius: 25px;
        padding: 12px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(100, 255, 218, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(100, 255, 218, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(30, 60, 114, 0.9);
        backdrop-filter: blur(10px);
    }
    
    /* Alert styling */
    .alert-high {
        background: linear-gradient(145deg, #ff5722, #d32f2f);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.3);
    }
    
    .alert-moderate {
        background: linear-gradient(145deg, #ff9800, #f57c00);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
    }
    
    .alert-low {
        background: linear-gradient(145deg, #4caf50, #388e3c);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: #64ffda;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 0 0 20px rgba(100, 255, 218, 0.5);
        margin-bottom: 30px;
        background: linear-gradient(45deg, #64ffda, #26a69a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .section-title {
        color: #64ffda;
        font-size: 1.5em;
        font-weight: bold;
        margin: 20px 0 10px 0;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(145deg, #64ffda, #26a69a);
    }
    
    /* File uploader styling */
    .stFileUploader > section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 2px dashed #64ffda;
    }
    
    /* Video container */
    .video-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
    
    /* Metrics grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()