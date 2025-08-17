import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
import random

app = Flask(__name__)

# Paths
dataset_path = 'dataset'  # Your dataset folder organized by gestures (e.g., A/, B/, C/)
model_path = 'sign_language_model.h5'  # Path to save/load the model (not used)

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the alphabet signs we want to recognize
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Updated finger states for each letter with more accurate configurations
FINGER_STATES = {
    'A': [1, 0, 0, 0, 0],  # Thumb to the side, other fingers closed
    'B': [0, 1, 1, 1, 1],  # Four fingers up, thumb tucked
    'C': [1, 1, 1, 1, 1],  # Curved hand, all fingers slightly bent in C shape
    'D': [1, 1, 0, 0, 0],  # Index up, thumb touching middle finger
    'E': [0, 0, 0, 0, 0],  # All fingers folded, palm facing forward
    'F': [1, 0, 1, 1, 1],  # Index and thumb touching, other fingers extended 
    'G': [1, 1, 0, 0, 0],  # Thumb and index extended in pointing position
    'H': [1, 1, 1, 0, 0],  # Thumb, index, middle extended horizontally
    'I': [0, 0, 0, 0, 1],  # Only pinky extended
    'K': [1, 1, 1, 0, 0],  # Thumb, index and middle in K shape
    'L': [1, 1, 0, 0, 0],  # Thumb and index in L shape
    'M': [0, 1, 1, 1, 0],  # Three middle fingers down, thumb tucked
    'N': [0, 1, 1, 0, 0],  # Index and middle down, thumb tucked
    'O': [1, 1, 1, 1, 1],  # All fingers curved meeting thumb in O shape
    'P': [1, 1, 0, 0, 0],  # Thumb and index extended, index pointing down
    'Q': [1, 0, 0, 0, 0],  # Thumb pointing down
    'R': [1, 1, 1, 0, 0],  # Thumb, index, middle with index and middle crossed
    'S': [0, 0, 0, 0, 0],  # Fist, thumb over fingers
    'T': [0, 0, 0, 0, 0],  # Fist with thumb between index and middle
    'U': [0, 1, 1, 0, 0],  # Index and middle extended parallel
    'V': [0, 1, 1, 0, 0],  # Index and middle extended in V shape
    'W': [0, 1, 1, 1, 0],  # Index, middle, ring extended in W shape
    'X': [0, 1, 0, 0, 0],  # Index bent at middle joint
    'Y': [1, 0, 0, 0, 1],  # Thumb and pinky extended, others closed
}

# Additional features for better hand shape recognition
def get_finger_angles(hand_landmarks):
    """Calculate angles between finger joints for better shape recognition"""
    angles = []
    
    # Calculate angle between three points
    def calculate_angle(p1, p2, p3):
        vector1 = [p1.x - p2.x, p1.y - p2.y]
        vector2 = [p3.x - p2.x, p3.y - p2.y]
        
        # Normalize vectors
        norm1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
        norm2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        vector1 = [vector1[0]/norm1, vector1[1]/norm1]
        vector2 = [vector2[0]/norm2, vector2[1]/norm2]
        
        # Dot product
        dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
        # Clamp to avoid numerical errors
        dot_product = max(min(dot_product, 1.0), -1.0)
        
        return np.arccos(dot_product) * 180 / np.pi
    
    # Angle at each finger PIP joint
    # Thumb: 1, 2, 3
    # Index: 5, 6, 7
    # Middle: 9, 10, 11
    # Ring: 13, 14, 15
    # Pinky: 17, 18, 19
    
    # Calculate angles at the middle joints
    finger_triplets = [
        [1, 2, 3],  # Thumb
        [5, 6, 7],  # Index
        [9, 10, 11],  # Middle
        [13, 14, 15],  # Ring
        [17, 18, 19]   # Pinky
    ]
    
    for triplet in finger_triplets:
        p1 = hand_landmarks.landmark[triplet[0]]
        p2 = hand_landmarks.landmark[triplet[1]]
        p3 = hand_landmarks.landmark[triplet[2]]
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)
    
    return angles

def is_finger_extended(hand_landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx, wrist_idx=0):
    """Improved check if a finger is extended based on its joints' positions"""
    tip = hand_landmarks.landmark[finger_tip_idx]
    pip = hand_landmarks.landmark[finger_pip_idx]
    mcp = hand_landmarks.landmark[finger_mcp_idx]
    wrist = hand_landmarks.landmark[wrist_idx]
    
    # For thumb (special case)
    if finger_tip_idx == 4:
        # Check if thumb is extended to the side
        # This uses the cross product of the thumb vector and palm normal to determine side extension
        thumb_vec = [tip.x - mcp.x, tip.y - mcp.y]
        # Use index and pinky MCPs to get palm orientation
        index_mcp = hand_landmarks.landmark[5]
        pinky_mcp = hand_landmarks.landmark[17]
        palm_vec = [pinky_mcp.x - index_mcp.x, pinky_mcp.y - index_mcp.y]
        
        # Check thumb distance from palm
        palm_center_x = (index_mcp.x + pinky_mcp.x) / 2
        palm_center_y = (index_mcp.y + pinky_mcp.y) / 2
        thumb_palm_dist = np.sqrt((tip.x - palm_center_x)**2 + (tip.y - palm_center_y)**2)
        
        # Simple cross product check
        cross_z = thumb_vec[0] * palm_vec[1] - thumb_vec[1] * palm_vec[0]
        
        # Check if thumb is in front of palm (z-coordinate)
        thumb_forward = tip.z < mcp.z
        
        # Combined check: thumb is either sticking out to the side or forward
        return (cross_z > 0.02) or (thumb_palm_dist > 0.1 and thumb_forward)
    else:
        # For other fingers, check if the tip is significantly above the PIP joint
        # and consider the 3D position (z-coordinate)
        length_mcp_to_pip = np.sqrt((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2 + (pip.z - mcp.z)**2)
        length_pip_to_tip = np.sqrt((tip.x - pip.x)**2 + (tip.y - pip.y)**2 + (tip.z - pip.z)**2)
        
        # If the finger is extended, the tip should be far from the MCP
        dist_mcp_to_tip = np.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2 + (tip.z - mcp.z)**2)
        
        # Compare with the wrist-to-MCP distance for normalization
        wrist_to_mcp_dist = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2)
        
        # Calculate the angle between the PIP-MCP and PIP-TIP segments
        vec1 = [mcp.x - pip.x, mcp.y - pip.y, mcp.z - pip.z]
        vec2 = [tip.x - pip.x, tip.y - pip.y, tip.z - pip.z]
        
        # Normalize vectors
        vec1_norm = np.sqrt(vec1[0]**2 + vec1[1]**2 + vec1[2]**2)
        vec2_norm = np.sqrt(vec2[0]**2 + vec2[1]**2 + vec2[2]**2)
        
        # Avoid division by zero
        if vec1_norm == 0 or vec2_norm == 0:
            finger_angle = 0
        else:
            # Normalize vectors
            vec1 = [v/vec1_norm for v in vec1]
            vec2 = [v/vec2_norm for v in vec2]
            
            # Calculate dot product
            dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
            
            # Clamp dot product to avoid numerical errors
            dot_product = max(min(dot_product, 1.0), -1.0)
            
            # Calculate angle in degrees
            finger_angle = np.arccos(dot_product) * 180 / np.pi
        
        # Extended if the distance is at least 55% of the expected length when fully extended
        # and the finger is pointing more upward than downward or the angle at the PIP joint is significant
        extended_length_thresh = 0.55 * (length_mcp_to_pip + length_pip_to_tip)
        
        # Primary check: is the finger extended distance-wise?
        distance_check = dist_mcp_to_tip > extended_length_thresh
        
        # Secondary checks:
        # 1. Is the finger tip above the MCP? (in image coords, lower Y is higher in image)
        direction_check = tip.y < mcp.y 
        
        # 2. Is the finger sufficiently straight? (angle at PIP joint)
        angle_check = finger_angle > 60
        
        # 3. Is the finger tip in front of the MCP? (closer to camera)
        depth_check = tip.z < mcp.z
        
        # A finger is considered extended if it passes the distance check AND 
        # (is pointing upward OR has a significant bend OR is pointing toward the camera)
        return distance_check and (direction_check or angle_check or depth_check)

def get_finger_state(hand_landmarks):
    """Get the state of all fingers (extended or not) with improved detection"""
    # Finger landmarks indices
    # Thumb: 4 (tip), 3 (pip), 2 (mcp)
    # Index: 8, 6, 5
    # Middle: 12, 10, 9
    # Ring: 16, 14, 13
    # Pinky: 20, 18, 17
    
    fingers = [
        is_finger_extended(hand_landmarks, 4, 3, 2),    # Thumb
        is_finger_extended(hand_landmarks, 8, 6, 5),    # Index
        is_finger_extended(hand_landmarks, 12, 10, 9),  # Middle
        is_finger_extended(hand_landmarks, 16, 14, 13), # Ring
        is_finger_extended(hand_landmarks, 20, 18, 17)  # Pinky
    ]
    
    return fingers

def extract_hand_roi(frame, hand_landmarks):
    """Extract a region of interest (ROI) around the hand for visualization"""
    if hand_landmarks is None:
        return None
        
    # Get hand bounding box
    h, w, _ = frame.shape
    x_min, x_max, y_min, y_max = w, 0, h, 0
    
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    
    # Add padding around hand (20% of hand size)
    padding_x = int((x_max - x_min) * 0.3)
    padding_y = int((y_max - y_min) * 0.3)
    
    # Ensure square bounding box (for consistent aspect ratio)
    box_size = max(x_max - x_min + 2 * padding_x, y_max - y_min + 2 * padding_y)
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    # Calculate new box coordinates
    x_min_padded = max(0, x_center - box_size // 2)
    y_min_padded = max(0, y_center - box_size // 2)
    x_max_padded = min(w, x_center + box_size // 2)
    y_max_padded = min(h, y_center + box_size // 2)
    
    # Extract ROI
    hand_roi = frame[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
    
    if hand_roi.size == 0:
        return None
    
    return hand_roi

def predict_letter(finger_state, hand_landmarks, frame=None):
    """Predict letter based on finger state and additional features"""
    # Enhanced rule-based approach with more distinct features
    # Calculate finger angles for additional shape analysis
    finger_angles = get_finger_angles(hand_landmarks)
    
    # Calculate palm orientation
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]  # Middle finger MCP
    palm_direction = [middle_mcp.x - wrist.x, middle_mcp.y - wrist.y]
    palm_angle = np.arctan2(palm_direction[1], palm_direction[0]) * 180 / np.pi
    
    # Calculate distance ratios between fingers (additional feature)
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
    tip_distances = []
    for i in range(len(finger_tips)-1):
        tip1 = hand_landmarks.landmark[finger_tips[i]]
        tip2 = hand_landmarks.landmark[finger_tips[i+1]]
        dist = np.sqrt((tip1.x - tip2.x)**2 + (tip1.y - tip2.y)**2)
        tip_distances.append(dist)
    
    # Special case handling for commonly confused letters
    
    # B vs C vs O (all have similar finger states)
    if finger_state == [1, 1, 1, 1, 1] or sum(finger_state) >= 4:  # All or most fingers extended
        # Calculate average finger angle
        avg_angle = sum(finger_angles) / len(finger_angles)
        
        # Calculate finger tip closeness (for O shape)
        thumb_tip = hand_landmarks.landmark[4]
        other_tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
        tip_to_thumb_distances = [np.sqrt((tip.x - thumb_tip.x)**2 + (tip.y - thumb_tip.y)**2) for tip in other_tips]
        avg_tip_distance = sum(tip_to_thumb_distances) / len(tip_to_thumb_distances)
        
        # Hand shape analysis
        if avg_tip_distance < 0.1:  # Tips close together (O shape)
            return 'O', 95.0
        elif 40 < avg_angle < 90:  # Moderately bent fingers (C shape)
            return 'C', 90.0
        else:  # Straighter fingers (B shape)
            return 'B', 85.0
    
    # D vs X (both use index finger)
    if finger_state == [0, 1, 0, 0, 0]:  # Only index extended
        # Check the angle of the index finger
        index_angle = finger_angles[1]  # Index finger angle
        
        # Check if thumb is touching middle finger (for D)
        thumb_tip = hand_landmarks.landmark[4]
        middle_pip = hand_landmarks.landmark[10]  # Middle finger PIP
        thumb_middle_dist = np.sqrt((thumb_tip.x - middle_pip.x)**2 + (thumb_tip.y - middle_pip.y)**2)
        
        if index_angle > 60:  # Index finger is bent
            return 'X', 90.0
        elif thumb_middle_dist < 0.08:  # Thumb close to middle finger
            return 'D', 95.0
        else:
            # Further disambiguate based on index finger position
            index_tip = hand_landmarks.landmark[8]
            index_mcp = hand_landmarks.landmark[5]
            # If index is pointing more upward than forward
            if abs(index_tip.y - index_mcp.y) > abs(index_tip.x - index_mcp.x):
                return 'D', 85.0
            else:
                return 'X', 85.0
    
    # K vs V (both use index and middle fingers but in different positions)
    if finger_state == [0, 1, 1, 0, 0]:  # Index and middle extended
        # Check if fingers are spread apart (V) or angled (K)
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        index_pip = hand_landmarks.landmark[6]
        middle_pip = hand_landmarks.landmark[10]
        
        # Calculate vectors for each finger
        index_vec = [index_tip.x - index_pip.x, index_tip.y - index_pip.y]
        middle_vec = [middle_tip.x - middle_pip.x, middle_tip.y - middle_pip.y]
        
        # Calculate angle between fingers
        dot_product = index_vec[0]*middle_vec[0] + index_vec[1]*middle_vec[1]
        index_len = np.sqrt(index_vec[0]**2 + index_vec[1]**2)
        middle_len = np.sqrt(middle_vec[0]**2 + middle_vec[1]**2)
        
        if index_len * middle_len == 0:
            angle_between = 0
        else:
            cos_angle = min(max(dot_product / (index_len * middle_len), -1.0), 1.0)
            angle_between = np.arccos(cos_angle) * 180 / np.pi
        
        # Also check lateral distance between fingertips
        tip_distance = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
        
        if angle_between > 25 and tip_distance > 0.1:  # Fingers spread apart
            return 'V', 90.0
        elif angle_between < 25:  # Fingers close/parallel
            return 'U', 90.0
        else:  # Angled fingers (K shape)
            return 'K', 85.0
    
    # Y vs I (pinky only vs thumb and pinky)
    if finger_state == [0, 0, 0, 0, 1]:  # Only pinky extended
        return 'I', 95.0
    elif finger_state == [1, 0, 0, 0, 1]:  # Thumb and pinky extended
        return 'Y', 95.0
    
    # A vs S (both are fist-like, but A has thumb to the side)
    if finger_state == [0, 0, 0, 0, 0]:  # Closed fist
        # Check thumb position
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        index_mcp = hand_landmarks.landmark[5]
        
        # Calculate if thumb is sticking out to the side
        thumb_out = abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_tip.y - thumb_mcp.y)
        
        if thumb_out:
            return 'A', 90.0
        else:
            return 'S', 90.0
    
    # Standard matching for other letters
    best_match = None
    best_score = -1
    
    for letter, template in FINGER_STATES.items():
        # Calculate similarity (count of matching fingers)
        score = sum(1 for a, b in zip(finger_state, template) if a == b)
        
        if score > best_score:
            best_score = score
            best_match = letter
    
    # Return the letter and confidence (as percentage)
    confidence = (best_score / 5) * 100
    return best_match, confidence

def generate_frames():
    """Generate webcam frames with hand tracking and sign language prediction"""
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get finger states
                finger_state = get_finger_state(results.multi_hand_landmarks[0])
                
                # Draw finger state (for debugging)
                for i, extended in enumerate(finger_state):
                    finger_name = ["Thumb", "Index", "Middle", "Ring", "Pinky"][i]
                    state = "Extended" if extended else "Closed"
                    color = (0, 255, 0) if extended else (0, 0, 255)
                    cv2.putText(frame, f"{finger_name}: {state}", (10, 100 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Predict letter using rule-based approach
                letter, confidence = predict_letter(finger_state, results.multi_hand_landmarks[0], frame)
                
                # Draw prediction on frame
                cv2.putText(frame, f"Sign: {letter}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw a hint for the user
                cv2.putText(frame, "Move your hand to get better prediction", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # No hands detected
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Show your hand to the camera", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        cap.release()

# Quiz data structure
QUIZ_DATA = {
    'beginner': {
        'letter_groups': {
            'group1': {  # A-H
                'questions': [
                    {
                        'id': 'b1_1',
                        'image': 'quiz/cartoon/beginner/A.png',
                        'question': 'What letter is being signed?',
                        'options': ['A', 'B', 'C', 'D'],
                        'correct': 'A'
                    },
                    {
                        'id': 'b1_2',
                        'image': 'quiz/cartoon/beginner/B.png',
                        'question': 'What letter is being signed?',
                        'options': ['B', 'E', 'F', 'H'],
                        'correct': 'B'
                    },
                    {
                        'id': 'b1_3',
                        'image': 'quiz/cartoon/beginner/C.png',
                        'question': 'What letter is being signed?',
                        'options': ['C', 'G', 'H', 'K'],
                        'correct': 'C'
                    }
                ]
            },
            'group2': {  # I-P
                'questions': [
                    {
                        'id': 'b2_1',
                        'image': 'quiz/cartoon/beginner/I.png',
                        'question': 'What letter is being signed?',
                        'options': ['I', 'J', 'K', 'L'],
                        'correct': 'I'
                    }
                ]
            },
            'group3': {  # Q-Z
                'questions': [
                    {
                        'id': 'b3_1',
                        'image': 'quiz/cartoon/beginner/Q.png',
                        'question': 'What letter is being signed?',
                        'options': ['Q', 'R', 'S', 'T'],
                        'correct': 'Q'
                    }
                ]
            }
        }
    },
    'intermediate': {
        'questions': [
            {
                'id': 'i1',
                'image': 'quiz/intermediate/hello.jpg',
                'question': 'What word is shown in this sign?',
                'options': ['Hello', 'Goodbye', 'Thanks', 'Please'],
                'correct': 'Hello'
            }
        ]
    },
    'advanced': {
        'questions': [
            {
                'id': 'a1',
                'image': 'quiz/advanced/nice_to_meet_you.jpg',
                'question': 'What sentence is being signed?',
                'options': [
                    'Nice to meet you',
                    'How are you',
                    'What is your name',
                    'Where are you from'
                ],
                'correct': 'Nice to meet you'
            }
        ]
    }
}

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/lesson')
def lesson():
    return render_template('lesson.html')

@app.route('/quiz')
def quiz_levels():
    return render_template('quiz_levels.html')

@app.route('/quiz/<level>', methods=['GET', 'POST'])
def quiz(level):
    if level not in QUIZ_DATA:
        return redirect(url_for('quiz_levels'))

    if request.method == 'POST':
        score = 0
        question_ids = request.form.getlist('question_ids')
        total_questions = len(question_ids)

        for q_id in question_ids:
            user_answer = request.form.get(f'answer_{q_id}')
            correct_answer = request.form.get(f'correct_{q_id}')
            if user_answer == correct_answer:
                score += 1

        percentage = (score / total_questions) * 100
        return render_template('quiz_results.html', score=score, total=total_questions, percentage=percentage, level=level)

    questions = generate_random_quiz(level)
    return render_template('quiz.html', level=level, questions=questions)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/message', methods=['GET', 'POST'])
def message():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        return render_template('message.html', success=True)
    return render_template('message.html')

def generate_random_quiz(level, num_questions=5):
    """Generate a random quiz with specified number of questions."""
    if level == 'beginner':
        all_questions = []
        for group in QUIZ_DATA[level]['letter_groups'].values():
            all_questions.extend(group['questions'])
        if len(all_questions) >= num_questions:
            return random.sample(all_questions, num_questions)
        else:
            return all_questions
    return QUIZ_DATA[level]['questions']

if __name__ == '__main__':
    app.run(debug=False, port=5001)
