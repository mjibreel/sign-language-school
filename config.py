"""
Configuration and settings for Amir Sign Language School
"""

import os
from typing import Dict, List, Any

class AppConfig:
    """Application configuration class"""
    
    def __init__(self):
        self.app_name = "Amir Sign Language School"
        self.version = "1.0.0"
        self.author = "Mohamed Hassan Jibril"
        self.description = "A comprehensive Python Flask web application for learning ASL"
        
        # Flask configuration
        self.flask_config = {
            "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production"),
            "DEBUG": os.environ.get("FLASK_DEBUG", "False").lower() == "true",
            "TESTING": False,
            "HOST": "127.0.0.1",
            "PORT": 5001
        }
        
        # MediaPipe configuration
        self.mediapipe_config = {
            "static_image_mode": False,
            "max_num_hands": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        }
        
        # OpenCV configuration
        self.opencv_config = {
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "fps": 30
        }
        
        # Model configuration
        self.model_config = {
            "model_path": "static/my_model/",
            "weights_file": "weights.bin",
            "metadata_file": "metadata.json",
            "model_file": "model.json"
        }
        
        # Quiz configuration
        self.quiz_config = {
            "difficulty_levels": ["beginner", "intermediate", "advanced"],
            "questions_per_quiz": 10,
            "time_limit": 300,  # 5 minutes
            "passing_score": 70
        }
        
        # Supported ASL signs
        self.supported_signs = {
            "alphabet": {
                "A": "Thumb to the side, other fingers closed",
                "B": "Four fingers up, thumb tucked",
                "C": "Curved hand, all fingers slightly bent in C shape",
                "D": "Index up, thumb touching middle finger",
                "E": "All fingers folded, palm facing forward",
                "F": "Index and thumb touching, other fingers extended",
                "G": "Thumb and index extended in pointing position",
                "H": "Thumb, index, middle extended horizontally",
                "I": "Only pinky extended",
                "K": "Thumb, index and middle in K shape",
                "L": "Thumb and index in L shape",
                "M": "Three middle fingers down, thumb tucked",
                "N": "Index and middle down, thumb tucked",
                "O": "All fingers curved meeting thumb in O shape",
                "P": "Thumb and index extended, index pointing down",
                "Q": "Thumb pointing down",
                "R": "Thumb, index, middle with index and middle crossed",
                "S": "Fist, thumb over fingers",
                "T": "Fist with thumb between index and middle",
                "U": "Index and middle extended parallel",
                "V": "Index and middle extended in V shape",
                "W": "Index, middle, ring extended in W shape",
                "X": "Index bent at middle joint",
                "Y": "Thumb and pinky extended, others closed"
            },
            "numbers": {
                "0": "Fist with thumb over fingers",
                "1": "Index finger extended",
                "2": "Index and middle extended",
                "3": "Index, middle, ring extended",
                "4": "All fingers extended except thumb",
                "5": "All fingers extended",
                "6": "Thumb and pinky extended",
                "7": "Thumb, index, middle extended",
                "8": "Thumb and index extended",
                "9": "Index and thumb extended"
            },
            "phrases": {
                "hello": "Wave hand from side to side",
                "thank_you": "Fingers from chin forward",
                "love": "Crossed arms over chest",
                "please": "Flat hand rubbing in circular motion on chest",
                "sorry": "Fist making circular motion on chest"
            }
        }
        
        # File paths
        self.paths = {
            "dataset": "dataset/",
            "templates": "templates/",
            "static": "static/",
            "models": "static/my_model/",
            "quiz_images": "static/quiz/cartoon/",
            "uploads": "uploads/"
        }
        
        # Security settings
        self.security = {
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "allowed_extensions": [".jpg", ".jpeg", ".png", ".gif"],
            "session_timeout": 3600  # 1 hour
        }
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask configuration"""
        return self.flask_config
    
    def get_mediapipe_config(self) -> Dict[str, Any]:
        """Get MediaPipe configuration"""
        return self.mediapipe_config
    
    def get_opencv_config(self) -> Dict[str, Any]:
        """Get OpenCV configuration"""
        return self.opencv_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.model_config
    
    def get_quiz_config(self) -> Dict[str, Any]:
        """Get quiz configuration"""
        return self.quiz_config
    
    def get_supported_signs(self) -> Dict[str, Any]:
        """Get supported ASL signs"""
        return self.supported_signs
    
    def get_paths(self) -> Dict[str, str]:
        """Get file paths"""
        return self.paths
    
    def get_security(self) -> Dict[str, Any]:
        """Get security settings"""
        return self.security
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check if required directories exist
            for path in self.paths.values():
                if not os.path.exists(path) and not path.endswith('/'):
                    os.makedirs(path, exist_ok=True)
            
            # Validate Flask config
            if not self.flask_config["SECRET_KEY"]:
                return False
            
            # Validate MediaPipe config
            if self.mediapipe_config["min_detection_confidence"] < 0 or self.mediapipe_config["min_detection_confidence"] > 1:
                return False
            
            return True
        except Exception:
            return False

# Global configuration instance
config = AppConfig()

if __name__ == "__main__":
    print("Amir Sign Language School - Configuration")
    print(f"App: {config.app_name} v{config.version}")
    print(f"Author: {config.author}")
    print(f"Description: {config.description}")
    print(f"Configuration Valid: {config.validate_config()}")
    print(f"Supported Signs: {len(config.supported_signs['alphabet'])} letters, {len(config.supported_signs['numbers'])} numbers, {len(config.supported_signs['phrases'])} phrases")
