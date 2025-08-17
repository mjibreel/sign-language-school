"""
Utility functions for Amir Sign Language School
"""

def get_app_info():
    """Get application information"""
    return {
        "name": "Amir Sign Language School",
        "version": "1.0.0",
        "description": "A comprehensive Python Flask web application for learning ASL",
        "author": "Mohamed Hassan Jibril",
        "technologies": ["Python", "Flask", "OpenCV", "MediaPipe", "TensorFlow"]
    }

def get_features():
    """Get list of application features"""
    return [
        "Real-time Hand Gesture Recognition",
        "Interactive ASL Lessons",
        "Live Camera Integration",
        "Quiz System",
        "Responsive Design",
        "Dark Mode Support"
    ]

def get_supported_signs():
    """Get list of supported ASL signs"""
    return {
        "alphabet": [chr(i) for i in range(65, 91) if i not in [74, 90]],  # A-Z excluding J and Z
        "numbers": list(range(10)),  # 0-9
        "phrases": ["Hello", "Thank you", "Love", "Please", "Sorry"]
    }

def get_tech_stack():
    """Get technology stack information"""
    return {
        "backend": "Python Flask",
        "computer_vision": ["OpenCV", "MediaPipe"],
        "data_processing": ["Pandas", "NumPy"],
        "frontend": ["HTML", "CSS", "JavaScript"],
        "machine_learning": "TensorFlow Lite"
    }

if __name__ == "__main__":
    print("Amir Sign Language School - Utilities")
    print(f"App Info: {get_app_info()}")
    print(f"Features: {get_features()}")
    print(f"Supported Signs: {get_supported_signs()}")
    print(f"Tech Stack: {get_tech_stack()}")
