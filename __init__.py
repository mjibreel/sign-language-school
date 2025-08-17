"""
Amir Sign Language School - A comprehensive Python Flask web application for learning ASL

This package provides:
- Real-time hand gesture recognition using MediaPipe and OpenCV
- Interactive ASL lessons with live camera integration
- Quiz system with multiple difficulty levels
- Modern web interface with dark mode support

Author: Mohamed Hassan Jibril
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Mohamed Hassan Jibril"
__email__ = "m.h.jibreel@gmail.com"
__description__ = "A comprehensive Python Flask web application for learning American Sign Language (ASL)"

# Import main components
from .config import config
from .utils import get_app_info, get_features, get_supported_signs, get_tech_stack

# Package level exports
__all__ = [
    "config",
    "get_app_info", 
    "get_features",
    "get_supported_signs",
    "get_tech_stack"
]

# Package metadata
PACKAGE_INFO = {
    "name": "amir_sign_language_school",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/mjibreel/sign-language-school",
    "license": "MIT",
    "python_requires": ">=3.11",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video :: Display",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Computer Vision"
    ]
}

def get_package_info():
    """Get comprehensive package information"""
    return PACKAGE_INFO.copy()

def get_installation_instructions():
    """Get installation instructions"""
    return """
Installation:
1. Clone the repository: git clone https://github.com/mjibreel/sign-language-school.git
2. Navigate to directory: cd sign-language-school
3. Create virtual environment: python -m venv venv
4. Activate virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)
5. Install dependencies: pip install -r requirements.txt
6. Run the application: python app.py
7. Open browser: http://127.0.0.1:5001
"""

def get_quick_start():
    """Get quick start guide"""
    return """
Quick Start:
1. Ensure you have Python 3.11+ installed
2. Install dependencies: pip install -r requirements.txt
3. Run: python app.py
4. Open: http://127.0.0.1:5001
5. Navigate to Lesson section and allow camera access
6. Start learning ASL with real-time hand recognition!
"""

# Package initialization
print(f"Initializing {PACKAGE_INFO['name']} v{PACKAGE_INFO['version']}")
print(f"Author: {PACKAGE_INFO['author']}")
print(f"Description: {PACKAGE_INFO['description']}")
print("Package initialized successfully!")
