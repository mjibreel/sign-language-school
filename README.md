# Amir Sign Language School

A comprehensive web application for learning American Sign Language (ASL) through interactive lessons, real-time hand gesture recognition, and quizzes.

## 🚀 Features

- **Real-time Hand Gesture Recognition**: Uses MediaPipe and OpenCV to detect and interpret hand signs
- **Interactive Lessons**: Learn ASL alphabet and common phrases with visual guides
- **Live Camera Integration**: Practice signs with real-time feedback
- **Quiz System**: Test your knowledge with interactive quizzes
- **Responsive Design**: Modern, user-friendly interface that works on all devices
- **Dark Mode Support**: Beautiful dark theme for better user experience

## 🛠️ Technologies Used

- **Backend**: Python Flask
- **Computer Vision**: OpenCV, MediaPipe
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: TensorFlow Lite (via MediaPipe)

## 📋 Prerequisites

- Python 3.11 or higher
- Webcam for hand gesture recognition
- Modern web browser

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/amir-sign-language-school.git
   cd amir-sign-language-school
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://127.0.0.1:5001`

## 📚 How to Use

### Learning Mode
1. Navigate to the **Lesson** section
2. Allow camera access when prompted
3. Show hand gestures to the camera
4. The system will recognize and display the corresponding ASL letter/phrase

### Quiz Mode
1. Go to the **Quiz** section
2. Choose your difficulty level
3. Answer questions about ASL signs
4. Track your progress and scores

## 🏗️ Project Structure

```
amir-sign-language-school/
├── app.py                 # Main Flask application
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Homepage
│   ├── lesson.html       # Lesson page
│   ├── quiz.html         # Quiz page
│   └── about.html        # About page
├── static/               # Static files
│   ├── css/             # Stylesheets
│   ├── img/             # Images and assets
│   └── quiz/            # Quiz resources
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

The application can be configured through environment variables:

- `FLASK_ENV`: Set to 'development' for debug mode
- `FLASK_DEBUG`: Enable/disable debug mode

## 📱 Supported ASL Signs

The application currently supports:
- **Alphabet**: A-Z (excluding J and Z which require motion)
- **Numbers**: 0-9
- **Common Phrases**: Hello, Thank you, Love, etc.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for hand tracking capabilities
- OpenCV community for computer vision tools
- Flask community for the web framework
- ASL community for sign language resources

## 📞 Contact

- **Project Link**: [https://github.com/yourusername/amir-sign-language-school](https://github.com/yourusername/amir-sign-language-school)
- **Issues**: [https://github.com/yourusername/amir-sign-language-school/issues](https://github.com/yourusername/amir-sign-language-school/issues)

## 🎯 Future Enhancements

- [ ] Support for dynamic signs (J, Z, etc.)
- [ ] Video lessons and tutorials
- [ ] Progress tracking and user accounts
- [ ] Mobile app version
- [ ] Multiple sign language support
- [ ] Advanced gesture recognition
- [ ] Social learning features

---

⭐ **Star this repository if you find it helpful!**
