# Pose Comparison System

A web-based application for real-time pose comparison using computer vision. This system allows users to upload a reference pose image and compare it with live camera feed using MediaPipe and OpenCV technologies.

## Features

- **Reference Pose Upload**: Upload any image containing a human pose as a reference
- **Live Camera Feed**: Real-time camera access for pose capture
- **Pose Detection**: Automatic pose landmark detection using MediaPipe
- **Real-time Comparison**: Compare live poses with the reference pose
- **Data Logging**: Automatic CSV logging of comparison results
- **Web Interface**: User-friendly web interface accessible from any browser
- **Data Export**: Download comparison data as CSV files

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Computer Vision**: OpenCV, MediaPipe
- **Data Processing**: NumPy, Pandas
- **Database**: SQLite (for user management)

## Installation

### Prerequisites

- Python 3.11 or higher
- Webcam or camera device
- Modern web browser with camera access support

### Setup Instructions

1. **Clone or Download the Project**
   ```bash
   # If you have the project files
   cd pose_comparison_app
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python src/main.py
   ```

5. **Access the Application**
   Open your web browser and navigate to: `http://localhost:5000`

## Usage Guide

### Step 1: Upload Reference Pose

1. Click the "Choose Reference Image" button
2. Select an image file containing a clear human pose
3. Wait for the system to process and detect the pose landmarks
4. You should see a success message with the number of detected landmarks

### Step 2: Start Camera

1. Click the "Start Camera" button
2. Allow camera access when prompted by your browser
3. Position yourself in front of the camera
4. Ensure good lighting and clear visibility

### Step 3: Capture and Compare

1. Position yourself to match the reference pose
2. Click "Capture & Compare" to analyze your current pose
3. View the comparison results including:
   - Distance score (lower is better)
   - Accuracy percentage
   - Pose detection status

### Step 4: Export Data

1. Click "View Data" to see recent comparison results
2. Click "Download CSV Data" to export all logged data
3. Use the CSV file for further analysis or record keeping

## API Endpoints

The application provides several REST API endpoints:

### Upload Reference Pose
- **URL**: `/api/pose/upload_reference`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**: `image` (file)
- **Response**: JSON with success/error message

### Compare Pose
- **URL**: `/api/pose/compare_pose`
- **Method**: POST
- **Content-Type**: application/json
- **Parameters**: `image` (base64 encoded image data)
- **Response**: JSON with comparison results

### Get CSV Data
- **URL**: `/api/pose/get_csv_data`
- **Method**: GET
- **Response**: JSON array of logged comparison data

## Data Format

The system logs comparison data in CSV format with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| timestamp | ISO format timestamp | 2025-07-13T06:00:00.000000 |
| pose_detected | Whether pose was detected | true/false |
| distance | Euclidean distance between poses | 1.2345 or "inf" |
| accuracy_score | Accuracy percentage (0-100) | 87.65 |

## Troubleshooting

### Camera Access Issues

If the camera doesn't work:
1. Ensure your browser has camera permissions
2. Check if other applications are using the camera
3. Try refreshing the page
4. Use HTTPS if accessing remotely

### Pose Detection Issues

If poses aren't detected:
1. Ensure good lighting conditions
2. Make sure the full body is visible
3. Avoid cluttered backgrounds
4. Check that the reference image contains a clear human pose

### Performance Issues

For better performance:
1. Use a modern browser (Chrome, Firefox, Safari)
2. Ensure stable internet connection
3. Close unnecessary browser tabs
4. Use adequate hardware specifications

## Development

### Project Structure

```
pose_comparison_app/
├── src/
│   ├── main.py              # Flask application entry point
│   ├── pose_comparison.py   # Core pose detection logic
│   ├── routes/
│   │   ├── pose.py         # Pose comparison API routes
│   │   └── user.py         # User management routes
│   ├── models/
│   │   └── user.py         # Database models
│   ├── static/
│   │   ├── index.html      # Main web interface
│   │   ├── style.css       # Styling
│   │   └── script.js       # Frontend JavaScript
│   └── database/
│       └── app.db          # SQLite database
├── venv/                   # Virtual environment
└── requirements.txt        # Python dependencies
```

### Adding New Features

To extend the application:

1. **Backend Changes**: Modify routes in `src/routes/pose.py`
2. **Frontend Changes**: Update `src/static/` files
3. **Pose Logic**: Enhance `src/pose_comparison.py`
4. **Database**: Modify models in `src/models/`

### Testing

The application includes basic error handling and validation. For production use, consider adding:

- Unit tests for pose comparison logic
- Integration tests for API endpoints
- Frontend testing with tools like Selenium
- Performance testing for concurrent users

## Deployment

### Local Deployment

The application runs locally on port 5000 by default. For production deployment:

1. **Use a Production WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 src.main:app
   ```

2. **Configure Environment Variables**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key
   ```

3. **Set Up Reverse Proxy** (nginx example)
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Cloud Deployment

For cloud deployment, consider:

- **Heroku**: Use the provided Procfile
- **AWS**: Deploy using Elastic Beanstalk or EC2
- **Google Cloud**: Use App Engine or Compute Engine
- **Docker**: Containerize the application

## Security Considerations

- The application uses CORS for cross-origin requests
- File uploads are validated for image types
- Consider adding authentication for production use
- Implement rate limiting for API endpoints
- Use HTTPS in production environments

## License

This project is provided as-is for educational and demonstration purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the browser console for error messages
3. Ensure all dependencies are properly installed
4. Verify camera and browser permissions

## Version History

- **v1.0.0**: Initial release with basic pose comparison functionality
- Features: Reference pose upload, live camera feed, real-time comparison, CSV export

