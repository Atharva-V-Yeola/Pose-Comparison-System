class PoseComparisonApp {
    constructor() {
        this.videoElement = document.getElementById('videoElement');
        this.canvasElement = document.getElementById('canvasElement');
        this.ctx = this.canvasElement.getContext('2d');
        this.stream = null;
        this.isCapturing = false;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Reference image upload
        document.getElementById('uploadBtn').addEventListener('click', () => {
            document.getElementById('referenceInput').click();
        });

        document.getElementById('referenceInput').addEventListener('change', (e) => {
            this.handleReferenceUpload(e.target.files[0]);
        });

        // Camera controls
        document.getElementById('startCameraBtn').addEventListener('click', () => {
            this.startCamera();
        });

        document.getElementById('stopCameraBtn').addEventListener('click', () => {
            this.stopCamera();
        });

        document.getElementById('captureBtn').addEventListener('click', () => {
            this.captureAndCompare();
        });

        // Data export
        document.getElementById('downloadCsvBtn').addEventListener('click', () => {
            this.downloadCsvData();
        });

        document.getElementById('viewDataBtn').addEventListener('click', () => {
            this.viewData();
        });
    }

    async handleReferenceUpload(file) {
        if (!file) return;

        const uploadStatus = document.getElementById('uploadStatus');
        const referencePreview = document.getElementById('referencePreview');

        try {
            uploadStatus.innerHTML = '<div class="loading"></div> Uploading reference image...';
            uploadStatus.className = 'status-message';

            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch('/api/pose/upload_reference', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                uploadStatus.innerHTML = `✓ Reference pose uploaded successfully! (${result.landmarks_count} landmarks detected)`;
                uploadStatus.className = 'status-message status-success';

                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    referencePreview.innerHTML = `<img src="${e.target.result}" alt="Reference pose">`;
                };
                reader.readAsDataURL(file);

                // Enable camera controls
                document.getElementById('startCameraBtn').disabled = false;
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            uploadStatus.innerHTML = `✗ Error: ${error.message}`;
            uploadStatus.className = 'status-message status-error';
        }
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 } 
                } 
            });
            
            this.videoElement.srcObject = this.stream;
            
            // Update button states
            document.getElementById('startCameraBtn').disabled = true;
            document.getElementById('stopCameraBtn').disabled = false;
            document.getElementById('captureBtn').disabled = false;

            this.updateResults('Camera started. Click "Capture & Compare" to analyze your pose.');
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.updateResults('Error accessing camera. Please ensure you have granted camera permissions.');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.videoElement.srcObject = null;
        }

        // Update button states
        document.getElementById('startCameraBtn').disabled = false;
        document.getElementById('stopCameraBtn').disabled = true;
        document.getElementById('captureBtn').disabled = true;

        this.updateResults('Camera stopped.');
    }

    async captureAndCompare() {
        if (!this.stream) return;

        try {
            // Set canvas size to match video
            this.canvasElement.width = this.videoElement.videoWidth;
            this.canvasElement.height = this.videoElement.videoHeight;

            // Draw current frame to canvas
            this.ctx.drawImage(this.videoElement, 0, 0);

            // Convert canvas to base64
            const imageData = this.canvasElement.toDataURL('image/jpeg', 0.8);

            // Send to backend for comparison
            const response = await fetch('/api/pose/compare_pose', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayComparisonResults(result);
            } else {
                throw new Error(result.error || 'Comparison failed');
            }
        } catch (error) {
            console.error('Error during capture and comparison:', error);
            this.updateResults(`Error: ${error.message}`);
        }
    }

    displayComparisonResults(result) {
        const { pose_detected, distance, landmarks_count } = result;

        // Update metrics
        document.getElementById('distanceValue').textContent = 
            distance === Infinity ? '∞' : distance.toFixed(4);
        
        document.getElementById('poseDetectedValue').textContent = 
            pose_detected ? 'Yes' : 'No';

        // Calculate accuracy score
        const accuracy = pose_detected ? Math.max(0, 100 - (distance * 10)) : 0;
        document.getElementById('accuracyValue').textContent = 
            accuracy.toFixed(1) + '%';

        // Update results display
        if (pose_detected) {
            const qualityMessage = this.getPoseQualityMessage(distance);
            this.updateResults(
                `Pose detected! Distance: ${distance.toFixed(4)}, Accuracy: ${accuracy.toFixed(1)}%. ${qualityMessage}`
            );
        } else {
            this.updateResults('No pose detected in the current frame. Please ensure you are visible in the camera.');
        }
    }

    getPoseQualityMessage(distance) {
        if (distance < 0.5) return 'Excellent match! 🎯';
        if (distance < 1.0) return 'Good match! 👍';
        if (distance < 2.0) return 'Fair match. Try adjusting your pose. 🔄';
        return 'Poor match. Please check your pose alignment. ⚠️';
    }

    updateResults(message) {
        document.getElementById('comparisonResults').innerHTML = `<p>${message}</p>`;
    }

    async downloadCsvData() {
        try {
            const response = await fetch('/api/pose/get_csv_data');
            const result = await response.json();

            if (response.ok && result.data.length > 0) {
                this.downloadCsv(result.data);
            } else {
                alert('No data available to download.');
            }
        } catch (error) {
            console.error('Error downloading CSV:', error);
            alert('Error downloading CSV data.');
        }
    }

    downloadCsv(data) {
        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => headers.map(header => row[header]).join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'pose_comparison_data.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    async viewData() {
        const dataDisplay = document.getElementById('dataDisplay');
        
        try {
            const response = await fetch('/api/pose/get_csv_data');
            const result = await response.json();

            if (response.ok) {
                if (result.data.length === 0) {
                    dataDisplay.innerHTML = '<p>No data available yet. Start capturing poses to see data here.</p>';
                } else {
                    dataDisplay.innerHTML = this.formatDataTable(result.data);
                }
                dataDisplay.style.display = 'block';
            } else {
                throw new Error(result.error || 'Failed to fetch data');
            }
        } catch (error) {
            console.error('Error viewing data:', error);
            dataDisplay.innerHTML = `<p>Error loading data: ${error.message}</p>`;
            dataDisplay.style.display = 'block';
        }
    }

    formatDataTable(data) {
        if (data.length === 0) return '<p>No data available.</p>';

        const headers = Object.keys(data[0]);
        const tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
                </thead>
                <tbody>
                    ${data.slice(-10).reverse().map(row => 
                        `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`
                    ).join('')}
                </tbody>
            </table>
            <p><small>Showing last 10 entries (most recent first)</small></p>
        `;
        return tableHtml;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new PoseComparisonApp();
});

