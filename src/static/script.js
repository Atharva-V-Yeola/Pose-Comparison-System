document.addEventListener('DOMContentLoaded', function() {
    const videoUpload = document.getElementById('videoUpload');
    const analyzeVideoButton = document.getElementById('analyzeVideoButton');
    const uploadStatus = document.getElementById('uploadStatus');
    const analyzedVideo = document.getElementById('analyzedVideo');
    const videoStatus = document.getElementById('videoStatus');
    const analysisReport = document.getElementById('analysisReport');
    const downloadCsvButton = document.getElementById('downloadCsvButton');

    let csvDataUrl = null;

    analyzeVideoButton.addEventListener('click', function() {
        const file = videoUpload.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a video file first.';
            return;
        }

        const formData = new FormData();
        formData.append('video', file);

        uploadStatus.textContent = 'Uploading and analyzing video...';
        analyzeVideoButton.disabled = true;

        fetch('/api/video/analyze_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                uploadStatus.textContent = `Error: ${data.error}`;
            } else {
                uploadStatus.textContent = 'Video analysis completed successfully!';
                
                // Display analyzed video
                if (data.analyzed_video_url) {
                    analyzedVideo.src = data.analyzed_video_url;
                    analyzedVideo.style.display = 'block';
                    videoStatus.textContent = 'Analyzed video with pose annotations';
                }

                // Display analysis report
                if (data.report) {
                    analysisReport.textContent = data.report;
                }

                // Setup CSV download
                if (data.csv_data_url) {
                    csvDataUrl = data.csv_data_url;
                    downloadCsvButton.style.display = 'inline-block';
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            uploadStatus.textContent = 'An error occurred during video analysis.';
        })
        .finally(() => {
            analyzeVideoButton.disabled = false;
        });
    });

    downloadCsvButton.addEventListener('click', function() {
        if (csvDataUrl) {
            const link = document.createElement('a');
            link.href = csvDataUrl;
            link.download = 'pose_analysis_data.csv';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });
});

