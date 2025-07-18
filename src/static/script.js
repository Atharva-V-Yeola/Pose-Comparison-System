document.addEventListener("DOMContentLoaded", function() {
    const videoUpload = document.getElementById("videoUpload");
    const analyzeVideoButton = document.getElementById("analyzeVideoButton");
    const uploadStatus = document.getElementById("uploadStatus");
    const analyzedVideo = document.getElementById("analyzedVideo");
    const videoStatus = document.getElementById("videoStatus");
    const analysisReport = document.getElementById("analysisReport");
    const downloadCsvButton = document.getElementById("downloadCsvButton");
    const activityTimingReport = document.getElementById("activityTimingReport");
    const hitMissReport = document.getElementById("hitMissReport");
    const distanceReport = document.getElementById("distanceReport");

    let csvDataUrl = null;

    analyzeVideoButton.addEventListener("click", function() {
        const file = videoUpload.files[0];
        if (!file) {
            uploadStatus.textContent = "Please select a video file first.";
            return;
        }

        const formData = new FormData();
        formData.append("video", file);

        uploadStatus.textContent = "Uploading and analyzing video...";
        analyzeVideoButton.disabled = true;
        analysisReport.textContent = "";
        activityTimingReport.innerHTML = "";
        hitMissReport.innerHTML = "";
        distanceReport.innerHTML = "";
        analyzedVideo.style.display = "none";
        videoStatus.textContent = "";
        downloadCsvButton.style.display = "none";

        fetch("/api/video/analyze_video", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                uploadStatus.textContent = `Error: ${data.error}`;
            } else {
                uploadStatus.textContent = "Video analysis completed successfully!";
                
                // Display analyzed video
                if (data.analyzed_video_url) {
                    analyzedVideo.src = data.analyzed_video_url;
                    analyzedVideo.style.display = "block";
                    videoStatus.textContent = "Analyzed video with pose annotations";
                }

                // Display analysis report
                if (data.report) {
                    analysisReport.textContent = data.report;
                }

                // Display activity timing
                if (data.activity_events) {
                    let html = "";
                    for (const activityId in data.activity_events) {
                        const event = data.activity_events[activityId];
                        if (event.start_frame !== null && event.end_frame !== null) {
                            html += `<p><strong>${activityId.replace(/_/g, ' ').toUpperCase()}:</strong> ${event.duration.toFixed(2)} seconds</p>`;
                        } else {
                            html += `<p><strong>${activityId.replace(/_/g, ' ').toUpperCase()}:</strong> Not completed or detected.</p>`;
                        }
                    }
                    activityTimingReport.innerHTML = html;
                }

                // Display hit/miss
                if (data.hit_miss_events) {
                    let html = "";
                    for (const hitEventId in data.hit_miss_events) {
                        const event = data.hit_miss_events[hitEventId];
                        html += `<p><strong>${hitEventId.replace(/_/g, ' ').toUpperCase()}:</strong> ${event.hit_detected ? 'Detected' : 'Not Detected'}</p>`;
                    }
                    hitMissReport.innerHTML = html;
                }

                // Display distance measurement
                if (data.distance_events) {
                    let html = "";
                    for (const distanceEventId in data.distance_events) {
                        const event = data.distance_events[distanceEventId];
                        if (event.start_point && event.end_point) {
                            html += `<p><strong>${distanceEventId.replace(/_/g, ' ').toUpperCase()}:</strong> ${event.distance.toFixed(2)} pixels</p>`;
                        } else if (event.current_point) {
                            html += `<p><strong>${distanceEventId.replace(/_/g, ' ').toUpperCase()}:</strong> Current X: ${event.current_point[0].toFixed(2)}, Y: ${event.current_point[1].toFixed(2)}</p>`;
                        }
                    }
                    distanceReport.innerHTML = html;
                }

                // Setup CSV download
                if (data.csv_data_url) {
                    csvDataUrl = data.csv_data_url;
                    downloadCsvButton.style.display = "inline-block";
                }
            }
        })
        .catch(error => {
            console.error("Error:", error);
            uploadStatus.textContent = "An error occurred during video analysis.";
        })
        .finally(() => {
            analyzeVideoButton.disabled = false;
        });
    });

    downloadCsvButton.addEventListener("click", function() {
        if (csvDataUrl) {
            const link = document.createElement("a");
            link.href = csvDataUrl;
            link.download = "pose_analysis_data.csv";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });
});

