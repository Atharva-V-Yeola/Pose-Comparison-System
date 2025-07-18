from flask import Blueprint, request, jsonify
import os
from src.video_pose_analyzer import VideoPoseAnalyzer
import pandas as pd

video_analysis_bp = Blueprint("video_analysis_bp", __name__)

@video_analysis_bp.route("/analyze_video", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected video file"}), 400

    if video_file:
        # Save the video temporarily
        upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        video_path = os.path.join(upload_folder, video_file.filename)
        video_file.save(video_path)

        analyzer = VideoPoseAnalyzer()
        output_video_filename = f"analyzed_{video_file.filename}"
        output_video_path = os.path.join(upload_folder, output_video_filename)
        csv_output_filename = f"analysis_{os.path.splitext(video_file.filename)[0]}.csv"
        csv_output_path = os.path.join(upload_folder, csv_output_filename)

        try:
            analysis_data = analyzer.process_video(video_path, output_video_path)
            analyzer.save_analysis_to_csv(csv_output_path)
            report = analyzer.generate_analysis_report()
            angle_stats = analyzer.get_angle_statistics()

            # Clean up temporary video files
            os.remove(video_path)
            # os.remove(output_video_path) # Keep analyzed video for now

            return jsonify({
                "message": "Video analysis complete",
                "report": report,
                "angle_statistics": angle_stats,
                "analyzed_video_url": f"/uploads/{output_video_filename}",
                "csv_data_url": f"/uploads/{csv_output_filename}",
                "activity_events": analyzer.activity_events,
                "hit_miss_events": analyzer.hit_miss_events,
                "distance_events": analyzer.distance_events
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Something went wrong"}), 500


