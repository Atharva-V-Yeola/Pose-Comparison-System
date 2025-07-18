from flask import Blueprint, request, jsonify
import os
from src.video_pose_analyzer import VideoPoseAnalyzer
from src.audio_feedback_generator import AudioFeedbackGenerator
import pandas as pd
from default_api import media_generate_speech # Import the tool

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

            # Generate audio feedback
            audio_generator = AudioFeedbackGenerator(voice_type="male_voice") # Pass selected voice type
            audio_files = []
            summary_audio_response = None
            
            if analyzer.feedback_messages:
                # Create audio directory
                audio_dir = os.path.join(upload_folder, "audio")
                os.makedirs(audio_dir, exist_ok=True)
                
                # Generate individual feedback audio files
                for i, msg_data in enumerate(analyzer.feedback_messages):
                    audio_filename = f"feedback_{i+1}_{msg_data["timestamp"]:.2f}s.wav"
                    audio_path = os.path.join(audio_dir, audio_filename)
                    
                    # Call the actual media_generate_speech tool
                    media_generate_speech(brief="Generating feedback audio", path=audio_path, text=msg_data["message"], voice=audio_generator.voice_type)
                    
                    audio_files.append({
                        "message": msg_data["message"],
                        "timestamp": msg_data["timestamp"],
                        "audio_url": f"/uploads/audio/{audio_filename}",
                        "audio_filename": audio_filename
                    })
                
                # Generate summary audio
                summary_audio_data = audio_generator.generate_summary_audio(report, audio_dir)
                summary_audio_path = summary_audio_data["audio_path"]
                summary_audio_filename = summary_audio_data["audio_filename"]
                
                media_generate_speech(brief="Generating summary audio", path=summary_audio_path, text=summary_audio_data["text"], voice=audio_generator.voice_type)

                summary_audio_response = {
                    "text": summary_audio_data["text"],
                    "audio_url": f"/uploads/audio/{summary_audio_filename}",
                    "audio_filename": summary_audio_filename
                }

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
                "distance_events": analyzer.distance_events,
                "feedback_messages": analyzer.feedback_messages,
                "audio_feedback": audio_files,
                "summary_audio": summary_audio_response
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Something went wrong"}), 500

