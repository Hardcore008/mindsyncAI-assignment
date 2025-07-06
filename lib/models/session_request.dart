class SessionRequest {
  final String? faceImagePath;
  final String? audioFilePath;
  final String? faceImageBase64;
  final String? audioBase64;
  final List<Map<String, dynamic>> motionData; // Changed from List<MotionData> to raw sensor data
  final Duration sessionDuration;
  final DateTime timestamp;

  SessionRequest({
    this.faceImagePath,
    this.audioFilePath,
    this.faceImageBase64,
    this.audioBase64,
    required this.motionData,
    required this.sessionDuration,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() {
    return {
      'face_image_path': faceImagePath,
      'audio_file_path': audioFilePath,
      'face_image_base64': faceImageBase64,
      'audio_base64': audioBase64,
      'motion_data': motionData, // Already in correct format
      'session_duration': sessionDuration.inSeconds,
      'timestamp': timestamp.toIso8601String(),
    };
  }
}
