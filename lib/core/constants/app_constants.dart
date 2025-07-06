class AppConstants {
  static const String appName = 'MindSync AI';
  static const String appVersion = '1.0.0';
  static const String apiBaseUrl = 'http://192.168.1.100:8000'; // Default IP
  
  // API Endpoints
  static const String analyzeEndpoint = '/analyze';
  static const String sessionsEndpoint = '/sessions';
  static const String sessionDetailEndpoint = '/session';
  static const String healthEndpoint = '/health';
  
  // Storage Keys
  static const String sessionsBoxKey = 'sessions';
  static const String settingsKey = 'settings';
  static const String apiUrlKey = 'api_url';
  
  // Animation Durations
  static const Duration shortAnimation = Duration(milliseconds: 300);
  static const Duration mediumAnimation = Duration(milliseconds: 500);
  static const Duration longAnimation = Duration(milliseconds: 800);
  
  // UI Constants
  static const double borderRadius = 20.0;
  static const double cardElevation = 8.0;
  static const double spacing = 16.0;
  static const double largeSpacing = 24.0;
  static const double hugeSpacing = 32.0;
  
  // Session Constants
  static const Duration recordingDuration = Duration(seconds: 10);
  static const Duration sensorSamplingDuration = Duration(seconds: 5);
  
  // Cognitive States
  static const List<String> cognitiveStates = [
    'Calm',
    'Anxious', 
    'Stressed',
    'Fatigued',
    'Focused',
    'Excited',
    'Neutral',
  ];
  
  // State Colors
  static const Map<String, int> stateColors = {
    'Calm': 0xFF34C759,      // Green
    'Anxious': 0xFFFF9500,   // Orange
    'Stressed': 0xFFFF3B30,  // Red
    'Fatigued': 0xFF8E8E93,  // Gray
    'Focused': 0xFF007AFF,   // Blue
    'Excited': 0xFF5856D6,   // Purple
    'Neutral': 0xFF8E8E93,   // Gray
  };
  
  // Animation Constants for extensions
  static const int fastAnimationMs = 200;
  static const int normalAnimationMs = 300;
  static const int slowAnimationMs = 500;
  static const int splashDurationMs = 2000;
  static const int onboardingAnimationMs = 800;
}

// Extension to add time utilities
extension AnimationDurations on int {
  Duration get ms => Duration(milliseconds: this);
  Duration get seconds => Duration(seconds: this);
}
