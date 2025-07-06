import 'session_response.dart';

class SessionHistory {
  final String id;
  final DateTime timestamp;
  final Duration duration;
  final SessionResponse? sessionResponse;

  SessionHistory({
    required this.id,
    required this.timestamp,
    required this.duration,
    this.sessionResponse,
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'timestamp': timestamp.toIso8601String(),
      'duration': duration.inSeconds,
      'sessionResponse': sessionResponse?.toJson(),
    };
  }

  factory SessionHistory.fromJson(Map<String, dynamic> json) {
    return SessionHistory(
      id: json['id'],
      timestamp: DateTime.parse(json['timestamp']),
      duration: Duration(seconds: json['duration'] ?? 0),
      sessionResponse: json['sessionResponse'] != null 
          ? SessionResponse.fromJson(json['sessionResponse'])
          : null,
    );
  }
}
