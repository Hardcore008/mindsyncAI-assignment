import 'package:equatable/equatable.dart';

class SessionModel extends Equatable {
  final String id;
  final DateTime timestamp;
  final String cognitiveState;
  final double confidence;
  final List<String> insights;
  final List<String> recommendations;
  final double stressLevel;
  final double attentionScore;
  final double overallScore;
  final Map<String, dynamic> emotionScores;
  final String? chartData;
  final bool isSynced;
  final Map<String, dynamic>? analysisResult;
  final int? duration;

  const SessionModel({
    required this.id,
    required this.timestamp,
    required this.cognitiveState,
    required this.confidence,
    required this.insights,
    required this.recommendations,
    required this.stressLevel,
    required this.attentionScore,
    required this.overallScore,
    required this.emotionScores,
    this.chartData,
    this.isSynced = false,
    this.analysisResult,
    this.duration,
  });

  factory SessionModel.fromJson(Map<String, dynamic> json) {
    return SessionModel(
      id: json['session_id'] ?? '',
      timestamp: DateTime.parse(json['timestamp'] ?? DateTime.now().toIso8601String()),
      cognitiveState: json['cognitive_state'] ?? 'Neutral',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      insights: List<String>.from(json['insights'] ?? []),
      recommendations: List<String>.from(json['recommendations'] ?? []),
      stressLevel: (json['stress_level'] ?? 0.0).toDouble(),
      attentionScore: (json['attention_score'] ?? 0.0).toDouble(),
      overallScore: (json['overall_score'] ?? 0.0).toDouble(),
      emotionScores: Map<String, dynamic>.from(json['emotion_scores'] ?? {}),
      chartData: json['chart_data'],
      isSynced: json['is_synced'] ?? false,
      analysisResult: json['analysis_result'] != null ? Map<String, dynamic>.from(json['analysis_result']) : null,
      duration: json['duration']?.toInt(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'session_id': id,
      'timestamp': timestamp.toIso8601String(),
      'cognitive_state': cognitiveState,
      'confidence': confidence,
      'insights': insights,
      'recommendations': recommendations,
      'stress_level': stressLevel,
      'attention_score': attentionScore,
      'overall_score': overallScore,
      'emotion_scores': emotionScores,
      'chart_data': chartData,
      'is_synced': isSynced,
      'analysis_result': analysisResult,
      'duration': duration,
    };
  }

  SessionModel copyWith({
    String? id,
    DateTime? timestamp,
    String? cognitiveState,
    double? confidence,
    List<String>? insights,
    List<String>? recommendations,
    double? stressLevel,
    double? attentionScore,
    double? overallScore,
    Map<String, dynamic>? emotionScores,
    String? chartData,
    bool? isSynced,
    Map<String, dynamic>? analysisResult,
    int? duration,
  }) {
    return SessionModel(
      id: id ?? this.id,
      timestamp: timestamp ?? this.timestamp,
      cognitiveState: cognitiveState ?? this.cognitiveState,
      confidence: confidence ?? this.confidence,
      insights: insights ?? this.insights,
      recommendations: recommendations ?? this.recommendations,
      stressLevel: stressLevel ?? this.stressLevel,
      attentionScore: attentionScore ?? this.attentionScore,
      overallScore: overallScore ?? this.overallScore,
      emotionScores: emotionScores ?? this.emotionScores,
      chartData: chartData ?? this.chartData,
      isSynced: isSynced ?? this.isSynced,
      analysisResult: analysisResult ?? this.analysisResult,
      duration: duration ?? this.duration,
    );
  }

  @override
  List<Object?> get props => [
        id,
        timestamp,
        cognitiveState,
        confidence,
        insights,
        recommendations,
        stressLevel,
        attentionScore,
        overallScore,
        emotionScores,
        chartData,
        isSynced,
        analysisResult,
        duration,
      ];
}
