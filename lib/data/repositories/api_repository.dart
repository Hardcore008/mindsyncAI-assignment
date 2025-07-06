import '../models/session_model.dart';
import '../datasources/remote/api_data_source.dart';

class ApiRepository {
  final ApiDataSource apiDataSource;

  ApiRepository({
    required this.apiDataSource,
  });

  Future<bool> checkHealth() async {
    try {
      final health = await apiDataSource.getHealth();
      return health['status'] == 'healthy';
    } catch (e) {
      return false;
    }
  }

  Future<SessionModel> analyzeSession({
    required String faceImageBase64,
    required String audioBase64,
    required List<Map<String, dynamic>> motionData,
  }) async {
    try {
      final response = await apiDataSource.analyzeSession(
        faceImageBase64: faceImageBase64,
        audioBase64: audioBase64,
        motionData: motionData,
      );
      
      // Convert Map response to SessionModel
      return SessionModel(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        timestamp: DateTime.now(),
        cognitiveState: response['cognitive_state'] ?? 'Neutral',
        confidence: response['confidence']?.toDouble() ?? 0.5,
        insights: List<String>.from(response['insights'] ?? ['Analysis completed']),
        recommendations: List<String>.from(response['recommendations'] ?? ['Review your session']),
        stressLevel: response['stress_level']?.toDouble() ?? 0.0,
        attentionScore: response['attention_score']?.toDouble() ?? 0.0,
        overallScore: response['overall_score']?.toDouble() ?? 0.5,
        emotionScores: Map<String, dynamic>.from(response['emotion_scores'] ?? {}),
        chartData: response['chart_data']?.toString(),
        isSynced: true,
        analysisResult: response,
      );
    } catch (e) {
      throw Exception('Session analysis failed: ${e.toString()}');
    }
  }

  Future<List<SessionModel>> getRemoteSessions() async {
    try {
      final responses = await apiDataSource.getSessions();
      
      // Convert each Map to SessionModel
      return responses.map((response) => SessionModel(
        id: response['id']?.toString() ?? DateTime.now().millisecondsSinceEpoch.toString(),
        timestamp: DateTime.tryParse(response['timestamp'] ?? '') ?? DateTime.now(),
        cognitiveState: response['cognitive_state'] ?? 'Neutral',
        confidence: response['confidence']?.toDouble() ?? 0.5,
        insights: List<String>.from(response['insights'] ?? ['Session data']),
        recommendations: List<String>.from(response['recommendations'] ?? ['No recommendations']),
        stressLevel: response['stress_level']?.toDouble() ?? 0.0,
        attentionScore: response['attention_score']?.toDouble() ?? 0.0,
        overallScore: response['overall_score']?.toDouble() ?? 0.5,
        emotionScores: Map<String, dynamic>.from(response['emotion_scores'] ?? {}),
        chartData: response['chart_data']?.toString(),
        isSynced: true,
        analysisResult: response,
      )).toList();
    } catch (e) {
      throw Exception('Failed to get remote sessions: ${e.toString()}');
    }
  }

  Future<SessionModel> getRemoteSession(String id) async {
    try {
      final response = await apiDataSource.getSession(id);
      if (response == null) {
        throw Exception('Session not found');
      }
      
      // Convert Map response to SessionModel
      return SessionModel(
        id: id,
        timestamp: DateTime.tryParse(response['timestamp'] ?? '') ?? DateTime.now(),
        cognitiveState: response['cognitive_state'] ?? 'Neutral',
        confidence: response['confidence']?.toDouble() ?? 0.5,
        insights: List<String>.from(response['insights'] ?? ['Session data']),
        recommendations: List<String>.from(response['recommendations'] ?? ['No recommendations']),
        stressLevel: response['stress_level']?.toDouble() ?? 0.0,
        attentionScore: response['attention_score']?.toDouble() ?? 0.0,
        overallScore: response['overall_score']?.toDouble() ?? 0.5,
        emotionScores: Map<String, dynamic>.from(response['emotion_scores'] ?? {}),
        chartData: response['chart_data']?.toString(),
        isSynced: true,
        analysisResult: response,
      );
    } catch (e) {
      throw Exception('Failed to get remote session: ${e.toString()}');
    }
  }
}
