class SessionResponse {
  final String cognitiveState;
  final double confidence;
  final List<String> insights;
  final List<String> recommendations;
  final Map<String, dynamic> metrics;
  final dynamic graphData; // Can be structured data or base64 image
  final String? sessionId;

  SessionResponse({
    required this.cognitiveState,
    required this.confidence,
    required this.insights,
    required this.recommendations,
    required this.metrics,
    this.graphData,
    this.sessionId,
  });

  factory SessionResponse.fromJson(Map<String, dynamic> json) {
    return SessionResponse(
      cognitiveState: json['cognitive_state'] ?? 'Unknown',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      insights: json['insights'] is List 
          ? List<String>.from(json['insights'])
          : [json['insights']?.toString() ?? 'No insights available'],
      recommendations: json['recommendations'] is List 
          ? List<String>.from(json['recommendations'])
          : [json['recommendations']?.toString() ?? 'No recommendations available'],
      metrics: json['metrics'] is Map 
          ? Map<String, dynamic>.from(json['metrics'])
          : {},
      graphData: json['graph_data'],
      sessionId: json['session_id'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'cognitive_state': cognitiveState,
      'confidence': confidence,
      'insights': insights,
      'recommendations': recommendations,
      'metrics': metrics,
      'graph_data': graphData,
      'session_id': sessionId,
    };
  }
}
