import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';

class ApiDataSource {
  static const String _baseUrl = 'http://192.168.29.73:8000';
  final HttpClient _httpClient;

  ApiDataSource({HttpClient? httpClient}) : _httpClient = httpClient ?? HttpClient();

  Future<Map<String, dynamic>> getHealth() async {
    return {'status': 'ok'};
  }

  Future<bool> checkConnection() async {
    try {
      final request = await _httpClient.getUrl(Uri.parse('$_baseUrl/health'));
      request.headers.set('Content-Type', 'application/json');
      
      final response = await request.close().timeout(const Duration(seconds: 10));
      
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  Future<Map<String, dynamic>> analyzeSession({
    required String faceImageBase64,
    required String audioBase64,
    required List<Map<String, dynamic>> motionData,
  }) async {
    try {
      // Transform motion data to backend format (motion data is already in correct format from sensors)
      final transformedMotionData = motionData.map((data) {
        return {
          'timestamp': (data['timestamp'] ?? DateTime.now().millisecondsSinceEpoch / 1000.0).toDouble(),
          'acceleration': data['acceleration'] ?? {'x': 0.0, 'y': 0.0, 'z': 0.0},
          'gyroscope': data['gyroscope'] ?? {'x': 0.0, 'y': 0.0, 'z': 0.0},
        };
      }).toList();

      final requestBody = {
        'face_image': faceImageBase64,
        'audio_data': audioBase64,
        'motion_data': transformedMotionData,
        'timestamp': DateTime.now().toIso8601String(),
        'duration_seconds': 180.0, // 3 minutes
      };

      debugPrint('🚀 Making API request to $_baseUrl/analyze');
      debugPrint('📊 Request body size: ${jsonEncode(requestBody).length} characters');

      final request = await _httpClient.postUrl(Uri.parse('$_baseUrl/analyze'));
      request.headers.set('Content-Type', 'application/json');
      request.headers.set('Accept', 'application/json');
      
      request.write(jsonEncode(requestBody));
      
      final response = await request.close().timeout(const Duration(seconds: 30));
      
      debugPrint('📡 Response status: ${response.statusCode}');
      
      final responseBody = await response.transform(utf8.decoder).join();
      debugPrint('📄 Response body: $responseBody');

      if (response.statusCode == 200) {
        final result = jsonDecode(responseBody);
        debugPrint('✅ Analysis successful: ${result.keys}');
        return result;
      } else {
        throw Exception('Analysis failed with status ${response.statusCode}: $responseBody');
      }
    } catch (e) {
      debugPrint('❌ Analysis request failed: $e');
      rethrow; // Don't use fallback data, let the error propagate
    }
  }

  Future<List<Map<String, dynamic>>> getSessions() async {
    // TODO: Implement real API call
    return [];
  }

  Future<Map<String, dynamic>?> getSession(String id) async {
    // TODO: Implement real API call
    return null;
  }
}
