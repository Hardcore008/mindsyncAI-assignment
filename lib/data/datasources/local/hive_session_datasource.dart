import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../../models/session_model.dart';

class HiveSessionDataSource {
  static const String _sessionsKey = 'sessions_data';

  Future<List<SessionModel>> getAllSessions() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final sessionsJson = prefs.getStringList(_sessionsKey) ?? [];
      
      final sessions = sessionsJson
          .map((json) => SessionModel.fromJson(jsonDecode(json)))
          .toList();
      
      // Sort by timestamp (newest first)
      sessions.sort((a, b) => b.timestamp.compareTo(a.timestamp));
      
      return sessions;
    } catch (e) {
      throw Exception('Failed to get sessions from local storage: ${e.toString()}');
    }
  }

  Future<SessionModel?> getSession(String id) async {
    try {
      final sessions = await getAllSessions();
      return sessions.firstWhere(
        (session) => session.id == id,
        orElse: () => throw Exception('Session not found'),
      );
    } catch (e) {
      return null;
    }
  }

  Future<void> saveSession(SessionModel session) async {
    try {
      final sessions = await getAllSessions();
      
      // Remove existing session with same ID
      sessions.removeWhere((s) => s.id == session.id);
      
      // Add new session
      sessions.add(session);
      
      // Save back to preferences
      final prefs = await SharedPreferences.getInstance();
      final sessionsJson = sessions
          .map((session) => jsonEncode(session.toJson()))
          .toList();
      
      await prefs.setStringList(_sessionsKey, sessionsJson);
    } catch (e) {
      throw Exception('Failed to save session to local storage: ${e.toString()}');
    }
  }

  Future<void> deleteSession(String id) async {
    try {
      final sessions = await getAllSessions();
      sessions.removeWhere((session) => session.id == id);
      
      final prefs = await SharedPreferences.getInstance();
      final sessionsJson = sessions
          .map((session) => jsonEncode(session.toJson()))
          .toList();
      
      await prefs.setStringList(_sessionsKey, sessionsJson);
    } catch (e) {
      throw Exception('Failed to delete session from local storage: ${e.toString()}');
    }
  }

  Future<void> clearAllSessions() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_sessionsKey);
    } catch (e) {
      throw Exception('Failed to clear sessions from local storage: ${e.toString()}');
    }
  }
}
