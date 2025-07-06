import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../models/session_model.dart';

class SharedPrefsSessionDataSource {
  static const String _sessionsKey = 'sessions_list';
  
  Future<void> saveSession(SessionModel session) async {
    final prefs = await SharedPreferences.getInstance();
    final sessions = await getSessions();
    sessions.add(session);
    
    final sessionsJson = sessions.map((s) => s.toJson()).toList();
    await prefs.setString(_sessionsKey, jsonEncode(sessionsJson));
  }
  
  Future<List<SessionModel>> getSessions() async {
    final prefs = await SharedPreferences.getInstance();
    final sessionsString = prefs.getString(_sessionsKey);
    
    if (sessionsString == null) {
      return [];
    }
    
    try {
      final sessionsList = jsonDecode(sessionsString) as List;
      return sessionsList.map((json) => SessionModel.fromJson(json)).toList();
    } catch (e) {
      debugPrint('Error loading sessions: $e');
      return [];
    }
  }
  
  Future<SessionModel?> getSessionById(String id) async {
    final sessions = await getSessions();
    try {
      return sessions.firstWhere((session) => session.id == id);
    } catch (e) {
      return null;
    }
  }
  
  Future<void> deleteSession(String id) async {
    final prefs = await SharedPreferences.getInstance();
    final sessions = await getSessions();
    sessions.removeWhere((session) => session.id == id);
    
    final sessionsJson = sessions.map((s) => s.toJson()).toList();
    await prefs.setString(_sessionsKey, jsonEncode(sessionsJson));
  }
  
  Future<void> clearAllSessions() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_sessionsKey);
  }
  
  Future<int> getSessionsCount() async {
    final sessions = await getSessions();
    return sessions.length;
  }
  
  Future<List<SessionModel>> getRecentSessions({int limit = 10}) async {
    final sessions = await getSessions();
    sessions.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return sessions.take(limit).toList();
  }
}
