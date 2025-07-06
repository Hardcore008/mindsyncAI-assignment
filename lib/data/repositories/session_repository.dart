import '../models/session_model.dart';
import '../datasources/local/shared_prefs_session_datasource.dart';
import '../datasources/remote/api_data_source.dart';

class SessionRepository {
  final SharedPrefsSessionDataSource localDataSource;
  final ApiDataSource remoteDataSource;

  SessionRepository({
    required this.localDataSource,
    required this.remoteDataSource,
  });

  Future<List<SessionModel>> getAllSessions() async {
    try {
      return await localDataSource.getSessions();
    } catch (e) {
      throw Exception('Failed to get sessions: ${e.toString()}');
    }
  }

  Future<SessionModel?> getSession(String id) async {
    try {
      return await localDataSource.getSessionById(id);
    } catch (e) {
      throw Exception('Failed to get session: ${e.toString()}');
    }
  }

  Future<void> saveSession(SessionModel session) async {
    try {
      await localDataSource.saveSession(session);
    } catch (e) {
      throw Exception('Failed to save session: ${e.toString()}');
    }
  }

  Future<void> deleteSession(String id) async {
    try {
      await localDataSource.deleteSession(id);
    } catch (e) {
      throw Exception('Failed to delete session: ${e.toString()}');
    }
  }

  Future<void> clearAllSessions() async {
    try {
      await localDataSource.clearAllSessions();
    } catch (e) {
      throw Exception('Failed to clear sessions: ${e.toString()}');
    }
  }
}
