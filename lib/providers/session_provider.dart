import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:camera/camera.dart';
import '../services/data_collection_service.dart';
import '../services/api_service.dart';
import '../models/session_request.dart';
import '../models/session_response.dart';
import '../models/session_history.dart';
import '../models/motion_data.dart';
import 'dart:convert';
import 'dart:async';

enum SessionState { idle, preparing, ready, running, processing, completed, error }

class SessionProvider extends ChangeNotifier {
  final DataCollectionService _dataService = DataCollectionService.instance;
  final ApiService _apiService = ApiService.instance;
  
  SessionState _sessionState = SessionState.idle;
  SessionResponse? _currentSessionResult;
  List<SessionHistory> _sessionHistory = [];
  String? _errorMessage;
  bool _isLoading = false;
  
  // Session timing
  Timer? _sessionTimer;
  DateTime? _sessionStartTime;
  final Duration _sessionDuration = const Duration(seconds: 25); // 25 seconds for testing
  Duration _remainingTime = const Duration(seconds: 25); // 25 seconds for testing
  
  // Permissions and sensor status
  bool _permissionsGranted = false;
  bool _isCameraActive = false;
  bool _isAudioActive = false;
  bool _isMotionActive = false;
  
  // Data collection
  CameraController? _cameraController;
  MotionData? _currentMotionData;
  double _audioLevel = 0.0;

  // Getters
  SessionState get sessionState => _sessionState;
  SessionResponse? get currentSessionResult => _currentSessionResult;
  List<SessionHistory> get sessionHistory => _sessionHistory;
  String? get errorMessage => _errorMessage;
  bool get isLoading => _isLoading;
  bool get permissionsGranted => _permissionsGranted;
  bool get isCameraActive => _isCameraActive;
  bool get isAudioActive => _isAudioActive;
  bool get isMotionActive => _isMotionActive;
  CameraController? get cameraController => _cameraController;
  MotionData? get currentMotionData => _currentMotionData;
  double get audioLevel => _audioLevel;
  Duration get remainingTime => _remainingTime;
  double get sessionProgress {
    if (_sessionStartTime == null) return 0.0;
    final elapsed = DateTime.now().difference(_sessionStartTime!);
    return (elapsed.inSeconds / _sessionDuration.inSeconds).clamp(0.0, 1.0);
  }

  // Request permissions
  Future<void> requestPermissions() async {
    try {
      _permissionsGranted = await _dataService.requestAllPermissions();
      if (!_permissionsGranted) {
        throw Exception('Required permissions not granted');
      }
      notifyListeners();
    } catch (e) {
      _errorMessage = e.toString();
      _sessionState = SessionState.error;
      notifyListeners();
      rethrow;
    }
  }

  // Prepare session
  Future<void> prepareSession() async {
    _sessionState = SessionState.preparing;
    _errorMessage = null;
    notifyListeners();

    try {
      // Get available cameras
      final cameras = await availableCameras();
      
      // Initialize camera
      await _dataService.initializeCamera(cameras);
      _cameraController = _dataService.cameraController;
      
      // Initialize audio recorder
      await _dataService.initializeAudioRecorder();
      
      // Initialize motion sensors
      await _dataService.initializeMotionSensors();
      
      _sessionState = SessionState.ready;
      notifyListeners();
    } catch (e) {
      _errorMessage = 'Failed to prepare session: $e';
      _sessionState = SessionState.error;
      notifyListeners();
    }
  }

  // Start session
  Future<void> startSession() async {
    debugPrint('DEBUG: startSession() called, current state: $_sessionState');
    if (_sessionState != SessionState.ready) return;
    
    debugPrint('DEBUG: Starting session...');
    _sessionState = SessionState.running;
    _sessionStartTime = DateTime.now();
    _remainingTime = _sessionDuration;
    _isCameraActive = true;
    _isAudioActive = true;
    _isMotionActive = true;
    notifyListeners();

    // Start data collection
    debugPrint('DEBUG: Starting data collection...');
    await _dataService.startDataCollection();
    
    debugPrint('DEBUG: Starting session timer (duration: ${_sessionDuration.inSeconds} seconds)...');
    // Start session timer
    _sessionTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      _remainingTime = _sessionDuration - DateTime.now().difference(_sessionStartTime!);
      
      if (_remainingTime.inSeconds <= 0) {
        debugPrint('DEBUG: Session timer elapsed, calling _completeSession()');
        _completeSession();
      } else {
        // Update sensor data
        _updateSensorData();
        notifyListeners();
      }
    });
  }

  // Pause session
  void pauseSession() {
    if (_sessionState != SessionState.running) return;
    
    _sessionTimer?.cancel();
    _isCameraActive = false;
    _isAudioActive = false;
    _isMotionActive = false;
    _dataService.pauseDataCollection();
    notifyListeners();
  }

  // Stop session
  void stopSession() {
    _sessionTimer?.cancel();
    _dataService.stopDataCollection();
    resetSession();
  }

  // Complete session automatically
  Future<void> _completeSession() async {
    debugPrint('DEBUG: _completeSession() called');
    _sessionTimer?.cancel();
    _isCameraActive = false;
    _isAudioActive = false;
    _isMotionActive = false;
    _sessionState = SessionState.processing;
    notifyListeners();

    try {
      debugPrint('DEBUG: Stopping data collection...');
      // Stop data collection
      await _dataService.stopDataCollection();
      
      // Get collected data separately
      final collectedData = await _dataService.getCollectedData();
      debugPrint('DEBUG: Collected data keys: ${collectedData.keys}');
      
      // Create session request with base64 data
      final sessionRequest = SessionRequest(
        faceImagePath: collectedData['faceImage'],
        audioFilePath: collectedData['audioFile'],
        faceImageBase64: collectedData['faceImageBase64'],
        audioBase64: collectedData['audioBase64'],
        motionData: collectedData['motionData'] ?? [],
        sessionDuration: _sessionDuration,
        timestamp: _sessionStartTime!,
      );

      debugPrint('DEBUG: Sending session request to backend...');
      // Send to backend for analysis
      final response = await _apiService.analyzeSession(sessionRequest);
      debugPrint('DEBUG: Backend response received: ${response.cognitiveState}');
      
      // Store result
      _currentSessionResult = response;
      _sessionState = SessionState.completed;
      
      // Save to history
      await _saveSessionToHistory(response);
      
      debugPrint('DEBUG: Session completed successfully');
      notifyListeners();
    } catch (e) {
      debugPrint('DEBUG: Error in _completeSession: $e');
      _errorMessage = 'Failed to process session: $e';
      _sessionState = SessionState.error;
      notifyListeners();
    }
  }

  // Update sensor data during session
  void _updateSensorData() {
    _currentMotionData = _dataService.getCurrentMotionData();
    _audioLevel = _dataService.getCurrentAudioLevel();
  }

  // Reset session
  void resetSession() {
    _sessionTimer?.cancel();
    _sessionState = SessionState.idle;
    _currentSessionResult = null;
    _errorMessage = null;
    _sessionStartTime = null;
    _remainingTime = _sessionDuration;
    _isCameraActive = false;
    _isAudioActive = false;
    _isMotionActive = false;
    _currentMotionData = null;
    _audioLevel = 0.0;
    
    // Clean up camera controller
    _cameraController?.dispose();
    _cameraController = null;
    
    notifyListeners();
  }

  // Save session to history
  Future<void> _saveSessionToHistory(SessionResponse response) async {
    final sessionHistory = SessionHistory(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      timestamp: _sessionStartTime!,
      duration: _sessionDuration,
      sessionResponse: response,
    );

    _sessionHistory.insert(0, sessionHistory);
    await _saveHistoryToPrefs();
    notifyListeners();
  }

  // Load session history from local storage
  Future<void> loadSessionHistory() async {
    _isLoading = true;
    notifyListeners();
    
    try {
      final prefs = await SharedPreferences.getInstance();
      final historyJson = prefs.getString('session_history');
      
      if (historyJson != null) {
        final List<dynamic> historyList = jsonDecode(historyJson);
        _sessionHistory = historyList
            .map((json) => SessionHistory.fromJson(json))
            .toList();
      }
    } catch (e) {
      debugPrint('Failed to load session history: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  // Save session history to local storage
  Future<void> _saveHistoryToPrefs() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final historyJson = jsonEncode(
        _sessionHistory.map((session) => session.toJson()).toList(),
      );
      await prefs.setString('session_history', historyJson);
    } catch (e) {
      debugPrint('Failed to save session history: $e');
    }
  }

  // Delete a specific session
  Future<void> deleteSession(String sessionId) async {
    _sessionHistory.removeWhere((session) => session.id == sessionId);
    await _saveHistoryToPrefs();
    notifyListeners();
  }

  // Clear all history
  Future<void> clearAllHistory() async {
    _sessionHistory.clear();
    await _saveHistoryToPrefs();
    notifyListeners();
  }

  @override
  void dispose() {
    _sessionTimer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }
}
