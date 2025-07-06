import 'package:flutter/material.dart';
import '../services/api_service.dart';

enum ApiConnectionStatus { unknown, connected, disconnected, checking }

class ApiProvider extends ChangeNotifier {
  final ApiService _apiService = ApiService.instance;
  ApiConnectionStatus _connectionStatus = ApiConnectionStatus.unknown;
  String? _errorMessage;

  ApiConnectionStatus get connectionStatus => _connectionStatus;
  String? get errorMessage => _errorMessage;
  String get baseUrl => _apiService.baseUrl;

  // Check backend connection
  Future<void> checkConnection() async {
    _connectionStatus = ApiConnectionStatus.checking;
    _errorMessage = null;
    notifyListeners();

    try {
      final isConnected = await _apiService.checkConnection();
      _connectionStatus = isConnected 
          ? ApiConnectionStatus.connected 
          : ApiConnectionStatus.disconnected;
      
      if (!isConnected) {
        _errorMessage = 'Unable to connect to backend at ${_apiService.baseUrl}';
      }
    } catch (e) {
      _connectionStatus = ApiConnectionStatus.disconnected;
      _errorMessage = 'Connection error: $e';
    }
    
    notifyListeners();
  }

  // Update backend URL
  void updateBaseUrl(String url) {
    _apiService.updateBaseUrl(url);
    notifyListeners();
  }

  // Get API service instance
  ApiService get apiService => _apiService;
}
