import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:flutter/foundation.dart';
import '../models/motion_data.dart';

class DataCollectionService {
  static final DataCollectionService _instance = DataCollectionService._internal();
  static DataCollectionService get instance => _instance;
  factory DataCollectionService() => _instance;
  DataCollectionService._internal();

  // Camera
  CameraController? _cameraController;
  String? _capturedImagePath;
  String? _faceImageBase64;

  // Audio recording (Enhanced realistic audio simulation)
  bool _isRecording = false;
  String? _audioBase64;
  Timer? _audioTimer;

  // Motion sensors (REAL sensors)
  final List<Map<String, dynamic>> _motionData = [];
  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroscopeSubscription;
  bool _isCollectingMotion = false;
  bool _isPaused = false;

  // Getters
  bool get isRecording => _isRecording;
  String? get capturedImagePath => _capturedImagePath;
  String? get faceImageBase64 => _faceImageBase64;
  String? get audioBase64 => _audioBase64;
  List<Map<String, dynamic>> get motionData => List.unmodifiable(_motionData);

  // Mock getters for compatibility
  MotionData? getCurrentMotionData() {
    if (_motionData.isEmpty) return null;
    final latest = _motionData.last;
    return MotionData(
      x: latest['acceleration']?['x'] ?? 0.0,
      y: latest['acceleration']?['y'] ?? 0.0,
      z: latest['acceleration']?['z'] ?? 0.0,
      timestamp: DateTime.now().millisecondsSinceEpoch,
    );
  }

  double getCurrentAudioLevel() {
    // Simulate audio level
    return _isRecording ? Random().nextDouble() * 0.5 + 0.1 : 0.0;
  }

  /// Request all necessary permissions for data collection
  Future<bool> requestAllPermissions() async {
    try {
      // For now, return true as we'll implement real permissions later
      debugPrint('Permissions requested (using WAV generation)');
      return true;
    } catch (e) {
      debugPrint('Error requesting permissions: $e');
      return false;
    }
  }

  /// Initialize audio recorder
  Future<void> initializeAudioRecorder() async {
    try {
      debugPrint('Audio recorder initialized (using WAV generation)');
    } catch (e) {
      debugPrint('Error initializing audio recorder: $e');
    }
  }

  /// Initialize motion sensors
  Future<void> initializeMotionSensors() async {
    try {
      // Motion sensors are already initialized in _startRealMotionSensors
      // This method ensures they're ready for use
      debugPrint('Motion sensors initialized');
    } catch (e) {
      debugPrint('Error initializing motion sensors: $e');
    }
  }

  /// Pause data collection
  void pauseDataCollection() {
    _isPaused = true;
    _isCollectingMotion = false;
    debugPrint('Data collection paused');
  }

  /// Resume data collection
  void resumeDataCollection() {
    _isPaused = false;
    _isCollectingMotion = true;
    debugPrint('Data collection resumed');
  }

  Future<void> initializeCamera(List<CameraDescription> cameras) async {
    if (cameras.isEmpty) return;

    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _cameraController!.initialize();
  }

  CameraController? get cameraController => _cameraController;

  Future<String?> captureImage() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return null;
    }

    try {
      final XFile imageFile = await _cameraController!.takePicture();
      _capturedImagePath = imageFile.path;
      
      // Convert to base64
      final bytes = await imageFile.readAsBytes();
      _faceImageBase64 = base64Encode(bytes);
      
      debugPrint('Image captured: ${imageFile.path}');
      return imageFile.path;
    } catch (e) {
      debugPrint('Error capturing image: $e');
      return null;
    }
  }

  Future<void> startDataCollection() async {
    debugPrint('Starting data collection...');
    _isRecording = true;
    _isPaused = false;
    _motionData.clear();

    // Start REAL motion sensor collection
    await _startRealMotionSensors();
    
    // Start enhanced audio simulation
    await _startRealAudioRecording();

    debugPrint('Data collection started');
  }

  Future<void> stopDataCollection() async {
    debugPrint('Stopping data collection...');
    _isRecording = false;
    _isPaused = false;
    
    // Stop motion sensors
    await _stopMotionSensors();
    
    // Stop audio recording timer
    _audioTimer?.cancel();
    
    // Capture final image
    await captureImage();

    debugPrint('Data collection stopped');
  }

  Future<Map<String, dynamic>> getCollectedData() async {
    return {
      'faceImage': _capturedImagePath,
      'audioFile': 'simulated_audio.wav',
      'faceImageBase64': _faceImageBase64,
      'audioBase64': _audioBase64,
      'motionData': _motionData,
    };
  }

  // REAL motion sensor implementation
  Future<void> _startRealMotionSensors() async {
    _isCollectingMotion = true;
    debugPrint('Starting REAL motion sensors...');
    
    // Real accelerometer data
    _accelerometerSubscription = accelerometerEvents.listen((AccelerometerEvent event) {
      if (_isCollectingMotion && !_isPaused) {
        _motionData.add({
          'timestamp': DateTime.now().millisecondsSinceEpoch / 1000.0,
          'acceleration': {
            'x': event.x,
            'y': event.y,
            'z': event.z,
          },
          'gyroscope': {
            'x': 0.0, // Will be updated by gyroscope subscription
            'y': 0.0,
            'z': 0.0,
          }
        });
      }
    });

    // Real gyroscope data
    _gyroscopeSubscription = gyroscopeEvents.listen((GyroscopeEvent event) {
      if (_isCollectingMotion && !_isPaused && _motionData.isNotEmpty) {
        // Update the latest motion data entry with gyroscope data
        final latest = _motionData.last;
        latest['gyroscope'] = {
          'x': event.x,
          'y': event.y,
          'z': event.z,
        };
      }
    });

    debugPrint('Real motion sensors started successfully');
  }

  Future<void> _stopMotionSensors() async {
    _isCollectingMotion = false;
    await _accelerometerSubscription?.cancel();
    await _gyroscopeSubscription?.cancel();
    debugPrint('Motion sensors stopped. Collected ${_motionData.length} samples');
  }

  // Enhanced audio simulation (until real recording is implemented)
  // This generates SOPHISTICATED WAV audio data that mimics speech patterns
  Future<void> _startRealAudioRecording() async {
    debugPrint('Starting enhanced audio recording simulation with speech-like patterns...');
    
    try {
      // Generate more sophisticated audio that mimics human speech patterns
      const int sampleRate = 22050; // Higher quality audio
      const int durationSeconds = 5; // 5 seconds of audio
      const int totalSamples = sampleRate * durationSeconds;
      
      // Generate speech-like audio waveform with varying patterns
      final List<int> audioSamples = [];
      final random = Random();
      
      for (int i = 0; i < totalSamples; i++) {
        double time = i / sampleRate;
        double wave = 0.0;
        
        // Simulate speech formants and characteristics
        // Fundamental frequency modulation (like human voice pitch)
        double pitchModulation = 1.0 + 0.3 * sin(2 * pi * 2 * time); // Slow pitch variation
        double fundamentalFreq = 120 * pitchModulation; // Base around 120Hz (typical male voice)
        
        // Add speech formants (resonant frequencies that characterize vowels)
        wave += 0.4 * sin(2 * pi * fundamentalFreq * time);           // Fundamental
        wave += 0.25 * sin(2 * pi * fundamentalFreq * 2 * time);      // Second harmonic
        wave += 0.15 * sin(2 * pi * fundamentalFreq * 3 * time);      // Third harmonic
        
        // Add formant frequencies (simulating vowel sounds)
        wave += 0.2 * sin(2 * pi * 700 * time);  // First formant (F1)
        wave += 0.15 * sin(2 * pi * 1200 * time); // Second formant (F2)
        wave += 0.1 * sin(2 * pi * 2500 * time);  // Third formant (F3)
        
        // Add consonant-like noise bursts
        if (random.nextDouble() < 0.02) { // 2% chance for consonant burst
          wave += 0.5 * (random.nextDouble() - 0.5);
        }
        
        // Add breath noise and natural variation
        wave += 0.05 * (random.nextDouble() - 0.5);
        
        // Apply amplitude envelope (speech-like volume variation)
        double envelope = 0.5 + 0.5 * sin(2 * pi * 0.5 * time + random.nextDouble()); // Varying amplitude
        wave *= envelope;
        
        // Apply some resonance filtering to make it more speech-like
        if (i > 0) {
          wave = 0.7 * wave + 0.3 * (audioSamples[audioSamples.length - 2] / 32767.0);
        }
        
        // Convert to 16-bit PCM with proper clipping
        int sample = (wave * 20000).round().clamp(-32768, 32767); // Lower amplitude to avoid clipping
        audioSamples.add(sample & 0xFF);        // Low byte
        audioSamples.add((sample >> 8) & 0xFF); // High byte
      }
      
      // Create WAV header
      final wavData = _createWavFile(audioSamples, sampleRate, 1); // 1 channel (mono)
      
      // Convert to base64
      _audioBase64 = base64Encode(wavData);
      
      debugPrint('Generated sophisticated speech-like WAV audio: ${wavData.length} bytes');
      debugPrint('Audio base64 length: ${_audioBase64!.length}');
      debugPrint('Sample rate: ${sampleRate}Hz, Duration: ${durationSeconds}s');
      
      // Start a timer to simulate recording progress
      _audioTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
        if (!_isRecording) {
          timer.cancel();
        }
        // Could emit progress updates here if needed
      });
      
    } catch (e) {
      debugPrint('Error generating sophisticated audio data: $e');
      // Fallback to minimal valid audio
      _audioBase64 = _generateMinimalWavBase64();
    }
  }
  
  /// Create a proper WAV file from PCM audio samples
  List<int> _createWavFile(List<int> pcmData, int sampleRate, int channels) {
    final int byteRate = sampleRate * channels * 2; // 16-bit = 2 bytes per sample
    final int dataSize = pcmData.length;
    final int fileSize = 36 + dataSize;
    
    final List<int> wavHeader = [
      // RIFF header
      0x52, 0x49, 0x46, 0x46, // "RIFF"
      fileSize & 0xFF, (fileSize >> 8) & 0xFF, (fileSize >> 16) & 0xFF, (fileSize >> 24) & 0xFF,
      0x57, 0x41, 0x56, 0x45, // "WAVE"
      
      // fmt chunk
      0x66, 0x6D, 0x74, 0x20, // "fmt "
      0x10, 0x00, 0x00, 0x00, // chunk size (16)
      0x01, 0x00,             // audio format (PCM)
      channels & 0xFF, (channels >> 8) & 0xFF, // number of channels
      sampleRate & 0xFF, (sampleRate >> 8) & 0xFF, (sampleRate >> 16) & 0xFF, (sampleRate >> 24) & 0xFF,
      byteRate & 0xFF, (byteRate >> 8) & 0xFF, (byteRate >> 16) & 0xFF, (byteRate >> 24) & 0xFF,
      (channels * 2) & 0xFF, ((channels * 2) >> 8) & 0xFF, // block align
      0x10, 0x00,             // bits per sample (16)
      
      // data chunk
      0x64, 0x61, 0x74, 0x61, // "data"
      dataSize & 0xFF, (dataSize >> 8) & 0xFF, (dataSize >> 16) & 0xFF, (dataSize >> 24) & 0xFF,
    ];
    
    return [...wavHeader, ...pcmData];
  }
  
  /// Generate minimal valid WAV as fallback
  String _generateMinimalWavBase64() {
    // Generate 1 second of silence as valid WAV
    const int sampleRate = 16000;
    const int samples = sampleRate;
    final List<int> audioData = List.filled(samples * 2, 0); // 16-bit silence
    
    final wavData = _createWavFile(audioData, sampleRate, 1);
    return base64Encode(wavData);
  }

  void dispose() {
    _accelerometerSubscription?.cancel();
    _gyroscopeSubscription?.cancel();
    _audioTimer?.cancel();
    _cameraController?.dispose();
  }
}


