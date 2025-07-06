import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'dart:async';
import 'dart:convert';
import 'dart:math';

import '../../../core/theme/app_theme.dart';
import '../../../core/constants/app_constants.dart';
import '../../../providers/session_provider.dart';
import '../session/result/result_screen.dart';

class SessionCaptureScreen extends StatefulWidget {
  const SessionCaptureScreen({super.key});

  @override
  State<SessionCaptureScreen> createState() => _SessionCaptureScreenState();
}

class _SessionCaptureScreenState extends State<SessionCaptureScreen>
    with TickerProviderStateMixin {
  CameraController? _cameraController;
  late AnimationController _pulseController;
  late AnimationController _captureController;
  
  bool _isRecording = false;
  bool _isInitialized = false;
  int _recordingSeconds = 0;
  Timer? _recordingTimer;
  
  // Simplified motion data
  final List<Map<String, double>> _accelerometerData = [];
  final List<Map<String, double>> _gyroscopeData = [];

  @override
  void initState() {
    super.initState();
    _initializeAnimations();
    _requestPermissions();
  }

  void _initializeAnimations() {
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    
    _captureController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
  }

  Future<void> _requestPermissions() async {
    final permissions = [
      Permission.camera,
      Permission.microphone,
    ];

    Map<Permission, PermissionStatus> statuses = 
        await permissions.request();

    bool allGranted = statuses.values
        .every((status) => status == PermissionStatus.granted);

    if (allGranted) {
      await _initializeCamera();
    } else {
      _showPermissionDialog();
    }
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      final frontCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      
      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _pulseController.dispose();
    _captureController.dispose();
    _recordingTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(),
            Expanded(child: _buildCameraPreview()),
            _buildControls(),
          ],
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    return Padding(
      padding: const EdgeInsets.all(AppConstants.spacing),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.of(context).pop(),
            icon: const Icon(Icons.close, color: Colors.white),
          ),
          const Spacer(),
          Text(
            'Cognitive Analysis',
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
              color: Colors.white,
            ),
          ),
          const Spacer(),
          const SizedBox(width: 48), // Balance the close button
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (!_isInitialized || _cameraController == null) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }

    return Stack(
      children: [
        // Camera preview
        Positioned.fill(
          child: CameraPreview(_cameraController!),
        ),
        
        // Recording indicator
        if (_isRecording)
          Positioned(
            top: 20,
            left: 20,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.red.withValues(alpha: 0.8),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: const BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _formatDuration(_recordingSeconds),
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
        
        // Center face guide
        Center(
          child: Container(
            width: 200,
            height: 250,
            decoration: BoxDecoration(
              border: Border.all(
                color: _isRecording ? AppTheme.primaryBlue : Colors.white.withValues(alpha: 0.5),
                width: 2,
              ),
              borderRadius: BorderRadius.circular(100),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildControls() {
    return Container(
      padding: const EdgeInsets.all(AppConstants.spacing * 2),
      child: Column(
        children: [
          // Instructions
          Text(
            _isRecording 
              ? 'Recording your cognitive state...'
              : 'Position your face in the guide and tap to start',
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
              color: Colors.white.withValues(alpha: 0.8),
            ),
            textAlign: TextAlign.center,
          ),
          
          const SizedBox(height: AppConstants.spacing * 2),
          
          // Capture button
          GestureDetector(
            onTap: _isRecording ? _stopRecording : _startRecording,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _isRecording ? Colors.red : AppTheme.primaryBlue,
                border: Border.all(
                  color: Colors.white.withValues(alpha: 0.3),
                  width: 3,
                ),
              ),
              child: Icon(
                _isRecording ? Icons.stop : Icons.play_arrow,
                color: Colors.white,
                size: 32,
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _formatDuration(int seconds) {
    final minutes = seconds ~/ 60;
    final remainingSeconds = seconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${remainingSeconds.toString().padLeft(2, '0')}';
  }

  void _startRecording() async {
    if (!_isInitialized) return;
    
    setState(() {
      _isRecording = true;
      _recordingSeconds = 0;
    });
    
    _pulseController.repeat();
    _startMotionTracking();
    
    // Start timer
    _recordingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        _recordingSeconds++;
      });
      
      // Auto-stop after 30 seconds
      if (_recordingSeconds >= 30) {
        _stopRecording();
      }
    });
  }

  void _stopRecording() async {
    if (!_isRecording) return;
    
    setState(() {
      _isRecording = false;
    });
    
    _pulseController.stop();
    _recordingTimer?.cancel();
    
    // Capture image
    final image = await _cameraController!.takePicture();
    final bytes = await image.readAsBytes();
    final imageBase64 = base64Encode(bytes);
    
    // Generate fake audio data (since we removed flutter_sound)
    final audioBase64 = _generateFakeAudioData();
    
    // Prepare motion data
    final motionData = {
      'accelerometer': _accelerometerData,
      'gyroscope': _gyroscopeData,
      'duration': _recordingSeconds,
    };
    
    _processAnalysis(imageBase64, audioBase64, motionData);
  }

  String _generateFakeAudioData() {
    // Generate some fake audio data for demo purposes
    final random = Random();
    final fakeAudioBytes = List.generate(1024, (index) => random.nextInt(256));
    return base64Encode(fakeAudioBytes);
  }

  void _startMotionTracking() {
    // Simulate motion data generation
    Timer.periodic(const Duration(milliseconds: 100), (timer) {
      if (!_isRecording) {
        timer.cancel();
        return;
      }
      
      final random = Random();
      _accelerometerData.add({
        'x': (random.nextDouble() - 0.5) * 2,
        'y': (random.nextDouble() - 0.5) * 2,
        'z': (random.nextDouble() - 0.5) * 2,
      });
      
      _gyroscopeData.add({
        'x': (random.nextDouble() - 0.5) * 10,
        'y': (random.nextDouble() - 0.5) * 10,
        'z': (random.nextDouble() - 0.5) * 10,
      });
    });
  }

  void _processAnalysis(String imageBase64, String audioBase64, Map<String, dynamic> motionData) async {
    try {
      // Show loading dialog
      if (mounted) {
        showDialog(
          context: context,
          barrierDismissible: false,
          builder: (context) => AlertDialog(
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const CircularProgressIndicator(),
                const SizedBox(height: 16),
                Text(
                  'Analyzing your cognitive state...',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
        );
      }

      // Get the session provider
      final sessionProvider = Provider.of<SessionProvider>(context, listen: false);
      
      // Check if there's a current session result 
      // (This should be triggered automatically by SessionProvider)
      await Future.delayed(const Duration(seconds: 2)); // Simulate processing
      
      if (mounted) {
        // Close loading dialog
        Navigator.of(context).pop();
        
        // Check if session completed with results
        if (sessionProvider.currentSessionResult != null) {
          // Navigate to results
          Navigator.of(context).pushReplacement(
            PageRouteBuilder(
              pageBuilder: (context, animation, secondaryAnimation) =>
                  ResultScreen(
                    analysisResult: sessionProvider.currentSessionResult!.toJson(),
                    sessionId: DateTime.now().millisecondsSinceEpoch.toString(),
                  ),
              transitionsBuilder: (context, animation, secondaryAnimation, child) {
                return FadeTransition(opacity: animation, child: child);
              },
              transitionDuration: AppConstants.mediumAnimation,
            ),
          );
        } else {
          // Show error if no results
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Analysis failed: No results available')),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        // Close loading dialog if open
        Navigator.of(context).pop();
        
        // Show error
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Analysis failed: $e')),
        );
      }
    }
  }

  void _showPermissionDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Permissions Required'),
        content: const Text(
          'This app needs camera and microphone access to perform cognitive analysis.',
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              Navigator.of(context).pop();
            },
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              _requestPermissions();
            },
            child: const Text('Grant'),
          ),
        ],
      ),
    );
  }
}
