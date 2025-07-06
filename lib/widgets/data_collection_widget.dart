import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import '../providers/session_provider.dart';

class DataCollectionWidget extends StatelessWidget {
  const DataCollectionWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<SessionProvider>(
      builder: (context, sessionProvider, child) {
        return Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              // Camera preview section
              Expanded(
                flex: 3,
                child: _buildCameraSection(sessionProvider),
              ),
              const SizedBox(height: 16),
              
              // Sensor data section
              Expanded(
                flex: 2,
                child: _buildSensorDataSection(sessionProvider),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildCameraSection(SessionProvider sessionProvider) {
    return Card(
      elevation: 4,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Stack(
          children: [
            // Camera preview or placeholder
            Container(
              width: double.infinity,
              height: double.infinity,
              color: Colors.black,
              child: sessionProvider.cameraController != null &&
                      sessionProvider.cameraController!.value.isInitialized
                  ? CameraPreview(sessionProvider.cameraController!)
                  : _buildCameraPlaceholder(),
            ),
            
            // Camera overlay with status
            Positioned(
              top: 16,
              left: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: sessionProvider.isCameraActive 
                      ? Colors.red.withValues(alpha: 0.8)
                      : Colors.grey.withValues(alpha: 0.8),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      sessionProvider.isCameraActive 
                          ? Icons.fiber_manual_record 
                          : Icons.camera_alt,
                      color: Colors.white,
                      size: 16,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      sessionProvider.isCameraActive ? 'Recording' : 'Standby',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            
            // Audio recording indicator
            if (sessionProvider.isAudioActive)
              Positioned(
                top: 16,
                right: 16,
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.red.withValues(alpha: 0.8),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.mic,
                    color: Colors.white,
                    size: 20,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraPlaceholder() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.camera_alt,
            size: 64,
            color: Colors.white54,
          ),
          SizedBox(height: 16),
          Text(
            'Camera Initializing...',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSensorDataSection(SessionProvider sessionProvider) {
    return Row(
      children: [
        // Motion data card
        Expanded(
          child: Card(
            elevation: 2,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.sensors,
                        color: sessionProvider.isMotionActive 
                            ? Colors.blue[700] 
                            : Colors.grey,
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        'Motion Sensors',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Expanded(
                    child: sessionProvider.currentMotionData != null
                        ? _buildMotionDataDisplay(sessionProvider.currentMotionData!)
                        : _buildNoDataDisplay(),
                  ),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(width: 8),
        
        // Audio level indicator
        SizedBox(
          width: 80,
          child: Card(
            elevation: 2,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Icon(
                    Icons.graphic_eq,
                    color: sessionProvider.isAudioActive 
                        ? Colors.green[700] 
                        : Colors.grey,
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Audio',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Expanded(
                    child: _buildAudioLevelDisplay(sessionProvider.audioLevel),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMotionDataDisplay(dynamic motionData) {
    // This would display real motion sensor data
    // For now, showing placeholder values
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildDataRow('X-Axis', '${(motionData?.x ?? 0.0).toStringAsFixed(2)}'),
        _buildDataRow('Y-Axis', '${(motionData?.y ?? 0.0).toStringAsFixed(2)}'),
        _buildDataRow('Z-Axis', '${(motionData?.z ?? 0.0).toStringAsFixed(2)}'),
      ],
    );
  }

  Widget _buildDataRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(fontSize: 12),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNoDataDisplay() {
    return const Center(
      child: Text(
        'No data available',
        style: TextStyle(
          color: Colors.grey,
          fontSize: 12,
        ),
      ),
    );
  }

  Widget _buildAudioLevelDisplay(double level) {
    return RotatedBox(
      quarterTurns: 3,
      child: LinearProgressIndicator(
        value: level,
        backgroundColor: Colors.grey[300],
        valueColor: AlwaysStoppedAnimation<Color>(
          level > 0.7 
              ? Colors.red 
              : level > 0.5 
                  ? Colors.orange 
                  : Colors.green,
        ),
        minHeight: 8,
      ),
    );
  }
}
