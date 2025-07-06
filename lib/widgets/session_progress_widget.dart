import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/session_provider.dart';

class SessionProgressWidget extends StatelessWidget {
  const SessionProgressWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<SessionProvider>(
      builder: (context, sessionProvider, child) {
        final progress = sessionProvider.sessionProgress;
        final remainingTime = sessionProvider.remainingTime;
        
        return Column(
          children: [
            // Time display
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Session Progress',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  _formatDuration(remainingTime),
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: remainingTime.inSeconds <= 30 
                        ? Colors.red 
                        : Colors.blue[700],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            
            // Progress bar
            LinearProgressIndicator(
              value: progress,
              backgroundColor: Colors.grey[300],
              valueColor: AlwaysStoppedAnimation<Color>(
                progress > 0.8 
                    ? Colors.green 
                    : progress > 0.5 
                        ? Colors.orange 
                        : Colors.blue[700]!,
              ),
              minHeight: 8,
            ),
            const SizedBox(height: 8),
            
            // Progress percentage
            Text(
              '${(progress * 100).toInt()}% Complete',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
              ),
            ),
            const SizedBox(height: 16),
            
            // Data collection status
            _buildDataCollectionStatus(sessionProvider),
          ],
        );
      },
    );
  }

  Widget _buildDataCollectionStatus(SessionProvider sessionProvider) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _buildStatusIndicator(
          icon: Icons.camera_alt,
          label: 'Camera',
          isActive: sessionProvider.isCameraActive,
        ),
        _buildStatusIndicator(
          icon: Icons.mic,
          label: 'Audio',
          isActive: sessionProvider.isAudioActive,
        ),
        _buildStatusIndicator(
          icon: Icons.sensors,
          label: 'Motion',
          isActive: sessionProvider.isMotionActive,
        ),
      ],
    );
  }

  Widget _buildStatusIndicator({
    required IconData icon,
    required String label,
    required bool isActive,
  }) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: isActive ? Colors.green[100] : Colors.grey[200],
            shape: BoxShape.circle,
            border: Border.all(
              color: isActive ? Colors.green : Colors.grey,
              width: 2,
            ),
          ),
          child: Icon(
            icon,
            color: isActive ? Colors.green[700] : Colors.grey[500],
            size: 20,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: isActive ? Colors.green[700] : Colors.grey[500],
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  String _formatDuration(Duration duration) {
    final minutes = duration.inMinutes;
    final seconds = duration.inSeconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
  }
}
