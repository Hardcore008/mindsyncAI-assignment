import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/constants/app_constants.dart';
import '../../../providers/session_provider.dart';
import '../../widgets/animated_card.dart';
import '../../../models/session_history.dart';
import '../../../models/session_response.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  @override
  void initState() {
    super.initState();
    // Load session history when screen is initialized
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<SessionProvider>().loadSessionHistory();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundLight,
      appBar: AppBar(
        title: const Text('Session History'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Consumer<SessionProvider>(
        builder: (context, sessionProvider, child) {
          if (sessionProvider.isLoading) {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }
          
          if (sessionProvider.errorMessage != null) {
            return _buildErrorState(sessionProvider.errorMessage!);
          }
          
          if (sessionProvider.sessionHistory.isEmpty) {
            return _buildEmptyState();
          }
          
          return _buildSessionsList(sessionProvider.sessionHistory);
        },
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.history,
            size: 80,
            color: Colors.grey[300],
          ),
          const SizedBox(height: 16),
          Text(
            'No sessions yet',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w600,
              color: Colors.grey[600],
              fontFamily: 'SF Pro Display',
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Complete your first cognitive analysis session',
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey[500],
              fontFamily: 'SF Pro Display',
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorState(String message) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.error_outline,
            size: 80,
            color: Colors.red[300],
          ),
          const SizedBox(height: 16),
          Text(
            'Error loading sessions',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w600,
              color: Colors.red[600],
              fontFamily: 'SF Pro Display',
            ),
          ),
          const SizedBox(height: 8),
          Text(
            message,
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey[500],
              fontFamily: 'SF Pro Display',
            ),
          ),
          const SizedBox(height: 24),
          Builder(
            builder: (BuildContext context) {
              return ElevatedButton(
                onPressed: () {
                  context.read<SessionProvider>().loadSessionHistory();
                },
                child: const Text('Retry'),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildSessionsList(List<SessionHistory> sessions) {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: sessions.length,
      itemBuilder: (context, index) {
        final session = sessions[index];
        return AnimatedCard(
          delay: Duration(milliseconds: index * 100),
          child: _buildSessionCard(session),
        );
      },
    );
  }

  Widget _buildSessionCard(SessionHistory session) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                _formatDate(session.timestamp),
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  fontFamily: 'SF Pro Display',
                ),
              ),
              if (session.sessionResponse != null)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: _getStateColor(session.sessionResponse!.cognitiveState),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    session.sessionResponse!.cognitiveState,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      fontFamily: 'SF Pro Display',
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          if (session.sessionResponse != null) ...[
            _buildAnalysisResults(session.sessionResponse!),
          ] else ...[
            Text(
              'Analysis in progress...',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
                fontFamily: 'SF Pro Display',
              ),
            ),
          ],
          const SizedBox(height: 16),
          Row(
            children: [
              Icon(
                Icons.access_time,
                size: 16,
                color: Colors.grey[500],
              ),
              const SizedBox(width: 4),
              Text(
                'Duration: ${session.duration}s',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                  fontFamily: 'SF Pro Display',
                ),
              ),
              const Spacer(),
              Icon(
                Icons.check_circle,
                size: 16,
                color: Colors.green[500],
              ),
              const SizedBox(width: 4),
              Text(
                'Completed',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.green[600],
                  fontFamily: 'SF Pro Display',
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAnalysisResults(SessionResponse results) {
    final stressLevel = results.metrics['stress_level'] as double?;
    final attentionScore = results.metrics['attention_score'] as double?;
    final overallScore = results.metrics['overall_score'] as double?;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (stressLevel != null) ...[
          Row(
            children: [
              Text(
                'Stress Level: ',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                  fontFamily: 'SF Pro Display',
                ),
              ),
              Text(
                '${(stressLevel * 10).toStringAsFixed(1)}/10',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  fontFamily: 'SF Pro Display',
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
        ],
        if (attentionScore != null) ...[
          Row(
            children: [
              Text(
                'Attention Score: ',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                  fontFamily: 'SF Pro Display',
                ),
              ),
              Text(
                '${(attentionScore * 10).toStringAsFixed(1)}/10',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  fontFamily: 'SF Pro Display',
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
        ],
        if (overallScore != null) ...[
          Row(
            children: [
              Text(
                'Overall Score: ',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                  fontFamily: 'SF Pro Display',
                ),
              ),
              Text(
                '${(overallScore * 10).toStringAsFixed(1)}/10',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  fontFamily: 'SF Pro Display',
                ),
              ),
            ],
          ),
        ],
      ],
    );
  }

  String _formatDate(DateTime dateTime) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);
    
    if (difference.inDays == 0) {
      if (difference.inHours == 0) {
        return '${difference.inMinutes} minutes ago';
      }
      return '${difference.inHours} hours ago';
    } else if (difference.inDays == 1) {
      return 'Yesterday';
    } else if (difference.inDays < 7) {
      return '${difference.inDays} days ago';
    } else {
      return '${dateTime.day}/${dateTime.month}/${dateTime.year}';
    }
  }

  Color _getStateColor(String state) {
    const stateColorMap = AppConstants.stateColors;
    return Color(stateColorMap[state] ?? stateColorMap['Neutral']!);
  }
}
