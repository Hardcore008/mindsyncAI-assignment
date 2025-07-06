import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../core/theme/app_theme.dart';
import '../../data/models/session_model.dart';

class HealthStatsCard extends StatelessWidget {
  final List<SessionModel> sessions;
  final String title;
  final IconData icon;

  const HealthStatsCard({
    super.key,
    required this.sessions,
    required this.title,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
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
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppTheme.primaryBlue.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  icon,
                  color: AppTheme.primaryBlue,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    fontFamily: 'SF Pro Display',
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          if (sessions.isNotEmpty) ...[
            _buildStatsContent(),
          ] else ...[
            _buildEmptyState(),
          ],
        ],
      ),
    );
  }

  Widget _buildStatsContent() {
    final recentSessions = sessions.take(7).toList();
    final avgMood = _calculateAverageMood(recentSessions);
    final trendDirection = _calculateTrend(recentSessions);

    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${avgMood.toStringAsFixed(1)}/10',
                  style: const TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w700,
                    fontFamily: 'SF Pro Display',
                  ),
                ),
                Text(
                  'Average Score',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey[600],
                    fontFamily: 'SF Pro Display',
                  ),
                ),
              ],
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: trendDirection > 0
                    ? Colors.green.withValues(alpha: 0.1)
                    : trendDirection < 0
                        ? Colors.red.withValues(alpha: 0.1)
                        : Colors.grey.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    trendDirection > 0
                        ? Icons.trending_up
                        : trendDirection < 0
                            ? Icons.trending_down
                            : Icons.trending_flat,
                    size: 16,
                    color: trendDirection > 0
                        ? Colors.green
                        : trendDirection < 0
                            ? Colors.red
                            : Colors.grey,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    trendDirection > 0
                        ? 'Improving'
                        : trendDirection < 0
                            ? 'Declining'
                            : 'Stable',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      color: trendDirection > 0
                          ? Colors.green
                          : trendDirection < 0
                              ? Colors.red
                              : Colors.grey,
                      fontFamily: 'SF Pro Display',
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 20),
        SizedBox(
          height: 60,
          child: LineChart(
            LineChartData(
              gridData: const FlGridData(show: false),
              titlesData: const FlTitlesData(show: false),
              borderData: FlBorderData(show: false),
              lineBarsData: [
                LineChartBarData(
                  spots: _generateChartSpots(recentSessions),
                  isCurved: true,
                  color: AppTheme.primaryBlue,
                  barWidth: 3,
                  isStrokeCapRound: true,
                  dotData: const FlDotData(show: false),
                  belowBarData: BarAreaData(
                    show: true,
                    color: AppTheme.primaryBlue.withValues(alpha: 0.1),
                  ),
                ),
              ],
              minY: 0,
              maxY: 10,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        children: [
          Icon(
            Icons.analytics_outlined,
            size: 48,
            color: Colors.grey[400],
          ),
          const SizedBox(height: 12),
          Text(
            'No data available',
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey[600],
              fontFamily: 'SF Pro Display',
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Complete a session to see stats',
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey[500],
              fontFamily: 'SF Pro Display',
            ),
          ),
        ],
      ),
    );
  }

  double _calculateAverageMood(List<SessionModel> sessions) {
    if (sessions.isEmpty) return 0.0;
    double total = 0.0;
    for (final session in sessions) {
      // Extract mood score from cognitive state or analysis
      if (session.analysisResult != null) {
        total += session.analysisResult!['mood_score'] ?? 5.0;
      }
    }
    return total / sessions.length;
  }

  double _calculateTrend(List<SessionModel> sessions) {
    if (sessions.length < 2) return 0.0;
    
    final recent = sessions.take(3).toList();
    final older = sessions.skip(3).take(3).toList();
    
    if (older.isEmpty) return 0.0;
    
    final recentAvg = _calculateAverageMood(recent);
    final olderAvg = _calculateAverageMood(older);
    
    return recentAvg - olderAvg;
  }

  List<FlSpot> _generateChartSpots(List<SessionModel> sessions) {
    if (sessions.isEmpty) return [];
    
    return sessions.asMap().entries.map((entry) {
      final index = entry.key;
      final session = entry.value;
      double value = 5.0; // Default value
      
      if (session.analysisResult != null) {
        value = session.analysisResult!['mood_score']?.toDouble() ?? 5.0;
      }
      
      return FlSpot(index.toDouble(), value);
    }).toList();
  }
}
