import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/session_response.dart';

class ResultsDisplayWidget extends StatelessWidget {
  final SessionResponse sessionResult;

  const ResultsDisplayWidget({
    super.key,
    required this.sessionResult,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Cognitive State Card
          _buildCognitiveStateCard(),
          const SizedBox(height: 16),
          
          // Metrics Overview
          _buildMetricsOverview(),
          const SizedBox(height: 16),
          
          // Charts Section
          _buildChartsSection(),
          const SizedBox(height: 16),
          
          // Insights Section
          _buildInsightsSection(),
          const SizedBox(height: 16),
          
          // Recommendations Section
          _buildRecommendationsSection(),
        ],
      ),
    );
  }

  Widget _buildCognitiveStateCard() {
    return Card(
      elevation: 4,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(8),
          gradient: LinearGradient(
            colors: [
              _getCognitiveStateColor(sessionResult.cognitiveState),
              _getCognitiveStateColor(sessionResult.cognitiveState).withValues(alpha: 0.7),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Column(
          children: [
            Icon(
              _getCognitiveStateIcon(sessionResult.cognitiveState),
              size: 64,
              color: Colors.white,
            ),
            const SizedBox(height: 16),
            Text(
              'Cognitive State',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              sessionResult.cognitiveState.toUpperCase(),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Confidence: ${(sessionResult.confidence * 100).toInt()}%',
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricsOverview() {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Metrics Overview',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: _buildMetricItem(
                    'Stress Level',
                    sessionResult.metrics['stress_level']?.toString() ?? 'N/A',
                    Icons.favorite,
                    Colors.red,
                  ),
                ),
                Expanded(
                  child: _buildMetricItem(
                    'Focus Score',
                    sessionResult.metrics['focus_score']?.toString() ?? 'N/A',
                    Icons.center_focus_strong,
                    Colors.blue,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildMetricItem(
                    'Energy Level',
                    sessionResult.metrics['energy_level']?.toString() ?? 'N/A',
                    Icons.battery_full,
                    Colors.green,
                  ),
                ),
                Expanded(
                  child: _buildMetricItem(
                    'Mood Score',
                    sessionResult.metrics['mood_score']?.toString() ?? 'N/A',
                    Icons.sentiment_satisfied,
                    Colors.orange,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricItem(String label, String value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      margin: const EdgeInsets.symmetric(horizontal: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 24),
          const SizedBox(height: 8),
          Text(
            value,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: const TextStyle(
              fontSize: 12,
              color: Colors.grey,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildChartsSection() {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Cognitive Patterns',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: _buildCognitiveChart(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCognitiveChart() {
    // Generate sample data based on session results
    final chartData = _generateChartData();
    
    return LineChart(
      LineChartData(
        gridData: FlGridData(show: true),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(showTitles: true, reservedSize: 40),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(showTitles: true, reservedSize: 22),
          ),
          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        borderData: FlBorderData(show: true),
        lineBarsData: [
          LineChartBarData(
            spots: chartData,
            isCurved: true,
            color: Colors.blue,
            barWidth: 3,
            dotData: FlDotData(show: false),
          ),
        ],
        minX: 0,
        maxX: chartData.length.toDouble() - 1,
        minY: 0,
        maxY: 100,
      ),
    );
  }

  List<FlSpot> _generateChartData() {
    // Generate sample cognitive pattern data
    final baseValue = sessionResult.confidence * 100;
    final List<FlSpot> spots = [];
    
    for (int i = 0; i < 10; i++) {
      final variation = (i % 3 - 1) * 10; // Add some variation
      final value = (baseValue + variation).clamp(0.0, 100.0);
      spots.add(FlSpot(i.toDouble(), value));
    }
    
    return spots;
  }

  Widget _buildInsightsSection() {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Key Insights',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            if (sessionResult.insights.isEmpty)
              const Text(
                'No specific insights available for this session.',
                style: TextStyle(color: Colors.grey),
              )
            else
              ...sessionResult.insights.map((insight) => Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.lightbulb,
                      color: Colors.amber[700],
                      size: 20,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        insight,
                        style: const TextStyle(fontSize: 14),
                      ),
                    ),
                  ],
                ),
              )),
          ],
        ),
      ),
    );
  }

  Widget _buildRecommendationsSection() {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Recommendations',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            if (sessionResult.recommendations.isEmpty)
              const Text(
                'No specific recommendations available.',
                style: TextStyle(color: Colors.grey),
              )
            else
              ...sessionResult.recommendations.map((recommendation) => Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.recommend,
                      color: Colors.green[700],
                      size: 20,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        recommendation,
                        style: const TextStyle(fontSize: 14),
                      ),
                    ),
                  ],
                ),
              )),
          ],
        ),
      ),
    );
  }

  Color _getCognitiveStateColor(String state) {
    switch (state.toLowerCase()) {
      case 'focused':
      case 'alert':
        return Colors.green;
      case 'relaxed':
      case 'calm':
        return Colors.blue;
      case 'stressed':
      case 'anxious':
        return Colors.orange;
      case 'fatigued':
      case 'tired':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  IconData _getCognitiveStateIcon(String state) {
    switch (state.toLowerCase()) {
      case 'focused':
      case 'alert':
        return Icons.center_focus_strong;
      case 'relaxed':
      case 'calm':
        return Icons.self_improvement;
      case 'stressed':
      case 'anxious':
        return Icons.warning;
      case 'fatigued':
      case 'tired':
        return Icons.battery_0_bar;
      default:
        return Icons.psychology;
    }
  }
}
