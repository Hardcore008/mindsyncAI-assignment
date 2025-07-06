import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/api_provider.dart';
import '../providers/session_provider.dart';
import 'session_screen.dart';
import 'history_screen.dart';

class OldDashboardScreen extends StatefulWidget {
  const OldDashboardScreen({super.key});

  @override
  State<OldDashboardScreen> createState() => _OldDashboardScreenState();
}

class _OldDashboardScreenState extends State<OldDashboardScreen> {
  @override
  void initState() {
    super.initState();
    // Check API connection on startup
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ApiProvider>().checkConnection();
      context.read<SessionProvider>().loadSessionHistory();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cognitive Analysis'),
        backgroundColor: Colors.blue[700],
        foregroundColor: Colors.white,
        actions: [
          Consumer<ApiProvider>(
            builder: (context, apiProvider, child) {
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                margin: const EdgeInsets.only(right: 16),
                decoration: BoxDecoration(
                  color: _getConnectionColor(apiProvider.connectionStatus),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _getConnectionIcon(apiProvider.connectionStatus),
                      size: 16,
                      color: Colors.white,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _getConnectionText(apiProvider.connectionStatus),
                      style: const TextStyle(fontSize: 12, color: Colors.white),
                    ),
                  ],
                ),
              );
            },
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blue[50]!, Colors.white],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Welcome section
                  const Text(
                    'Welcome to Cognitive Analysis',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Start a new session to analyze your cognitive state using facial recognition, voice analysis, and motion tracking.',
                    style: TextStyle(
                    fontSize: 16,
                    color: Colors.black54,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 40),

                // Main action card
                Card(
                  elevation: 8,
                  shadowColor: Colors.blue.withValues(alpha: 0.3),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Container(
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(16),
                      gradient: LinearGradient(
                        colors: [Colors.blue[600]!, Colors.blue[800]!],
                      ),
                    ),
                    child: Column(
                      children: [
                        const Icon(
                          Icons.psychology,
                          size: 64,
                          color: Colors.white,
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          'Start New Session',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 8),
                        const Text(
                          'Capture face, voice, and motion data for analysis',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.white70,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 24),
                        Consumer2<ApiProvider, SessionProvider>(
                          builder: (context, apiProvider, sessionProvider, child) {
                            final isConnected = apiProvider.connectionStatus == ApiConnectionStatus.connected;
                            final isIdle = sessionProvider.sessionState == SessionState.idle;
                            
                            return ElevatedButton(
                              onPressed: isConnected && isIdle ? _startNewSession : null,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.white,
                                foregroundColor: Colors.blue[700],
                                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                elevation: 4,
                              ),
                              child: Text(
                                isConnected ? 'Begin Analysis' : 'Backend Disconnected',
                                style: const TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 32),

                // Secondary actions
                Row(
                  children: [
                    Expanded(
                      child: _buildActionCard(
                        icon: Icons.history,
                        title: 'View History',
                        subtitle: 'Previous sessions',
                        onTap: _viewHistory,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: _buildActionCard(
                        icon: Icons.settings,
                        title: 'Settings',
                        subtitle: 'Configure backend',
                        onTap: _showSettings,
                      ),
                    ),
                  ],
                ),

                const Spacer(),

                // Connection status details
                Consumer<ApiProvider>(
                  builder: (context, apiProvider, child) {
                    if (apiProvider.errorMessage != null) {
                      return Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.orange[50],
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.orange[200]!),
                        ),
                        child: Row(
                          children: [
                            Icon(Icons.warning, color: Colors.orange[700]),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                apiProvider.errorMessage!,
                                style: TextStyle(color: Colors.orange[700]),
                              ),
                            ),
                            TextButton(
                              onPressed: () => apiProvider.checkConnection(),
                              child: const Text('Retry'),
                            ),
                          ],
                        ),
                      );
                    }
                    return const SizedBox.shrink();
                  },
                ),
              ],
            ),
          ),
        ),
        ),
      ),
    );
  }

  Widget _buildActionCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Icon(icon, size: 32, color: Colors.blue[600]),
              const SizedBox(height: 8),
              Text(
                title,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
              Text(
                subtitle,
                style: TextStyle(
                  color: Colors.grey[600],
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _startNewSession() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const SessionScreen()),
    );
  }

  void _viewHistory() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const HistoryScreen()),
    );
  }

  void _showSettings() {
    final apiProvider = context.read<ApiProvider>();
    
    showDialog(
      context: context,
      builder: (context) {
        String url = apiProvider.baseUrl;
        return AlertDialog(
          title: const Text('Backend Settings'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                decoration: const InputDecoration(
                  labelText: 'Backend URL',
                  hintText: 'http://192.168.29.73:8000',
                ),
                controller: TextEditingController(text: url),
                onChanged: (value) => url = value,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                apiProvider.updateBaseUrl(url);
                apiProvider.checkConnection();
                Navigator.pop(context);
              },
              child: const Text('Save'),
            ),
          ],
        );
      },
    );
  }

  Color _getConnectionColor(ApiConnectionStatus status) {
    switch (status) {
      case ApiConnectionStatus.connected:
        return Colors.green;
      case ApiConnectionStatus.disconnected:
        return Colors.red;
      case ApiConnectionStatus.checking:
        return Colors.orange;
      case ApiConnectionStatus.unknown:
        return Colors.grey;
    }
  }

  IconData _getConnectionIcon(ApiConnectionStatus status) {
    switch (status) {
      case ApiConnectionStatus.connected:
        return Icons.wifi;
      case ApiConnectionStatus.disconnected:
        return Icons.wifi_off;
      case ApiConnectionStatus.checking:
        return Icons.wifi_find;
      case ApiConnectionStatus.unknown:
        return Icons.help;
    }
  }

  String _getConnectionText(ApiConnectionStatus status) {
    switch (status) {
      case ApiConnectionStatus.connected:
        return 'Connected';
      case ApiConnectionStatus.disconnected:
        return 'Offline';
      case ApiConnectionStatus.checking:
        return 'Checking...';
      case ApiConnectionStatus.unknown:
        return 'Unknown';
    }
  }
}
