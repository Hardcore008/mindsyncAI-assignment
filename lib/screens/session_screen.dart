import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/session_provider.dart';
import '../widgets/session_progress_widget.dart';
import '../widgets/data_collection_widget.dart';
import '../widgets/results_display_widget.dart';

class SessionScreen extends StatefulWidget {
  const SessionScreen({super.key});

  @override
  State<SessionScreen> createState() => _SessionScreenState();
}

class _SessionScreenState extends State<SessionScreen> {
  @override
  void initState() {
    super.initState();
    // Request permissions and start session preparation
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _prepareSession();
    });
  }

  Future<void> _prepareSession() async {
    final sessionProvider = context.read<SessionProvider>();
    try {
      await sessionProvider.requestPermissions();
      if (mounted) {
        sessionProvider.prepareSession();
      }
    } catch (e) {
      if (mounted) {
        _showErrorDialog('Permission Error', e.toString());
      }
    }
  }

  void _showErrorDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              Navigator.of(context).pop(); // Return to dashboard
            },
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cognitive Session'),
        backgroundColor: Colors.blue[700],
        foregroundColor: Colors.white,
        leading: Consumer<SessionProvider>(
          builder: (context, sessionProvider, child) {
            return IconButton(
              icon: const Icon(Icons.arrow_back),
              onPressed: sessionProvider.sessionState == SessionState.running
                  ? null // Disable back button during active session
                  : () => Navigator.of(context).pop(),
            );
          },
        ),
      ),
      body: Consumer<SessionProvider>(
        builder: (context, sessionProvider, child) {
          return _buildSessionContent(sessionProvider);
        },
      ),
    );
  }

  Widget _buildSessionContent(SessionProvider sessionProvider) {
    switch (sessionProvider.sessionState) {
      case SessionState.idle:
        return _buildWelcomeScreen(sessionProvider);
      case SessionState.preparing:
        return _buildPreparingScreen();
      case SessionState.ready:
        return _buildReadyScreen(sessionProvider);
      case SessionState.running:
        return _buildRunningScreen(sessionProvider);
      case SessionState.processing:
        return _buildProcessingScreen();
      case SessionState.completed:
        return _buildResultsScreen(sessionProvider);
      case SessionState.error:
        return _buildErrorScreen(sessionProvider);
    }
  }

  Widget _buildWelcomeScreen(SessionProvider sessionProvider) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.psychology,
            size: 100,
            color: Colors.blue[700],
          ),
          const SizedBox(height: 24),
          const Text(
            'Welcome to Cognitive Analysis',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          const Text(
            'This session will collect multi-sensor data to analyze your cognitive state. Please ensure you are in a quiet environment.',
            style: TextStyle(fontSize: 16),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          const Card(
            child: Padding(
              padding: EdgeInsets.all(16.0),
              child: Column(
                children: [
                  Text(
                    'Session Duration: 10 seconds',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 8),
                  Text('Data Collection:'),
                  SizedBox(height: 4),
                  Text('• Face imaging'),
                  Text('• Voice recording'),
                  Text('• Motion sensors'),
                ],
              ),
            ),
          ),
          const SizedBox(height: 32),
          if (sessionProvider.permissionsGranted)
            ElevatedButton(
              onPressed: () => sessionProvider.prepareSession(),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue[700],
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
              ),
              child: const Text(
                'Start Session',
                style: TextStyle(fontSize: 18),
              ),
            )
          else
            ElevatedButton(
              onPressed: _prepareSession,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.orange,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
              ),
              child: const Text(
                'Grant Permissions',
                style: TextStyle(fontSize: 18),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildPreparingScreen() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 24),
          Text(
            'Preparing session...',
            style: TextStyle(fontSize: 18),
          ),
          SizedBox(height: 8),
          Text('Initializing sensors and camera'),
        ],
      ),
    );
  }

  Widget _buildReadyScreen(SessionProvider sessionProvider) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.check_circle,
            size: 100,
            color: Colors.green[600],
          ),
          const SizedBox(height: 24),
          const Text(
            'Ready to Begin',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'All sensors are ready. The session will last 10 seconds.',
            style: TextStyle(fontSize: 16),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          ElevatedButton(
            onPressed: () => sessionProvider.startSession(),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green[600],
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
            ),
            child: const Text(
              'Begin Session',
              style: TextStyle(fontSize: 18),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRunningScreen(SessionProvider sessionProvider) {
    return Column(
      children: [
        // Progress indicator
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(24),
          color: Colors.blue[50],
          child: const SessionProgressWidget(),
        ),
        // Data collection display
        const Expanded(
          child: DataCollectionWidget(),
        ),
        // Session controls
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Colors.white,
            boxShadow: [
              BoxShadow(
                color: Colors.grey.withValues(alpha: 0.3),
                spreadRadius: 1,
                blurRadius: 5,
                offset: const Offset(0, -2),
              ),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                onPressed: () => sessionProvider.pauseSession(),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange,
                  foregroundColor: Colors.white,
                ),
                child: const Text('Pause'),
              ),
              ElevatedButton(
                onPressed: () => _showStopConfirmation(sessionProvider),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  foregroundColor: Colors.white,
                ),
                child: const Text('Stop'),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildProcessingScreen() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 24),
          Text(
            'Processing Data...',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 8),
          Text('Analyzing cognitive patterns'),
        ],
      ),
    );
  }

  Widget _buildResultsScreen(SessionProvider sessionProvider) {
    return Column(
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(24),
          color: Colors.green[50],
          child: Column(
            children: [
              Icon(
                Icons.check_circle,
                size: 60,
                color: Colors.green[600],
              ),
              const SizedBox(height: 16),
              const Text(
                'Session Complete',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Text('Analysis results are ready'),
            ],
          ),
        ),        Expanded(
          child: sessionProvider.currentSessionResult != null
              ? ResultsDisplayWidget(
                  sessionResult: sessionProvider.currentSessionResult!,
                )
              : const Center(
                  child: Text('No results available'),
                ),
        ),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(24),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                onPressed: () => sessionProvider.resetSession(),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue[700],
                  foregroundColor: Colors.white,
                ),
                child: const Text('New Session'),
              ),
              ElevatedButton(
                onPressed: () => Navigator.of(context).pop(),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.grey[600],
                  foregroundColor: Colors.white,
                ),
                child: const Text('Return to Dashboard'),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildErrorScreen(SessionProvider sessionProvider) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.error,
            size: 100,
            color: Colors.red[600],
          ),
          const SizedBox(height: 24),
          const Text(
            'Session Error',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            sessionProvider.errorMessage ?? 'An unknown error occurred',
            style: const TextStyle(fontSize: 16),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                onPressed: () => sessionProvider.resetSession(),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue[700],
                  foregroundColor: Colors.white,
                ),
                child: const Text('Try Again'),
              ),
              ElevatedButton(
                onPressed: () => Navigator.of(context).pop(),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.grey[600],
                  foregroundColor: Colors.white,
                ),
                child: const Text('Go Back'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _showStopConfirmation(SessionProvider sessionProvider) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Stop Session'),
        content: const Text(
          'Are you sure you want to stop the session? All collected data will be lost.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              sessionProvider.stopSession();
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Stop'),
          ),
        ],
      ),
    );
  }
}
