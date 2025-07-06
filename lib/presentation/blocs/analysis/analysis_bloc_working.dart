import 'dart:async';
import 'package:equatable/equatable.dart';

import '../../../data/repositories/api_repository.dart';
import '../../../data/models/session_model.dart';

// Events
abstract class AnalysisEvent extends Equatable {
  const AnalysisEvent();

  @override
  List<Object> get props => [];
}

class StartAnalysis extends AnalysisEvent {
  final String faceImageBase64;
  final String audioBase64;
  final List<Map<String, dynamic>> motionData;

  const StartAnalysis({
    required this.faceImageBase64,
    required this.audioBase64,
    required this.motionData,
  });

  @override
  List<Object> get props => [faceImageBase64, audioBase64, motionData];
}

class ResetAnalysis extends AnalysisEvent {}

// States
abstract class AnalysisState extends Equatable {
  const AnalysisState();

  @override
  List<Object?> get props => [];
}

class AnalysisInitial extends AnalysisState {}

class AnalysisLoading extends AnalysisState {}

class AnalysisCompleted extends AnalysisState {
  final SessionModel session;

  const AnalysisCompleted(this.session);

  @override
  List<Object> get props => [session];
}

class AnalysisError extends AnalysisState {
  final String message;

  const AnalysisError(this.message);

  @override
  List<Object> get props => [message];
}

// Simple Bloc implementation without flutter_bloc
class AnalysisBloc {
  final ApiRepository apiRepository;
  final StreamController<AnalysisState> _stateController = StreamController<AnalysisState>.broadcast();
  final StreamController<AnalysisEvent> _eventController = StreamController<AnalysisEvent>();

  AnalysisState _currentState = AnalysisInitial();

  AnalysisBloc({required this.apiRepository}) {
    _eventController.stream.listen(_handleEvent);
  }

  Stream<AnalysisState> get stream => _stateController.stream;
  AnalysisState get state => _currentState;

  void add(AnalysisEvent event) {
    _eventController.add(event);
  }

  void _emit(AnalysisState state) {
    _currentState = state;
    _stateController.add(state);
  }

  Future<void> _handleEvent(AnalysisEvent event) async {
    if (event is StartAnalysis) {
      await _handleStartAnalysis(event);
    } else if (event is ResetAnalysis) {
      _emit(AnalysisInitial());
    }
  }

  Future<void> _handleStartAnalysis(StartAnalysis event) async {
    try {
      _emit(AnalysisLoading());
      final session = await apiRepository.analyzeSession(
        faceImageBase64: event.faceImageBase64,
        audioBase64: event.audioBase64,
        motionData: event.motionData,
      );
      _emit(AnalysisCompleted(session));
    } catch (e) {
      _emit(AnalysisError(e.toString()));
    }
  }

  void dispose() {
    _stateController.close();
    _eventController.close();
  }
}
