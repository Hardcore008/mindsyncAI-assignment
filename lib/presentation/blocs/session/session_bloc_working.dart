import 'dart:async';
import 'package:equatable/equatable.dart';

import '../../../data/repositories/session_repository.dart';
import '../../../data/models/session_model.dart';

// Events
abstract class SessionEvent extends Equatable {
  const SessionEvent();

  @override
  List<Object> get props => [];
}

class LoadSessions extends SessionEvent {}

class SaveSession extends SessionEvent {
  final SessionModel session;

  const SaveSession(this.session);

  @override
  List<Object> get props => [session];
}

class DeleteSession extends SessionEvent {
  final String sessionId;

  const DeleteSession(this.sessionId);

  @override
  List<Object> get props => [sessionId];
}

// States
abstract class SessionState extends Equatable {
  const SessionState();

  @override
  List<Object?> get props => [];
}

class SessionInitial extends SessionState {}

class SessionLoading extends SessionState {}

class SessionLoaded extends SessionState {
  final List<SessionModel> sessions;

  const SessionLoaded(this.sessions);

  @override
  List<Object> get props => [sessions];
}

class SessionError extends SessionState {
  final String message;

  const SessionError(this.message);

  @override
  List<Object> get props => [message];
}

// Simple Bloc implementation without flutter_bloc
class SessionBloc {
  final SessionRepository sessionRepository;
  final StreamController<SessionState> _stateController = StreamController<SessionState>.broadcast();
  final StreamController<SessionEvent> _eventController = StreamController<SessionEvent>();

  SessionState _currentState = SessionInitial();

  SessionBloc({required this.sessionRepository}) {
    _eventController.stream.listen(_handleEvent);
  }

  Stream<SessionState> get stream => _stateController.stream;
  SessionState get state => _currentState;

  void add(SessionEvent event) {
    _eventController.add(event);
  }

  void _emit(SessionState state) {
    _currentState = state;
    _stateController.add(state);
  }

  Future<void> _handleEvent(SessionEvent event) async {
    if (event is LoadSessions) {
      await _handleLoadSessions();
    } else if (event is SaveSession) {
      await _handleSaveSession(event);
    } else if (event is DeleteSession) {
      await _handleDeleteSession(event);
    }
  }

  Future<void> _handleLoadSessions() async {
    try {
      _emit(SessionLoading());
      final sessions = await sessionRepository.getAllSessions();
      _emit(SessionLoaded(sessions));
    } catch (e) {
      _emit(SessionError(e.toString()));
    }
  }

  Future<void> _handleSaveSession(SaveSession event) async {
    try {
      await sessionRepository.saveSession(event.session);
      // Reload sessions
      add(LoadSessions());
    } catch (e) {
      _emit(SessionError(e.toString()));
    }
  }

  Future<void> _handleDeleteSession(DeleteSession event) async {
    try {
      await sessionRepository.deleteSession(event.sessionId);
      // Reload sessions
      add(LoadSessions());
    } catch (e) {
      _emit(SessionError(e.toString()));
    }
  }

  void dispose() {
    _stateController.close();
    _eventController.close();
  }
}
