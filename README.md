# MindSync AI Flutter App

This project uses Provider for state management, SQLite for local storage, and follows a privacy-first architecture. All user data is stored locally and never leaves the device.

## Project Structure
- lib/models: Data models
- lib/providers: State management
- lib/services: Core services (sensors, ML, etc.)
- lib/screens: UI screens
- lib/widgets: Reusable widgets

## Getting Started
1. Run `flutter pub get` to install dependencies.
2. Run the app with `flutter run`.

## Dependencies
- provider
- sqflite
- path_provider

## Privacy
All data is processed and stored locally. No data is sent to external servers.
