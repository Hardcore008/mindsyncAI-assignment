# MindSync AI - Training and Calibration Data

This directory contains data files used for model training, calibration, and user onboarding.

## Data Files

### 1. cognitive_state_templates.json
- **Purpose**: Template patterns for different cognitive states
- **Content**: Feature patterns, thresholds, and characteristics for each state type
- **Usage**: Initial model calibration and baseline establishment

### 2. intervention_effectiveness.json
- **Purpose**: Data on intervention effectiveness for different user profiles
- **Content**: Success rates, user preferences, and optimal timing for interventions
- **Usage**: Personalized intervention recommendations

### 3. sensor_calibration.json
- **Purpose**: Sensor calibration parameters and normalization factors
- **Content**: Device-specific calibration data, noise thresholds, quality metrics
- **Usage**: Sensor data preprocessing and quality assessment

### 4. breathing_patterns.json
- **Purpose**: Reference breathing patterns for guided exercises
- **Content**: Optimal breathing rates, patterns, and progression sequences
- **Usage**: Breathing exercise guidance and effectiveness measurement

### 5. user_onboarding_data.json
- **Purpose**: Baseline data collection templates for new users
- **Content**: Questionnaires, initial assessments, and calibration protocols
- **Usage**: User onboarding and initial model setup

## Data Privacy

All data files are designed to be:
- **Privacy-first**: No personal information stored
- **Anonymized**: Template patterns only, no individual data
- **Local**: Processed entirely on-device
- **Secure**: Encrypted when stored locally

## Mock Data

For this demo, mock data generators are provided that create realistic patterns for testing and demonstration purposes. In a production environment, these would be replaced with actual trained data and user-collected patterns.

## Usage in App

Data is accessed through:
- `LocalStorageService`: Local data persistence
- `PersonalizedModelTraining`: Template and calibration data
- `ComprehensiveMLService`: Reference patterns and thresholds
