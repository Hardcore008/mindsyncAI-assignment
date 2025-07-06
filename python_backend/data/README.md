# Data Directory

This directory contains the SQLite database and related data files for the MindSync AI backend.

## Contents

- `mindsync.db` - SQLite database file (created automatically)
- Backup files (if database backup is performed)
- Log files (if file logging is enabled)

## Database Schema

The database contains the following tables:

1. **sessions** - Analysis session data
2. **user_preferences** - User configuration and privacy settings
3. **analysis_metrics** - Performance and quality metrics

## Privacy Notes

- No raw sensor data (images, audio) is stored in the database
- Only extracted features and analysis results are stored
- User data can be automatically cleaned based on preferences
- All data storage complies with privacy best practices

## Backup

Regular backups can be performed using the database manager utility functions.
