// This is a basic Flutter widget test.
import 'package:flutter_test/flutter_test.dart';
import 'package:mindsync_ai/main.dart';

void main() {
  testWidgets('MindSync app loads correctly', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MindSyncApp());

    // Verify that the app loads
    expect(find.text('MindSync AI'), findsOneWidget);
  });
}
