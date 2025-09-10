// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:rice_prediction_mobile/main.dart';

void main() {
  testWidgets('Rice Prediction App smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const RicePredictionApp());

    // Verify that our app shows the main title
    expect(find.text('Rice Variety Prediction'), findsOneWidget);
    expect(find.text('Upload Rice Image'), findsOneWidget);

    // Verify that the camera and gallery buttons are present
    expect(find.byIcon(Icons.camera), findsOneWidget);
    expect(find.byIcon(Icons.photo_library), findsOneWidget);

    // Verify that the predict button is present
    expect(find.text('Predict Rice Variety'), findsOneWidget);
  });
}
