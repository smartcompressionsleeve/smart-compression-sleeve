// lib/main.dart

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'screens/homescreen.dart';
import 'screens/calibration_screen.dart';
import 'screens/live_session_screen.dart';
import 'screens/exercise_select_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Supabase.initialize(
    url: 'https://alucpjqomzuxbuzkfgce.supabase.co',
    anonKey: 'sb_publishable_GWS8MTRzWUlUDOr7aKA5hA_0BBOU3YI',
  );

  runApp(const SmartCompressionSleeveApp());
}

class SmartCompressionSleeveApp extends StatelessWidget {
  const SmartCompressionSleeveApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Smart Compression Sleeve',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF4F8FF),
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1B63E9)),
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          elevation: 0,
          backgroundColor: Color(0xFF1B63E9),
          foregroundColor: Colors.white,
        ),
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const HomeScreen(),
        '/exercise-select': (context) => const ExerciseSelectScreen(),
        '/calibration': (context) => const CalibrationScreen(),
        '/live-session': (context) =>
            const LiveSessionScreen(calibratedThreshold: 0.0),
      },
      onUnknownRoute: (settings) {
        return MaterialPageRoute(builder: (context) => const HomeScreen());
      },
    );
  }
}
