import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io' show Platform;

class AppConfig {
  // Backend API Configuration
  // IMPORTANT: Change this based on your setup
  static String get baseUrl {
    if (kIsWeb) {
      // Web platform - use localhost
      return 'http://localhost/deepshield';
    } else {
      try {
        if (Platform.isAndroid) {
          // For Android:
          // - Emulator: use 10.0.2.2 (special alias for host machine)
          // - Physical device: use your computer's IP address
          
          // OPTION 1: For Android Emulator (uncomment this line)
          // return 'http://10.0.2.2/deepshield';
          
          // OPTION 2: For Physical Android Device (uncomment this line)
          return 'http://10.135.167.97/deepshield';  // Your computer's WiFi IP
          
          // To find your IP: Run 'ipconfig' in PowerShell and look for WiFi IPv4 Address
        } else {
          // iOS, Windows, macOS, Linux
          return 'http://localhost/deepshield';
        }
      } catch (e) {
        // Fallback to localhost
        return 'http://localhost/deepshield';
      }
    }
  }
  
  static const String uploadEndpoint = '/upload_handler.php';
  
  // Supabase Configuration
  static const String supabaseUrl = 'https://ibuwaynqglbtcjhnzuyt.supabase.co';
  static const String supabaseAnonKey = 
      'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlidXdheW5xZ2xidGNqaG56dXl0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njc1MDg5NjcsImV4cCI6MjA4MzA4NDk2N30.a1poZZLTyLU4rJ1vpU78I_Hs8yms6kcX_Gi4_mekfcg';
  
  // File Size Limits (in bytes)
  static const int maxImageSize = 5 * 1024 * 1024; // 5 MB
  static const int maxVideoSize = 50 * 1024 * 1024; // 50 MB
  
  // Supported File Formats
  static const List<String> supportedImageFormats = ['jpg', 'jpeg', 'png'];
  static const List<String> supportedVideoFormats = ['mp4', 'avi', 'mov', 'webm'];
  
  // API Timeouts
  static const Duration connectionTimeout = Duration(seconds: 30);
  static const Duration receiveTimeout = Duration(minutes: 5);
  
  // App Settings
  static const String appName = 'DeepShield';
  static const int maxHistoryItems = 100;
}
