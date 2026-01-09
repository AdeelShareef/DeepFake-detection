import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/detection_history.dart';

class StorageService {
  static const String _historyKey = 'detection_history';
  static const String _userPrefsKey = 'user_preferences';
  
  /// Save detection history
  Future<void> saveHistory(List<DetectionHistory> history) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final jsonList = history.map((h) => h.toJson()).toList();
      await prefs.setString(_historyKey, jsonEncode(jsonList));
    } catch (e) {
      throw Exception('Failed to save history: $e');
    }
  }
  
  /// Load detection history
  Future<List<DetectionHistory>> loadHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final jsonString = prefs.getString(_historyKey);
      
      if (jsonString == null) {
        return [];
      }
      
      final jsonList = jsonDecode(jsonString) as List;
      return jsonList
          .map((json) => DetectionHistory.fromJson(json as Map<String, dynamic>))
          .toList();
    } catch (e) {
      // Return empty list if parsing fails
      return [];
    }
  }
  
  /// Add detection to history
  Future<void> addToHistory(DetectionHistory detection) async {
    try {
      final history = await loadHistory();
      history.insert(0, detection); // Add to beginning
      
      // Keep only recent items
      if (history.length > 100) {
        history.removeRange(100, history.length);
      }
      
      await saveHistory(history);
    } catch (e) {
      throw Exception('Failed to add to history: $e');
    }
  }
  
  /// Clear history
  Future<void> clearHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_historyKey);
    } catch (e) {
      throw Exception('Failed to clear history: $e');
    }
  }
  
  /// Save user preference
  Future<void> savePreference(String key, dynamic value) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      
      if (value is String) {
        await prefs.setString(key, value);
      } else if (value is int) {
        await prefs.setInt(key, value);
      } else if (value is double) {
        await prefs.setDouble(key, value);
      } else if (value is bool) {
        await prefs.setBool(key, value);
      } else {
        await prefs.setString(key, jsonEncode(value));
      }
    } catch (e) {
      throw Exception('Failed to save preference: $e');
    }
  }
  
  /// Load user preference
  Future<T?> loadPreference<T>(String key) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      
      if (T == String) {
        return prefs.getString(key) as T?;
      } else if (T == int) {
        return prefs.getInt(key) as T?;
      } else if (T == double) {
        return prefs.getDouble(key) as T?;
      } else if (T == bool) {
        return prefs.getBool(key) as T?;
      } else {
        final jsonString = prefs.getString(key);
        if (jsonString != null) {
          return jsonDecode(jsonString) as T;
        }
      }
      
      return null;
    } catch (e) {
      return null;
    }
  }
  
  /// Clear all data
  Future<void> clearAll() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.clear();
    } catch (e) {
      throw Exception('Failed to clear storage: $e');
    }
  }
}
