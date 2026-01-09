import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:image_picker/image_picker.dart';
import '../config/app_config.dart';
import '../models/detection_result.dart';

class ApiService {
  late final Dio _dio;
  
  ApiService() {
    _dio = Dio(
      BaseOptions(
        baseUrl: AppConfig.baseUrl,
        connectTimeout: AppConfig.connectionTimeout,
        receiveTimeout: AppConfig.receiveTimeout,
        headers: {
          'Accept': 'application/json',
        },
      ),
    );
    
    // Add logging interceptor for debugging
    _dio.interceptors.add(
      LogInterceptor(
        requestBody: true,
        responseBody: true,
        error: true,
      ),
    );
  }
  
  /// Upload media file and get deepfake prediction
  /// Returns DetectionResult on success
  /// Throws Exception on failure
  Future<DetectionResult> predictMedia(
    XFile file, {
    Function(double)? onProgress,
  }) async {
    try {
      // Get file size and validate
      final fileSize = await file.length();
      final fileName = file.name;
      final extension = fileName.split('.').last.toLowerCase();
      
      // Validate file format
      if (!_isValidFormat(extension)) {
        throw Exception(
          'Unsupported file format. Supported formats: '
          '${AppConfig.supportedImageFormats.join(', ')}, '
          '${AppConfig.supportedVideoFormats.join(', ')}',
        );
      }
      
      // Validate file size
      final isImage = AppConfig.supportedImageFormats.contains(extension);
      final maxSize = isImage ? AppConfig.maxImageSize : AppConfig.maxVideoSize;
      
      if (fileSize > maxSize) {
        final maxSizeMB = (maxSize / (1024 * 1024)).toStringAsFixed(0);
        throw Exception(
          'File size exceeds limit. Maximum size for ${isImage ? 'images' : 'videos'}: $maxSizeMB MB',
        );
      }
      
      // Create multipart form data
      final bytes = await file.readAsBytes();
      final formData = FormData.fromMap({
        'mediaFile': MultipartFile.fromBytes(
          bytes,
          filename: fileName,
        ),
      });
      
      print('=== API REQUEST ===');
      print('URL: ${AppConfig.baseUrl}${AppConfig.uploadEndpoint}');
      print('File: $fileName');
      print('Size: ${(fileSize / 1024).toStringAsFixed(2)} KB');
      
      // Upload file with progress tracking
      final response = await _dio.post(
        AppConfig.uploadEndpoint,
        data: formData,
        onSendProgress: (sent, total) {
          if (onProgress != null && total > 0) {
            final progress = sent / total;
            onProgress(progress);
            print('Upload progress: ${(progress * 100).toStringAsFixed(1)}%');
          }
        },
      );
      
      print('=== API RESPONSE ===');
      print('Status Code: ${response.statusCode}');
      print('Response Data: ${response.data}');
      
      // Parse response
      if (response.statusCode == 200) {
        final data = response.data;
        
        // Handle different response formats
        if (data is Map<String, dynamic>) {
          print('Parsed result: ${data['result']}');
          print('Confidence: ${data['confidence']}');
          return DetectionResult.fromJson(data);
        } else if (data is String) {
          // Try to parse JSON string
          try {
            final jsonData = jsonDecode(data) as Map<String, dynamic>;
            print('Parsed from string - result: ${jsonData['result']}');
            return DetectionResult.fromJson(jsonData);
          } catch (e) {
            print('Failed to parse JSON string: $e');
            throw Exception('Invalid response format from server: $data');
          }
        } else {
          print('Unexpected data type: ${data.runtimeType}');
          throw Exception('Invalid response format from server');
        }
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } on DioException catch (e) {
      print('=== DIO ERROR ===');
      print('Type: ${e.type}');
      print('Message: ${e.message}');
      print('Response: ${e.response?.data}');
      
      // Handle Dio-specific errors
      if (e.type == DioExceptionType.connectionTimeout) {
        throw Exception('Connection timeout. Please check your internet connection and backend URL.');
      } else if (e.type == DioExceptionType.receiveTimeout) {
        throw Exception('Server took too long to respond. Please try again.');
      } else if (e.type == DioExceptionType.badResponse) {
        final statusCode = e.response?.statusCode;
        final message = e.response?.data?['error'] ?? e.response?.data ?? 'Server error';
        throw Exception('Server error ($statusCode): $message');
      } else if (e.type == DioExceptionType.cancel) {
        throw Exception('Upload cancelled');
      } else if (e.type == DioExceptionType.connectionError) {
        throw Exception('Cannot connect to backend at ${AppConfig.baseUrl}. Please check:\n1. Backend is running\n2. URL is correct\n3. Network connection');
      } else {
        throw Exception('Network error: ${e.message}');
      }
    } catch (e) {
      print('=== UNEXPECTED ERROR ===');
      print('Error: $e');
      
      // Re-throw if already an Exception
      if (e is Exception) {
        rethrow;
      }
      throw Exception('Unexpected error: $e');
    }
  }
  
  /// Validate file format
  bool _isValidFormat(String extension) {
    return AppConfig.supportedImageFormats.contains(extension) ||
        AppConfig.supportedVideoFormats.contains(extension);
  }
  
  /// Cancel ongoing upload
  void cancelUpload() {
    _dio.close(force: true);
  }
}
