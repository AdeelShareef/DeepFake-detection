import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import '../models/detection_result.dart';
import '../services/api_service.dart';

// API service provider
final apiServiceProvider = Provider<ApiService>((ref) => ApiService());

// Detection state
class DetectionState {
  final DetectionResult? result;
  final bool isLoading;
  final double uploadProgress;
  final String? error;
  
  const DetectionState({
    this.result,
    this.isLoading = false,
    this.uploadProgress = 0.0,
    this.error,
  });
  
  DetectionState copyWith({
    DetectionResult? result,
    bool? isLoading,
    double? uploadProgress,
    String? error,
  }) {
    return DetectionState(
      result: result ?? this.result,
      isLoading: isLoading ?? this.isLoading,
      uploadProgress: uploadProgress ?? this.uploadProgress,
      error: error ?? this.error,
    );
  }
}

// Detection notifier
class DetectionNotifier extends StateNotifier<DetectionState> {
  final ApiService _apiService;
  
  DetectionNotifier(this._apiService) : super(const DetectionState());
  
  Future<void> analyzeMedia(XFile file) async {
    // Reset state
    state = const DetectionState(isLoading: true, uploadProgress: 0.0);
    
    try {
      final result = await _apiService.predictMedia(
        file,
        onProgress: (progress) {
          state = state.copyWith(uploadProgress: progress);
        },
      );
      
      state = DetectionState(
        result: result,
        isLoading: false,
        uploadProgress: 1.0,
      );
    } catch (e) {
      state = DetectionState(
        isLoading: false,
        error: e.toString().replaceFirst('Exception: ', ''),
      );
      rethrow;
    }
  }
  
  void reset() {
    state = const DetectionState();
  }
  
  void clearError() {
    state = state.copyWith(error: null);
  }
}

// Detection notifier provider
final detectionNotifierProvider = StateNotifierProvider<DetectionNotifier, DetectionState>((ref) {
  final apiService = ref.watch(apiServiceProvider);
  return DetectionNotifier(apiService);
});
