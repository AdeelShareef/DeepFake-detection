import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/detection_history.dart';
import '../services/storage_service.dart';

// Storage service provider
final storageServiceProvider = Provider<StorageService>((ref) => StorageService());

// History notifier
class HistoryNotifier extends StateNotifier<AsyncValue<List<DetectionHistory>>> {
  final StorageService _storageService;
  
  HistoryNotifier(this._storageService) : super(const AsyncValue.loading()) {
    loadHistory();
  }
  
  Future<void> loadHistory() async {
    state = const AsyncValue.loading();
    
    try {
      final history = await _storageService.loadHistory();
      state = AsyncValue.data(history);
    } catch (e) {
      state = AsyncValue.error(e, StackTrace.current);
    }
  }
  
  Future<void> addToHistory(DetectionHistory detection) async {
    try {
      await _storageService.addToHistory(detection);
      await loadHistory();
    } catch (e) {
      state = AsyncValue.error(e, StackTrace.current);
    }
  }
  
  Future<void> clearHistory() async {
    try {
      await _storageService.clearHistory();
      state = const AsyncValue.data([]);
    } catch (e) {
      state = AsyncValue.error(e, StackTrace.current);
    }
  }
}

// History notifier provider
final historyNotifierProvider = StateNotifierProvider<HistoryNotifier, AsyncValue<List<DetectionHistory>>>((ref) {
  final storageService = ref.watch(storageServiceProvider);
  return HistoryNotifier(storageService);
});
