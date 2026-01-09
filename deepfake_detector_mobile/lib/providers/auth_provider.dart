import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/user_profile.dart';
import '../services/auth_service.dart';

// Auth service provider
final authServiceProvider = Provider<AuthService>((ref) => AuthService());

// Auth state provider
final authStateProvider = StreamProvider<AuthState>((ref) {
  final authService = ref.watch(authServiceProvider);
  return authService.authStateChanges;
});

// Current user provider
final currentUserProvider = Provider<User?>((ref) {
  final authService = ref.watch(authServiceProvider);
  return authService.currentUser;
});

// User profile provider
final userProfileProvider = FutureProvider<UserProfile?>((ref) async {
  final user = ref.watch(currentUserProvider);
  
  if (user == null) {
    return null;
  }
  
  final authService = ref.watch(authServiceProvider);
  try {
    return await authService.getUserProfile(user.id);
  } catch (e) {
    return null;
  }
});

// Auth state notifier
class AuthNotifier extends StateNotifier<AsyncValue<UserProfile?>> {
  final AuthService _authService;
  
  AuthNotifier(this._authService) : super(const AsyncValue.loading()) {
    _init();
  }
  
  void _init() async {
    if (_authService.isAuthenticated) {
      try {
        final profile = await _authService.getUserProfile(
          _authService.currentUser!.id,
        );
        state = AsyncValue.data(profile);
      } catch (e) {
        state = AsyncValue.error(e, StackTrace.current);
      }
    } else {
      state = const AsyncValue.data(null);
    }
  }
  
  Future<void> signUp({
    required String email,
    required String password,
    required String name,
  }) async {
    state = const AsyncValue.loading();
    
    try {
      final profile = await _authService.signUp(
        email: email,
        password: password,
        name: name,
      );
      state = AsyncValue.data(profile);
    } catch (e) {
      state = AsyncValue.error(e, StackTrace.current);
      rethrow;
    }
  }
  
  Future<void> signIn({
    required String email,
    required String password,
  }) async {
    state = const AsyncValue.loading();
    
    try {
      final profile = await _authService.signIn(
        email: email,
        password: password,
      );
      state = AsyncValue.data(profile);
    } catch (e) {
      state = AsyncValue.error(e, StackTrace.current);
      rethrow;
    }
  }
  
  Future<void> signOut() async {
    try {
      await _authService.signOut();
      state = const AsyncValue.data(null);
    } catch (e) {
      state = AsyncValue.error(e, StackTrace.current);
      rethrow;
    }
  }
  
  Future<void> updateProfile({
    String? name,
    String? avatarUrl,
  }) async {
    final currentProfile = state.value;
    if (currentProfile == null) return;
    
    try {
      final updatedProfile = await _authService.updateProfile(
        userId: currentProfile.id,
        name: name,
        avatarUrl: avatarUrl,
      );
      state = AsyncValue.data(updatedProfile);
    } catch (e) {
      state = AsyncValue.error(e, StackTrace.current);
      rethrow;
    }
  }
  
  Future<String> uploadAvatar(String filePath) async {
    final currentProfile = state.value;
    if (currentProfile == null) {
      throw Exception('No user logged in');
    }
    
    try {
      final avatarUrl = await _authService.uploadAvatar(
        currentProfile.id,
        filePath,
      );
      await updateProfile(avatarUrl: avatarUrl);
      return avatarUrl;
    } catch (e) {
      rethrow;
    }
  }
}

// Auth notifier provider
final authNotifierProvider = StateNotifierProvider<AuthNotifier, AsyncValue<UserProfile?>>((ref) {
  final authService = ref.watch(authServiceProvider);
  return AuthNotifier(authService);
});
