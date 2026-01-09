import 'dart:io';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/user_profile.dart';

class AuthService {
  final SupabaseClient _supabase = Supabase.instance.client;
  
  /// Get current user
  User? get currentUser => _supabase.auth.currentUser;
  
  /// Check if user is authenticated
  bool get isAuthenticated => currentUser != null;
  
  /// Get auth state stream
  Stream<AuthState> get authStateChanges => _supabase.auth.onAuthStateChange;
  
  /// Sign up with email and password
  Future<UserProfile> signUp({
    required String email,
    required String password,
    required String name,
  }) async {
    try {
      final response = await _supabase.auth.signUp(
        email: email,
        password: password,
        data: {'name': name},
      );
      
      if (response.user == null) {
        throw Exception('Sign up failed. Please try again.');
      }
      
      // Create profile in profiles table
      try {
        await _supabase.from('profiles').insert({
          'id': response.user!.id,
          'email': email,
          'name': name,
          'created_at': DateTime.now().toIso8601String(),
        });
      } catch (e) {
        // If profile creation fails, it might be RLS policy issue
        // User is still created in auth, so we can continue
        print('Profile creation warning: $e');
      }
      
      return UserProfile(
        id: response.user!.id,
        email: email,
        name: name,
        createdAt: DateTime.now(),
      );
    } on AuthException catch (e) {
      // Check if user already exists
      if (e.message.toLowerCase().contains('already registered') ||
          e.message.toLowerCase().contains('already exists') ||
          e.statusCode == '422') {
        throw Exception('Account already exists. Please log in instead.');
      }
      throw Exception(_getAuthErrorMessage(e));
    } catch (e) {
      if (e.toString().contains('already exists')) {
        throw Exception('Account already exists. Please log in instead.');
      }
      throw Exception('Sign up failed: $e');
    }
  }
  
  /// Sign in with email and password
  Future<UserProfile> signIn({
    required String email,
    required String password,
  }) async {
    try {
      final response = await _supabase.auth.signInWithPassword(
        email: email,
        password: password,
      );
      
      if (response.user == null) {
        throw Exception('Sign in failed. Please check your credentials.');
      }
      
      // Try to fetch user profile, create if doesn't exist
      try {
        return await getUserProfile(response.user!.id);
      } catch (e) {
        // Profile doesn't exist, create it
        try {
          await _supabase.from('profiles').insert({
            'id': response.user!.id,
            'email': email,
            'name': email.split('@').first,
            'created_at': DateTime.now().toIso8601String(),
          });
          
          return UserProfile(
            id: response.user!.id,
            email: email,
            name: email.split('@').first,
            createdAt: DateTime.now(),
          );
        } catch (profileError) {
          // Return basic profile even if creation fails
          return UserProfile(
            id: response.user!.id,
            email: email,
            createdAt: DateTime.now(),
          );
        }
      }
    } on AuthException catch (e) {
      // Check if user doesn't exist
      if (e.message.toLowerCase().contains('invalid login') ||
          e.message.toLowerCase().contains('not found') ||
          e.message.toLowerCase().contains('invalid credentials') ||
          e.statusCode == '400') {
        throw Exception('No account found. Please sign up first.');
      }
      throw Exception(_getAuthErrorMessage(e));
    } catch (e) {
      if (e.toString().contains('No account found')) {
        rethrow;
      }
      throw Exception('Sign in failed: $e');
    }
  }
  
  /// Sign out
  Future<void> signOut() async {
    try {
      await _supabase.auth.signOut();
    } catch (e) {
      throw Exception('Sign out failed: $e');
    }
  }
  
  /// Get user profile from database
  Future<UserProfile> getUserProfile(String userId) async {
    try {
      final response = await _supabase
          .from('profiles')
          .select()
          .eq('id', userId)
          .single();
      
      return UserProfile.fromJson(response);
    } catch (e) {
      throw Exception('Failed to fetch user profile: $e');
    }
  }
  
  /// Update user profile
  Future<UserProfile> updateProfile({
    required String userId,
    String? name,
    String? avatarUrl,
  }) async {
    try {
      final updates = <String, dynamic>{
        'updated_at': DateTime.now().toIso8601String(),
      };
      
      if (name != null) updates['name'] = name;
      if (avatarUrl != null) updates['avatar_url'] = avatarUrl;
      
      await _supabase
          .from('profiles')
          .update(updates)
          .eq('id', userId);
      
      return await getUserProfile(userId);
    } catch (e) {
      throw Exception('Failed to update profile: $e');
    }
  }
  
  /// Reset password
  Future<void> resetPassword(String email) async {
    try {
      await _supabase.auth.resetPasswordForEmail(email);
    } on AuthException catch (e) {
      throw Exception(_getAuthErrorMessage(e));
    } catch (e) {
      throw Exception('Password reset failed: $e');
    }
  }
  
  /// Upload profile avatar
  Future<String> uploadAvatar(String userId, String filePath) async {
    try {
      final fileName = '$userId-${DateTime.now().millisecondsSinceEpoch}.jpg';
      
      await _supabase.storage
          .from('avatars')
          .upload(fileName, File(filePath));
      
      final avatarUrl = _supabase.storage
          .from('avatars')
          .getPublicUrl(fileName);
      
      return avatarUrl;
    } catch (e) {
      throw Exception('Failed to upload avatar: $e');
    }
  }
  
  /// Get user-friendly error message
  String _getAuthErrorMessage(AuthException e) {
    switch (e.statusCode) {
      case '400':
        return 'Invalid email or password';
      case '422':
        return 'Email already registered';
      case '429':
        return 'Too many attempts. Please try again later';
      default:
        return e.message;
    }
  }
}
