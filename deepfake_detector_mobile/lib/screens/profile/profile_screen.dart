import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'dart:io';
import '../../config/theme.dart';
import '../../providers/auth_provider.dart';
import '../../widgets/common/custom_button.dart';
import '../../widgets/common/custom_text_field.dart';
import '../../widgets/common/loading_indicator.dart';
import '../auth/login_screen.dart';
import 'history_screen.dart';

class ProfileScreen extends ConsumerStatefulWidget {
  const ProfileScreen({super.key});
  
  @override
  ConsumerState<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends ConsumerState<ProfileScreen> {
  final _nameController = TextEditingController();
  bool _isEditing = false;
  bool _isSaving = false;
  
  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }
  
  Future<void> _pickAvatar() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 512,
      maxHeight: 512,
    );
    
    if (image != null) {
      try {
        final avatarUrl = await ref.read(authNotifierProvider.notifier).uploadAvatar(image.path);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Profile picture updated'),
              backgroundColor: AppTheme.accentGreen,
            ),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Failed to upload avatar: $e'),
              backgroundColor: AppTheme.accentRed,
            ),
          );
        }
      }
    }
  }
  
  Future<void> _saveProfile() async {
    setState(() => _isSaving = true);
    
    try {
      await ref.read(authNotifierProvider.notifier).updateProfile(
        name: _nameController.text.trim(),
      );
      
      setState(() => _isEditing = false);
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Profile updated successfully'),
            backgroundColor: AppTheme.accentGreen,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to update profile: $e'),
            backgroundColor: AppTheme.accentRed,
          ),
        );
      }
    } finally {
      setState(() => _isSaving = false);
    }
  }
  
  Future<void> _handleLogout() async {
    try {
      await ref.read(authNotifierProvider.notifier).signOut();
      
      if (mounted) {
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (_) => const LoginScreen()),
          (route) => false,
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to logout: $e'),
            backgroundColor: AppTheme.accentRed,
          ),
        );
      }
    }
  }
  
  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authNotifierProvider);
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile'),
        actions: [
          if (!_isEditing)
            IconButton(
              icon: const Icon(Icons.edit),
              onPressed: () {
                final profile = authState.value;
                if (profile != null) {
                  _nameController.text = profile.name ?? '';
                  setState(() => _isEditing = true);
                }
              },
            )
          else
            IconButton(
              icon: const Icon(Icons.close),
              onPressed: () => setState(() => _isEditing = false),
            ),
        ],
      ),
      body: authState.when(
        data: (profile) {
          if (profile == null) {
            return const Center(child: Text('Not logged in'));
          }
          
          return SingleChildScrollView(
            padding: const EdgeInsets.all(AppTheme.spacingLg),
            child: Column(
              children: [
                // Avatar
                GestureDetector(
                  onTap: _pickAvatar,
                  child: Stack(
                    children: [
                      Container(
                        width: 120,
                        height: 120,
                        decoration: const BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: AppTheme.primaryGradient,
                        ),
                        child: profile.avatarUrl != null
                            ? ClipOval(
                                child: Image.network(
                                  profile.avatarUrl!,
                                  fit: BoxFit.cover,
                                ),
                              )
                            : Center(
                                child: Text(
                                  profile.initials,
                                  style: const TextStyle(
                                    fontSize: 40,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white,
                                  ),
                                ),
                              ),
                      ),
                      Positioned(
                        bottom: 0,
                        right: 0,
                        child: Container(
                          padding: const EdgeInsets.all(AppTheme.spacingSm),
                          decoration: BoxDecoration(
                            color: AppTheme.primaryTeal,
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: AppTheme.backgroundDark,
                              width: 3,
                            ),
                          ),
                          child: const Icon(
                            Icons.camera_alt,
                            size: 20,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                ).animate().scale(
                  duration: AppTheme.animationNormal,
                  curve: Curves.easeOutBack,
                ),
                
                const SizedBox(height: AppTheme.spacingLg),
                
                if (!_isEditing) ...[
                  Text(
                    profile.displayName,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: AppTheme.textPrimary,
                    ),
                  ).animate().fadeIn(delay: 100.ms),
                  const SizedBox(height: AppTheme.spacingSm),
                  Text(
                    profile.email,
                    style: const TextStyle(
                      fontSize: 16,
                      color: AppTheme.textSecondary,
                    ),
                  ).animate().fadeIn(delay: 200.ms),
                ] else ...[
                  CustomTextField(
                    label: 'Name',
                    controller: _nameController,
                    prefixIcon: Icons.person,
                  ),
                  const SizedBox(height: AppTheme.spacingLg),
                  CustomButton(
                    text: 'Save Changes',
                    onPressed: _saveProfile,
                    isLoading: _isSaving,
                  ),
                ],
                
                const SizedBox(height: AppTheme.spacingXl),
                
                // Menu items
                _MenuItem(
                  icon: Icons.history,
                  title: 'Detection History',
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(builder: (_) => const HistoryScreen()),
                    );
                  },
                ).animate().fadeIn(delay: 300.ms),
                
                const SizedBox(height: AppTheme.spacingMd),
                
                _MenuItem(
                  icon: Icons.logout,
                  title: 'Logout',
                  textColor: AppTheme.accentRed,
                  onTap: _handleLogout,
                ).animate().fadeIn(delay: 400.ms),
              ],
            ),
          );
        },
        loading: () => const LoadingIndicator(message: 'Loading profile...'),
        error: (error, _) => Center(
          child: Text('Error: $error'),
        ),
      ),
    );
  }
}

class _MenuItem extends StatelessWidget {
  final IconData icon;
  final String title;
  final Color? textColor;
  final VoidCallback onTap;
  
  const _MenuItem({
    required this.icon,
    required this.title,
    this.textColor,
    required this.onTap,
  });
  
  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppTheme.radiusMd),
      child: Container(
        padding: const EdgeInsets.all(AppTheme.spacingMd),
        decoration: BoxDecoration(
          color: AppTheme.surfaceDark,
          borderRadius: BorderRadius.circular(AppTheme.radiusMd),
        ),
        child: Row(
          children: [
            Icon(
              icon,
              color: textColor ?? AppTheme.textPrimary,
            ),
            const SizedBox(width: AppTheme.spacingMd),
            Expanded(
              child: Text(
                title,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  color: textColor ?? AppTheme.textPrimary,
                ),
              ),
            ),
            const Icon(
              Icons.chevron_right,
              color: AppTheme.textSecondary,
            ),
          ],
        ),
      ),
    );
  }
}
