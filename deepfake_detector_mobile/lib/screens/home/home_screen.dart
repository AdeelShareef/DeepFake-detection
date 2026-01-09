import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../config/theme.dart';
import '../../providers/auth_provider.dart';
import '../../providers/profile_provider.dart';
import '../../widgets/common/custom_button.dart';
import '../detection/upload_screen.dart';
import '../profile/profile_screen.dart';
import '../profile/history_screen.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});
  
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authNotifierProvider);
    final historyState = ref.watch(historyNotifierProvider);
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('DeepShield'),
        actions: [
          IconButton(
            icon: const Icon(Icons.person),
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const ProfileScreen()),
              );
            },
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(AppTheme.spacingLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Welcome section
              authState.when(
                data: (profile) {
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Welcome back,',
                        style: TextStyle(
                          fontSize: 16,
                          color: AppTheme.textSecondary,
                        ),
                      ).animate().fadeIn(),
                      const SizedBox(height: AppTheme.spacingSm),
                      Text(
                        profile?.displayName ?? 'User',
                        style: const TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: AppTheme.textPrimary,
                        ),
                      ).animate().fadeIn(delay: 100.ms),
                    ],
                  );
                },
                loading: () => const SizedBox.shrink(),
                error: (_, __) => const SizedBox.shrink(),
              ),
              
              const SizedBox(height: AppTheme.spacingXl),
              
              // Hero section with gradient
              Container(
                decoration: BoxDecoration(
                  gradient: AppTheme.primaryGradient,
                  borderRadius: BorderRadius.circular(AppTheme.radiusLg),
                  boxShadow: [
                    BoxShadow(
                      color: AppTheme.primaryTeal.withOpacity(0.3),
                      blurRadius: 20,
                      offset: const Offset(0, 10),
                    ),
                  ],
                ),
                padding: const EdgeInsets.all(AppTheme.spacingXl),
                child: const Column(
                  children: [
                    Icon(
                      Icons.shield,
                      size: 60,
                      color: Colors.white,
                    ),
                    SizedBox(height: AppTheme.spacingMd),
                    Text(
                      'Detect Deepfakes',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    SizedBox(height: AppTheme.spacingSm),
                    Text(
                      'Upload an image or video to analyze',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.white70,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ).animate().fadeIn(delay: 200.ms).slideY(begin: 0.2, end: 0),
              
              const SizedBox(height: AppTheme.spacingXl),
              
              // Action buttons
              Row(
                children: [
                  Expanded(
                    child: _ActionCard(
                      icon: Icons.image,
                      title: 'Scan Image',
                      description: 'Upload a photo',
                      gradient: AppTheme.primaryGradient,
                      onTap: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => const UploadScreen(isVideo: false),
                          ),
                        );
                      },
                    ).animate().fadeIn(delay: 300.ms).slideX(begin: -0.2, end: 0),
                  ),
                  const SizedBox(width: AppTheme.spacingMd),
                  Expanded(
                    child: _ActionCard(
                      icon: Icons.videocam,
                      title: 'Scan Video',
                      description: 'Upload a video',
                      gradient: const LinearGradient(
                        colors: [Color(0xFF8B5CF6), Color(0xFFA78BFA)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      onTap: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => const UploadScreen(isVideo: true),
                          ),
                        );
                      },
                    ).animate().fadeIn(delay: 400.ms).slideX(begin: 0.2, end: 0),
                  ),
                ],
              ),
              
              const SizedBox(height: AppTheme.spacingXl),
              
              // Recent history section
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Recent Scans',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: AppTheme.textPrimary,
                    ),
                  ),
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(builder: (_) => const HistoryScreen()),
                      );
                    },
                    child: const Text('View All'),
                  ),
                ],
              ).animate().fadeIn(delay: 500.ms),
              
              const SizedBox(height: AppTheme.spacingMd),
              
              // History list
              historyState.when(
                data: (history) {
                  if (history.isEmpty) {
                    return Container(
                      padding: const EdgeInsets.all(AppTheme.spacingXl),
                      decoration: BoxDecoration(
                        color: AppTheme.surfaceDark,
                        borderRadius: BorderRadius.circular(AppTheme.radiusMd),
                      ),
                      child: const Column(
                        children: [
                          Icon(
                            Icons.history,
                            size: 48,
                            color: AppTheme.textTertiary,
                          ),
                          SizedBox(height: AppTheme.spacingMd),
                          Text(
                            'No scans yet',
                            style: TextStyle(
                              fontSize: 16,
                              color: AppTheme.textSecondary,
                            ),
                          ),
                          SizedBox(height: AppTheme.spacingSm),
                          Text(
                            'Upload an image or video to get started',
                            style: TextStyle(
                              fontSize: 14,
                              color: AppTheme.textTertiary,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ],
                      ),
                    ).animate().fadeIn(delay: 600.ms);
                  }
                  
                  return Column(
                    children: history.take(3).map((item) {
                      return _HistoryItem(
                        history: item,
                      ).animate().fadeIn(delay: 600.ms);
                    }).toList(),
                  );
                },
                loading: () => const Center(child: CircularProgressIndicator()),
                error: (_, __) => const SizedBox.shrink(),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;
  final LinearGradient gradient;
  final VoidCallback onTap;
  
  const _ActionCard({
    required this.icon,
    required this.title,
    required this.description,
    required this.gradient,
    required this.onTap,
  });
  
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          color: AppTheme.surfaceDark,
          borderRadius: BorderRadius.circular(AppTheme.radiusMd),
          border: Border.all(
            color: AppTheme.cardDark,
            width: 1,
          ),
        ),
        padding: const EdgeInsets.all(AppTheme.spacingLg),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(AppTheme.spacingMd),
              decoration: BoxDecoration(
                gradient: gradient,
                borderRadius: BorderRadius.circular(AppTheme.radiusSm),
              ),
              child: Icon(
                icon,
                size: 32,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: AppTheme.spacingMd),
            Text(
              title,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: AppTheme.textPrimary,
              ),
            ),
            const SizedBox(height: AppTheme.spacingSm),
            Text(
              description,
              style: const TextStyle(
                fontSize: 12,
                color: AppTheme.textSecondary,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

class _HistoryItem extends StatelessWidget {
  final dynamic history;
  
  const _HistoryItem({required this.history});
  
  @override
  Widget build(BuildContext context) {
    final resultColor = AppTheme.getResultColor(history.result.result);
    
    return Container(
      margin: const EdgeInsets.only(bottom: AppTheme.spacingMd),
      padding: const EdgeInsets.all(AppTheme.spacingMd),
      decoration: BoxDecoration(
        color: AppTheme.surfaceDark,
        borderRadius: BorderRadius.circular(AppTheme.radiusMd),
      ),
      child: Row(
        children: [
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              color: resultColor.withOpacity(0.2),
              borderRadius: BorderRadius.circular(AppTheme.radiusSm),
            ),
            child: Icon(
              history.result.isVideo ? Icons.videocam : Icons.image,
              color: resultColor,
            ),
          ),
          const SizedBox(width: AppTheme.spacingMd),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  history.fileName,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                    color: AppTheme.textPrimary,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: AppTheme.spacingSm),
                Text(
                  history.relativeTime,
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppTheme.textTertiary,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: AppTheme.spacingMd,
              vertical: AppTheme.spacingSm,
            ),
            decoration: BoxDecoration(
              color: resultColor.withOpacity(0.2),
              borderRadius: BorderRadius.circular(AppTheme.radiusSm),
            ),
            child: Text(
              history.result.result.toUpperCase(),
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: resultColor,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
