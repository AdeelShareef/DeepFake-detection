import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../config/theme.dart';
import '../../providers/profile_provider.dart';
import '../../widgets/common/loading_indicator.dart';
import '../detection/result_screen.dart';

class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});
  
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final historyState = ref.watch(historyNotifierProvider);
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection History'),
        actions: [
          historyState.whenOrNull(
            data: (history) => history.isNotEmpty
                ? IconButton(
                    icon: const Icon(Icons.delete_outline),
                    onPressed: () async {
                      final confirm = await showDialog<bool>(
                        context: context,
                        builder: (context) => AlertDialog(
                          title: const Text('Clear History'),
                          content: const Text(
                            'Are you sure you want to clear all detection history?',
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.of(context).pop(false),
                              child: const Text('Cancel'),
                            ),
                            TextButton(
                              onPressed: () => Navigator.of(context).pop(true),
                              child: const Text(
                                'Clear',
                                style: TextStyle(color: AppTheme.accentRed),
                              ),
                            ),
                          ],
                        ),
                      );
                      
                      if (confirm == true) {
                        await ref.read(historyNotifierProvider.notifier).clearHistory();
                      }
                    },
                  )
                : null,
          ) ?? const SizedBox.shrink(),
        ],
      ),
      body: historyState.when(
        data: (history) {
          if (history.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.history,
                    size: 80,
                    color: AppTheme.textTertiary,
                  ).animate().scale(
                    duration: AppTheme.animationNormal,
                    curve: Curves.easeOutBack,
                  ),
                  const SizedBox(height: AppTheme.spacingLg),
                  const Text(
                    'No detection history',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: AppTheme.textPrimary,
                    ),
                  ).animate().fadeIn(delay: 100.ms),
                  const SizedBox(height: AppTheme.spacingSm),
                  const Text(
                    'Your scanned images and videos\nwill appear here',
                    style: TextStyle(
                      fontSize: 14,
                      color: AppTheme.textSecondary,
                    ),
                    textAlign: TextAlign.center,
                  ).animate().fadeIn(delay: 200.ms),
                ],
              ),
            );
          }
          
          return ListView.builder(
            padding: const EdgeInsets.all(AppTheme.spacingLg),
            itemCount: history.length,
            itemBuilder: (context, index) {
              final item = history[index];
              final resultColor = AppTheme.getResultColor(item.result.result);
              
              return Container(
                margin: const EdgeInsets.only(bottom: AppTheme.spacingMd),
                child: InkWell(
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => ResultScreen(result: item.result),
                      ),
                    );
                  },
                  borderRadius: BorderRadius.circular(AppTheme.radiusMd),
                  child: Container(
                    padding: const EdgeInsets.all(AppTheme.spacingMd),
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceDark,
                      borderRadius: BorderRadius.circular(AppTheme.radiusMd),
                      border: Border.all(
                        color: resultColor.withOpacity(0.3),
                        width: 1,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 60,
                              height: 60,
                              decoration: BoxDecoration(
                                color: resultColor.withOpacity(0.2),
                                borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                              ),
                              child: Icon(
                                item.result.isVideo ? Icons.videocam : Icons.image,
                                color: resultColor,
                                size: 30,
                              ),
                            ),
                            const SizedBox(width: AppTheme.spacingMd),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    item.fileName,
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w600,
                                      color: AppTheme.textPrimary,
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                  const SizedBox(height: AppTheme.spacingSm),
                                  Row(
                                    children: [
                                      const Icon(
                                        Icons.access_time,
                                        size: 14,
                                        color: AppTheme.textTertiary,
                                      ),
                                      const SizedBox(width: AppTheme.spacingSm),
                                      Text(
                                        item.relativeTime,
                                        style: const TextStyle(
                                          fontSize: 12,
                                          color: AppTheme.textTertiary,
                                        ),
                                      ),
                                      const SizedBox(width: AppTheme.spacingMd),
                                      const Icon(
                                        Icons.storage,
                                        size: 14,
                                        color: AppTheme.textTertiary,
                                      ),
                                      const SizedBox(width: AppTheme.spacingSm),
                                      Text(
                                        item.fileSizeFormatted,
                                        style: const TextStyle(
                                          fontSize: 12,
                                          color: AppTheme.textTertiary,
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: AppTheme.spacingMd),
                        Container(
                          padding: const EdgeInsets.all(AppTheme.spacingMd),
                          decoration: BoxDecoration(
                            color: AppTheme.cardDark,
                            borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text(
                                    'Result',
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: AppTheme.textTertiary,
                                    ),
                                  ),
                                  const SizedBox(height: AppTheme.spacingSm),
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
                                      item.result.result.toUpperCase(),
                                      style: TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.bold,
                                        color: resultColor,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.end,
                                children: [
                                  const Text(
                                    'Confidence',
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: AppTheme.textTertiary,
                                    ),
                                  ),
                                  const SizedBox(height: AppTheme.spacingSm),
                                  Text(
                                    item.result.confidencePercentage,
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: resultColor,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ).animate().fadeIn(
                delay: Duration(milliseconds: 50 * index),
              ).slideX(
                begin: -0.1,
                end: 0,
                duration: AppTheme.animationNormal,
              );
            },
          );
        },
        loading: () => const LoadingIndicator(message: 'Loading history...'),
        error: (error, _) => Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.error_outline,
                size: 60,
                color: AppTheme.accentRed,
              ),
              const SizedBox(height: AppTheme.spacingMd),
              const Text(
                'Failed to load history',
                style: TextStyle(
                  fontSize: 18,
                  color: AppTheme.textPrimary,
                ),
              ),
              const SizedBox(height: AppTheme.spacingSm),
              Text(
                error.toString(),
                style: const TextStyle(
                  fontSize: 14,
                  color: AppTheme.textSecondary,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
