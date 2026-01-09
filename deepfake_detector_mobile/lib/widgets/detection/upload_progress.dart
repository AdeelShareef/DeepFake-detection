import 'package:flutter/material.dart';
import '../../config/theme.dart';

class UploadProgress extends StatelessWidget {
  final double progress;
  final String? message;
  
  const UploadProgress({
    super.key,
    required this.progress,
    this.message,
  });
  
  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Circular progress
        SizedBox(
          width: 100,
          height: 100,
          child: Stack(
            alignment: Alignment.center,
            children: [
              SizedBox(
                width: 100,
                height: 100,
                child: CircularProgressIndicator(
                  value: progress,
                  strokeWidth: 8,
                  backgroundColor: AppTheme.cardDark,
                  valueColor: const AlwaysStoppedAnimation<Color>(
                    AppTheme.primaryTeal,
                  ),
                ),
              ),
              Text(
                '${(progress * 100).toStringAsFixed(0)}%',
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: AppTheme.textPrimary,
                ),
              ),
            ],
          ),
        ),
        
        const SizedBox(height: AppTheme.spacingLg),
        
        // Progress bar
        ClipRRect(
          borderRadius: BorderRadius.circular(AppTheme.radiusSm),
          child: SizedBox(
            height: 6,
            width: 200,
            child: LinearProgressIndicator(
              value: progress,
              backgroundColor: AppTheme.cardDark,
              valueColor: const AlwaysStoppedAnimation<Color>(
                AppTheme.primaryTeal,
              ),
            ),
          ),
        ),
        
        if (message != null) ...[
          const SizedBox(height: AppTheme.spacingMd),
          Text(
            message!,
            style: const TextStyle(
              fontSize: 14,
              color: AppTheme.textSecondary,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ],
    );
  }
}
