import 'package:flutter/material.dart';
import 'dart:math' as math;
import '../../config/theme.dart';

class ConfidenceMeter extends StatelessWidget {
  final double confidence;
  final String label;
  final Color? color;
  
  const ConfidenceMeter({
    super.key,
    required this.confidence,
    this.label = 'Confidence',
    this.color,
  });
  
  @override
  Widget build(BuildContext context) {
    final displayColor = color ?? AppTheme.primaryTeal;
    
    return Column(
      children: [
        // Circular progress
        SizedBox(
          width: 120,
          height: 120,
          child: Stack(
            alignment: Alignment.center,
            children: [
              // Background circle
              const SizedBox(
                width: 120,
                height: 120,
                child: CircularProgressIndicator(
                  value: 1.0,
                  strokeWidth: 12,
                  valueColor: AlwaysStoppedAnimation<Color>(
                    AppTheme.cardDark,
                  ),
                ),
              ),
              // Progress circle
              SizedBox(
                width: 120,
                height: 120,
                child: TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.0, end: confidence),
                  duration: const Duration(milliseconds: 1500),
                  curve: Curves.easeOutCubic,
                  builder: (context, value, child) {
                    return CircularProgressIndicator(
                      value: value,
                      strokeWidth: 12,
                      valueColor: AlwaysStoppedAnimation<Color>(displayColor),
                    );
                  },
                ),
              ),
              // Percentage text
              TweenAnimationBuilder<double>(
                tween: Tween(begin: 0.0, end: confidence * 100),
                duration: const Duration(milliseconds: 1500),
                curve: Curves.easeOutCubic,
                builder: (context, value, child) {
                  return Text(
                    '${value.toStringAsFixed(1)}%',
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: AppTheme.textPrimary,
                    ),
                  );
                },
              ),
            ],
          ),
        ),
        const SizedBox(height: AppTheme.spacingMd),
        Text(
          label,
          style: const TextStyle(
            fontSize: 14,
            color: AppTheme.textSecondary,
          ),
        ),
      ],
    );
  }
}

class LinearConfidenceMeter extends StatelessWidget {
  final double confidence;
  final String label;
  final Color? color;
  
  const LinearConfidenceMeter({
    super.key,
    required this.confidence,
    required this.label,
    this.color,
  });
  
  @override
  Widget build(BuildContext context) {
    final displayColor = color ?? AppTheme.primaryTeal;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 14,
                color: AppTheme.textSecondary,
              ),
            ),
            TweenAnimationBuilder<double>(
              tween: Tween(begin: 0.0, end: confidence * 100),
              duration: const Duration(milliseconds: 1000),
              curve: Curves.easeOutCubic,
              builder: (context, value, child) {
                return Text(
                  '${value.toStringAsFixed(1)}%',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: displayColor,
                  ),
                );
              },
            ),
          ],
        ),
        const SizedBox(height: AppTheme.spacingSm),
        ClipRRect(
          borderRadius: BorderRadius.circular(AppTheme.radiusSm),
          child: SizedBox(
            height: 8,
            child: Stack(
              children: [
                Container(
                  width: double.infinity,
                  color: AppTheme.cardDark,
                ),
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.0, end: confidence),
                  duration: const Duration(milliseconds: 1000),
                  curve: Curves.easeOutCubic,
                  builder: (context, value, child) {
                    return FractionallySizedBox(
                      widthFactor: value,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [displayColor, displayColor.withOpacity(0.7)],
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
