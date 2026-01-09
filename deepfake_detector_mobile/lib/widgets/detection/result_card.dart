import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../config/theme.dart';
import '../../models/detection_result.dart';

class ResultCard extends StatelessWidget {
  final DetectionResult result;
  
  const ResultCard({
    super.key,
    required this.result,
  });
  
  @override
  Widget build(BuildContext context) {
    final resultColor = AppTheme.getResultColor(result.result);
    final gradient = AppTheme.getResultGradient(result.result);
    
    return Container(
      decoration: BoxDecoration(
        gradient: gradient,
        borderRadius: BorderRadius.circular(AppTheme.radiusLg),
        boxShadow: [
          BoxShadow(
            color: resultColor.withOpacity(0.3),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(AppTheme.spacingXl),
        child: Column(
          children: [
            // Result icon
            Icon(
              _getResultIcon(),
              size: 80,
              color: Colors.white,
            ).animate().scale(
              duration: AppTheme.animationNormal,
              curve: Curves.easeOutBack,
            ),
            
            const SizedBox(height: AppTheme.spacingMd),
            
            // Result text
            Text(
              result.result.toUpperCase(),
              style: const TextStyle(
                fontSize: 36,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ).animate().fadeIn(
              duration: AppTheme.animationNormal,
              delay: 100.ms,
            ),
            
            const SizedBox(height: AppTheme.spacingSm),
            
            // Confidence
            Text(
              'Confidence: ${result.confidencePercentage}',
              style: const TextStyle(
                fontSize: 18,
                color: Colors.white,
                fontWeight: FontWeight.w500,
              ),
            ).animate().fadeIn(
              duration: AppTheme.animationNormal,
              delay: 200.ms,
            ),
            
            if (result.isVideo) ...[
              const SizedBox(height: AppTheme.spacingSm),
              Text(
                '${result.fakeRatioPercentage} of frames flagged',
                style: const TextStyle(
                  fontSize: 14,
                  color: Colors.white70,
                ),
              ).animate().fadeIn(
                duration: AppTheme.animationNormal,
                delay: 300.ms,
              ),
            ],
          ],
        ),
      ),
    ).animate().fadeIn(
      duration: AppTheme.animationSlow,
    ).slideY(
      begin: 0.2,
      end: 0,
      duration: AppTheme.animationSlow,
      curve: Curves.easeOutCubic,
    );
  }
  
  IconData _getResultIcon() {
    switch (result.result.toUpperCase()) {
      case 'REAL':
        return Icons.check_circle;
      case 'FAKE':
        return Icons.warning;
      case 'UNCERTAIN':
        return Icons.help;
      default:
        return Icons.info;
    }
  }
}
