import 'package:flutter/material.dart';
import '../../config/theme.dart';

class LoadingIndicator extends StatelessWidget {
  final String? message;
  final double size;
  
  const LoadingIndicator({
    super.key,
    this.message,
    this.size = 50,
  });
  
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: size,
            height: size,
            child: const CircularProgressIndicator(
              strokeWidth: 3,
              valueColor: AlwaysStoppedAnimation<Color>(
                AppTheme.primaryTeal,
              ),
            ),
          ),
          if (message != null) ...[
            const SizedBox(height: AppTheme.spacingMd),
            Text(
              message!,
              style: const TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 14,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ],
      ),
    );
  }
}

class GradientLoadingIndicator extends StatelessWidget {
  final String? message;
  final double size;
  
  const GradientLoadingIndicator({
    super.key,
    this.message,
    this.size = 50,
  });
  
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: size,
            height: size,
            child: ShaderMask(
              shaderCallback: (bounds) => AppTheme.primaryGradient.createShader(bounds),
              child: const CircularProgressIndicator(
                strokeWidth: 3,
                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
              ),
            ),
          ),
          if (message != null) ...[
            const SizedBox(height: AppTheme.spacingMd),
            Text(
              message!,
              style: const TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 14,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ],
      ),
    );
  }
}
