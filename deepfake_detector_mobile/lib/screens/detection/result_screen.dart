import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../config/theme.dart';
import '../../models/detection_result.dart';
import '../../widgets/detection/result_card.dart';
import '../../widgets/detection/confidence_meter.dart';
import '../../widgets/common/custom_button.dart';
import '../home/home_screen.dart';

class ResultScreen extends StatelessWidget {
  final DetectionResult result;
  final String? filePath;
  
  const ResultScreen({
    super.key,
    required this.result,
    this.filePath,
  });
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection Result'),
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: () {
            Navigator.of(context).pushAndRemoveUntil(
              MaterialPageRoute(builder: (_) => const HomeScreen()),
              (route) => false,
            );
          },
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(AppTheme.spacingLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // File preview - responsive sizing
              if (filePath != null && !result.isVideo)
                LayoutBuilder(
                  builder: (context, constraints) {
                    final imageHeight = (MediaQuery.of(context).size.height * 0.25).clamp(150.0, 300.0);
                    return ClipRRect(
                      borderRadius: BorderRadius.circular(AppTheme.radiusLg),
                      child: ConstrainedBox(
                        constraints: BoxConstraints(
                          maxHeight: imageHeight,
                          maxWidth: constraints.maxWidth,
                        ),
                        child: Image.network(
                          filePath!,
                          fit: BoxFit.cover,
                          errorBuilder: (context, error, stackTrace) {
                            return Container(
                              height: imageHeight,
                              color: AppTheme.surfaceDark,
                              child: const Center(
                                child: Icon(Icons.image, size: 60, color: AppTheme.textSecondary),
                              ),
                            );
                          },
                        ),
                      ),
                    ).animate().fadeIn();
                  },
                ),
              
              if (filePath != null && result.isVideo)
                LayoutBuilder(
                  builder: (context, constraints) {
                    final videoHeight = (MediaQuery.of(context).size.height * 0.25).clamp(150.0, 250.0);
                    return Container(
                      height: videoHeight,
                      decoration: BoxDecoration(
                        gradient: AppTheme.primaryGradient,
                        borderRadius: BorderRadius.circular(AppTheme.radiusLg),
                      ),
                      child: Center(
                        child: Icon(
                          Icons.videocam,
                          size: (videoHeight * 0.4).clamp(50.0, 80.0),
                          color: Colors.white,
                        ),
                      ),
                    ).animate().fadeIn();
                  },
                ),
              
              const SizedBox(height: AppTheme.spacingXl),
              
              // Result card
              ResultCard(result: result),
              
              const SizedBox(height: AppTheme.spacingXl),
              
              // Detailed analysis
              Container(
                padding: const EdgeInsets.all(AppTheme.spacingLg),
                decoration: BoxDecoration(
                  color: AppTheme.surfaceDark,
                  borderRadius: BorderRadius.circular(AppTheme.radiusMd),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Detailed Analysis',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: AppTheme.spacingLg),
                    
                    // Confidence meter
                    Center(
                      child: ConfidenceMeter(
                        confidence: result.confidence,
                        label: 'Detection Confidence',
                        color: AppTheme.getResultColor(result.result),
                      ),
                    ),
                    
                    const SizedBox(height: AppTheme.spacingXl),
                    
                    // Additional details
                    _DetailRow(
                      label: 'Media Type',
                      value: result.isVideo ? 'Video' : 'Image',
                      icon: result.isVideo ? Icons.videocam : Icons.image,
                    ),
                    
                    if (result.isVideo) ...[
                      const SizedBox(height: AppTheme.spacingMd),
                      _DetailRow(
                        label: 'Frames Analyzed',
                        value: result.framesUsed.toString(),
                        icon: Icons.grid_on,
                      ),
                      const SizedBox(height: AppTheme.spacingMd),
                      LinearConfidenceMeter(
                        confidence: result.fakeRatio,
                        label: 'Fake Frames Ratio',
                        color: AppTheme.accentRed,
                      ),
                    ],
                    
                    const SizedBox(height: AppTheme.spacingLg),
                    
                    // Explanation
                    Container(
                      padding: const EdgeInsets.all(AppTheme.spacingMd),
                      decoration: BoxDecoration(
                        color: AppTheme.cardDark,
                        borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                      ),
                      child: Row(
                        children: [
                          const Icon(
                            Icons.info_outline,
                            color: AppTheme.primaryTeal,
                            size: 20,
                          ),
                          const SizedBox(width: AppTheme.spacingMd),
                          Expanded(
                            child: Text(
                              _getExplanation(),
                              style: const TextStyle(
                                fontSize: 14,
                                color: AppTheme.textSecondary,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ).animate().fadeIn(delay: 500.ms).slideY(begin: 0.2, end: 0),
              
              const SizedBox(height: AppTheme.spacingXl),
              
              // Action buttons
              CustomButton(
                text: 'Scan Another',
                icon: Icons.refresh,
                onPressed: () {
                  Navigator.of(context).pushAndRemoveUntil(
                    MaterialPageRoute(builder: (_) => const HomeScreen()),
                    (route) => false,
                  );
                },
              ).animate().fadeIn(delay: 700.ms),
            ],
          ),
        ),
      ),
    );
  }
  
  String _getExplanation() {
    if (result.isReal) {
      return 'This ${result.mediaType} appears to be authentic. Our AI model detected no signs of manipulation.';
    } else if (result.isFake) {
      if (result.isVideo) {
        return 'This video shows signs of deepfake manipulation. ${result.fakeRatioPercentage} of analyzed frames were flagged as potentially fake.';
      } else {
        return 'This image shows signs of deepfake manipulation. Our AI model detected patterns consistent with synthetic media.';
      }
    } else {
      return 'The analysis is inconclusive. This could be due to low quality, unusual lighting, or other factors. Consider trying a different ${result.mediaType}.';
    }
  }
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  
  const _DetailRow({
    required this.label,
    required this.value,
    required this.icon,
  });
  
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(
          icon,
          size: 20,
          color: AppTheme.textSecondary,
        ),
        const SizedBox(width: AppTheme.spacingMd),
        Expanded(
          child: Text(
            label,
            style: const TextStyle(
              fontSize: 14,
              color: AppTheme.textSecondary,
            ),
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: AppTheme.textPrimary,
          ),
        ),
      ],
    );
  }
}
