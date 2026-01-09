import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import '../../config/theme.dart';
import '../../config/app_config.dart';
import '../../providers/detection_provider.dart';
import '../../providers/profile_provider.dart';
import '../../models/detection_history.dart';
import '../../widgets/detection/upload_progress.dart';
import '../../widgets/common/custom_button.dart';
import 'result_screen.dart';

class UploadScreen extends ConsumerStatefulWidget {
  final bool isVideo;
  
  const UploadScreen({
    super.key,
    required this.isVideo,
  });
  
  @override
  ConsumerState<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends ConsumerState<UploadScreen> {
  XFile? _selectedFile;
  final ImagePicker _imagePicker = ImagePicker();
  
  Future<void> _pickFile() async {
    try {
      if (widget.isVideo) {
        // Pick video using ImagePicker (better Windows support)
        final XFile? video = await _imagePicker.pickVideo(
          source: ImageSource.gallery,
        );
        
        if (video != null) {
          final fileSize = await video.length();
          
          if (fileSize > AppConfig.maxVideoSize) {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Video size exceeds 50 MB limit'),
                  backgroundColor: AppTheme.accentRed,
                ),
              );
            }
            return;
          }
          
          setState(() => _selectedFile = video);
        }
      } else {
        // Pick image
        final XFile? image = await _imagePicker.pickImage(
          source: ImageSource.gallery,
          maxWidth: 1920,
          maxHeight: 1920,
        );
        
        if (image != null) {
          final fileSize = await image.length();
          
          if (fileSize > AppConfig.maxImageSize) {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Image size exceeds 5 MB limit'),
                  backgroundColor: AppTheme.accentRed,
                ),
              );
            }
            return;
          }
          
          setState(() => _selectedFile = image);
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to pick file: $e'),
            backgroundColor: AppTheme.accentRed,
          ),
        );
      }
    }
  }
  
  Future<void> _uploadAndAnalyze() async {
    if (_selectedFile == null) return;
    
    try {
      await ref.read(detectionNotifierProvider.notifier).analyzeMedia(_selectedFile!);
      
      final state = ref.read(detectionNotifierProvider);
      
      if (state.result != null && mounted) {
        // Save to history
        final fileName = _selectedFile!.name;
        final fileSize = await _selectedFile!.length();
        
        final historyItem = DetectionHistory(
          id: DateTime.now().millisecondsSinceEpoch.toString(),
          userId: 'current_user', // Will be replaced with actual user ID
          result: state.result!,
          fileName: fileName,
          fileSize: fileSize,
          createdAt: DateTime.now(),
        );
        
        await ref.read(historyNotifierProvider.notifier).addToHistory(historyItem);
        
        // Navigate to result screen
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(
            builder: (_) => ResultScreen(
              result: state.result!,
              filePath: _selectedFile!.path,
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.toString().replaceFirst('Exception: ', '')),
            backgroundColor: AppTheme.accentRed,
          ),
        );
      }
    }
  }
  
  @override
  Widget build(BuildContext context) {
    final detectionState = ref.watch(detectionNotifierProvider);
    
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.isVideo ? 'Scan Video' : 'Scan Image'),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(AppTheme.spacingLg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Upload area
              Expanded(
                child: detectionState.isLoading
                    ? Center(
                        child: UploadProgress(
                          progress: detectionState.uploadProgress,
                          message: detectionState.uploadProgress < 1.0
                              ? 'Uploading...'
                              : 'Analyzing...',
                        ),
                      )
                    : _selectedFile == null
                        ? _buildUploadPrompt()
                        : _buildFilePreview(),
              ),
              
              const SizedBox(height: AppTheme.spacingLg),
              
              // Action buttons
              if (!detectionState.isLoading) ...[
                if (_selectedFile == null)
                  CustomButton(
                    text: widget.isVideo ? 'Select Video' : 'Select Image',
                    icon: widget.isVideo ? Icons.videocam : Icons.image,
                    onPressed: _pickFile,
                  )
                else ...[
                  CustomButton(
                    text: 'Analyze',
                    icon: Icons.shield,
                    onPressed: _uploadAndAnalyze,
                  ),
                  const SizedBox(height: AppTheme.spacingMd),
                  CustomButton(
                    text: 'Choose Different File',
                    onPressed: _pickFile,
                    isOutlined: true,
                  ),
                ],
              ],
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildUploadPrompt() {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.surfaceDark,
        borderRadius: BorderRadius.circular(AppTheme.radiusLg),
        border: Border.all(
          color: AppTheme.primaryTeal.withOpacity(0.3),
          width: 2,
          strokeAlign: BorderSide.strokeAlignInside,
        ),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(AppTheme.spacingXl),
            decoration: const BoxDecoration(
              gradient: AppTheme.primaryGradient,
              shape: BoxShape.circle,
            ),
            child: Icon(
              widget.isVideo ? Icons.videocam : Icons.image,
              size: 60,
              color: Colors.white,
            ),
          ).animate().scale(
            duration: AppTheme.animationNormal,
            curve: Curves.easeOutBack,
          ),
          const SizedBox(height: AppTheme.spacingXl),
          Text(
            widget.isVideo ? 'Select a Video' : 'Select an Image',
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: AppTheme.textPrimary,
            ),
          ).animate().fadeIn(delay: 100.ms),
          const SizedBox(height: AppTheme.spacingMd),
          Text(
            widget.isVideo
                ? 'Supported formats: MP4, AVI, MOV, WEBM\nMax size: 50 MB'
                : 'Supported formats: JPG, JPEG, PNG\nMax size: 5 MB',
            style: const TextStyle(
              fontSize: 14,
              color: AppTheme.textSecondary,
            ),
            textAlign: TextAlign.center,
          ).animate().fadeIn(delay: 200.ms),
        ],
      ),
    );
  }
  
  Widget _buildFilePreview() {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate responsive sizes based on available space
        final maxImageHeight = constraints.maxHeight * 0.6; // 60% of available height
        final maxImageWidth = constraints.maxWidth * 0.9; // 90% of available width
        final imageHeight = maxImageHeight.clamp(150.0, 400.0); // Min 150, Max 400
        
        return SingleChildScrollView(
          child: Container(
            constraints: BoxConstraints(minHeight: constraints.maxHeight),
            decoration: BoxDecoration(
              color: AppTheme.surfaceDark,
              borderRadius: BorderRadius.circular(AppTheme.radiusLg),
            ),
            child: Padding(
              padding: const EdgeInsets.all(AppTheme.spacingLg),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (!widget.isVideo)
                    ClipRRect(
                      borderRadius: BorderRadius.circular(AppTheme.radiusMd),
                      child: kIsWeb
                          ? FutureBuilder<Uint8List>(
                              future: _selectedFile!.readAsBytes(),
                              builder: (context, snapshot) {
                                if (snapshot.hasData) {
                                  return ConstrainedBox(
                                    constraints: BoxConstraints(
                                      maxHeight: imageHeight,
                                      maxWidth: maxImageWidth,
                                    ),
                                    child: Image.memory(
                                      snapshot.data!,
                                      fit: BoxFit.contain,
                                    ),
                                  );
                                }
                                return SizedBox(
                                  height: imageHeight,
                                  child: const Center(child: CircularProgressIndicator()),
                                );
                              },
                            )
                          : Image.network(
                              _selectedFile!.path,
                              height: imageHeight,
                              fit: BoxFit.contain,
                              errorBuilder: (context, error, stackTrace) {
                                // Fallback for mobile - try to load as file
                                return FutureBuilder<Uint8List>(
                                  future: _selectedFile!.readAsBytes(),
                                  builder: (context, snapshot) {
                                    if (snapshot.hasData) {
                                      return ConstrainedBox(
                                        constraints: BoxConstraints(
                                          maxHeight: imageHeight,
                                          maxWidth: maxImageWidth,
                                        ),
                                        child: Image.memory(
                                          snapshot.data!,
                                          fit: BoxFit.contain,
                                        ),
                                      );
                                    }
                                    return SizedBox(
                                      height: imageHeight,
                                      child: const Center(child: CircularProgressIndicator()),
                                    );
                                  },
                                );
                              },
                            ),
                    ).animate().fadeIn()
                  else
                    Container(
                      height: imageHeight.clamp(150.0, 250.0),
                      width: imageHeight.clamp(150.0, 250.0),
                      decoration: BoxDecoration(
                        gradient: AppTheme.primaryGradient,
                        borderRadius: BorderRadius.circular(AppTheme.radiusMd),
                      ),
                      child: Icon(
                        Icons.videocam,
                        size: (imageHeight * 0.4).clamp(60.0, 100.0),
                        color: Colors.white,
                      ),
                    ).animate().fadeIn(),
                  const SizedBox(height: AppTheme.spacingLg),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: AppTheme.spacingMd),
                    child: Text(
                      _selectedFile!.name,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                        color: AppTheme.textPrimary,
                      ),
                      textAlign: TextAlign.center,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(height: AppTheme.spacingSm),
                  FutureBuilder<int>(
                    future: _selectedFile!.length(),
                    builder: (context, snapshot) {
                      if (snapshot.hasData) {
                        final sizeMB = (snapshot.data! / (1024 * 1024)).toStringAsFixed(2);
                        return Text(
                          '$sizeMB MB',
                          style: const TextStyle(
                            fontSize: 14,
                            color: AppTheme.textSecondary,
                          ),
                        );
                      }
                      return const SizedBox.shrink();
                    },
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
