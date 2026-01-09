class DetectionResult {
  final String result; // FAKE | REAL | UNCERTAIN
  final double confidence;
  final double fakeRatio;
  final int framesUsed;
  final String mediaType; // image | video
  final DateTime timestamp;
  final String? filePath;
  
  DetectionResult({
    required this.result,
    required this.confidence,
    required this.fakeRatio,
    required this.framesUsed,
    required this.mediaType,
    DateTime? timestamp,
    this.filePath,
  }) : timestamp = timestamp ?? DateTime.now();
  
  // Factory constructor from JSON
  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      result: json['result'] as String? ?? 'UNCERTAIN',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      fakeRatio: (json['fake_ratio'] as num?)?.toDouble() ?? 0.0,
      framesUsed: json['frames_used'] as int? ?? 0,
      mediaType: json['media_type'] as String? ?? 'image',
      timestamp: json['timestamp'] != null 
          ? DateTime.parse(json['timestamp'] as String)
          : DateTime.now(),
      filePath: json['file_path'] as String?,
    );
  }
  
  // Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'result': result,
      'confidence': confidence,
      'fake_ratio': fakeRatio,
      'frames_used': framesUsed,
      'media_type': mediaType,
      'timestamp': timestamp.toIso8601String(),
      'file_path': filePath,
    };
  }
  
  // Helper getters
  bool get isFake => result.toUpperCase() == 'FAKE';
  bool get isReal => result.toUpperCase() == 'REAL';
  bool get isUncertain => result.toUpperCase() == 'UNCERTAIN';
  bool get isVideo => mediaType.toLowerCase() == 'video';
  bool get isImage => mediaType.toLowerCase() == 'image';
  
  // Get confidence percentage
  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(1)}%';
  
  // Get fake ratio percentage
  String get fakeRatioPercentage => '${(fakeRatio * 100).toStringAsFixed(1)}%';
  
  // Get result description
  String get description {
    if (isVideo) {
      return '$fakeRatioPercentage of frames flagged as fake';
    } else {
      return 'Confidence: $confidencePercentage';
    }
  }
  
  @override
  String toString() {
    return 'DetectionResult(result: $result, confidence: $confidence, mediaType: $mediaType)';
  }
}
