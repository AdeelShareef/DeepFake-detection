import 'detection_result.dart';

class DetectionHistory {
  final String id;
  final String userId;
  final DetectionResult result;
  final String fileName;
  final int fileSize;
  final DateTime createdAt;
  
  DetectionHistory({
    required this.id,
    required this.userId,
    required this.result,
    required this.fileName,
    required this.fileSize,
    required this.createdAt,
  });
  
  // Factory constructor from JSON
  factory DetectionHistory.fromJson(Map<String, dynamic> json) {
    return DetectionHistory(
      id: json['id'] as String,
      userId: json['user_id'] as String,
      result: DetectionResult.fromJson(json['result'] as Map<String, dynamic>),
      fileName: json['file_name'] as String,
      fileSize: json['file_size'] as int,
      createdAt: DateTime.parse(json['created_at'] as String),
    );
  }
  
  // Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'user_id': userId,
      'result': result.toJson(),
      'file_name': fileName,
      'file_size': fileSize,
      'created_at': createdAt.toIso8601String(),
    };
  }
  
  // Get file size in human-readable format
  String get fileSizeFormatted {
    if (fileSize < 1024) {
      return '$fileSize B';
    } else if (fileSize < 1024 * 1024) {
      return '${(fileSize / 1024).toStringAsFixed(1)} KB';
    } else {
      return '${(fileSize / (1024 * 1024)).toStringAsFixed(1)} MB';
    }
  }
  
  // Get relative time
  String get relativeTime {
    final now = DateTime.now();
    final difference = now.difference(createdAt);
    
    if (difference.inDays > 365) {
      return '${(difference.inDays / 365).floor()} year${(difference.inDays / 365).floor() > 1 ? 's' : ''} ago';
    } else if (difference.inDays > 30) {
      return '${(difference.inDays / 30).floor()} month${(difference.inDays / 30).floor() > 1 ? 's' : ''} ago';
    } else if (difference.inDays > 0) {
      return '${difference.inDays} day${difference.inDays > 1 ? 's' : ''} ago';
    } else if (difference.inHours > 0) {
      return '${difference.inHours} hour${difference.inHours > 1 ? 's' : ''} ago';
    } else if (difference.inMinutes > 0) {
      return '${difference.inMinutes} minute${difference.inMinutes > 1 ? 's' : ''} ago';
    } else {
      return 'Just now';
    }
  }
  
  @override
  String toString() {
    return 'DetectionHistory(id: $id, fileName: $fileName, result: ${result.result})';
  }
}
