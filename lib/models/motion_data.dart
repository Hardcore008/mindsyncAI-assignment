class MotionData {
  final double x;
  final double y;
  final double z;
  final int timestamp;

  MotionData({
    required this.x,
    required this.y,
    required this.z,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() {
    return {
      'x': x,
      'y': y,
      'z': z,
      'timestamp': timestamp,
    };
  }

  factory MotionData.fromJson(Map<String, dynamic> json) {
    return MotionData(
      x: json['x'].toDouble(),
      y: json['y'].toDouble(),
      z: json['z'].toDouble(),
      timestamp: json['timestamp'],
    );
  }
}
