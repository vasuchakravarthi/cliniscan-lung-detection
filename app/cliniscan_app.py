import { Card } from './ui/card';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { CheckCircle, AlertTriangle, Target, BarChart3, TrendingUp } from 'lucide-react';
import { motion } from 'motion/react';

interface DetectionResultsProps {
  detection: {
    detections: Array<{
      class: string;
      confidence: number;
    }>;
    totalDetections: number;
  };
  originalImage: string | null;
}

export function DetectionResults({ detection, originalImage }: DetectionResultsProps) {
  const hasDetections = detection.totalDetections > 0;
  const avgConfidence = hasDetections
    ? detection.detections.reduce((sum, d) => sum + d.confidence, 0) / detection.totalDetections
    : 0;

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, x: 30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="bg-gradient-to-br from-white to-cyan-50 p-6 shadow-2xl border-2 border-cyan-200 hover:shadow-3xl transition-all duration-300">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg">
              <Target className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-gray-800">Object Detection</h2>
          </div>
          
          {originalImage && (
            <motion.div 
              className="mb-6 relative rounded-2xl overflow-hidden bg-gradient-to-br from-gray-900 to-gray-800 shadow-2xl"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <img 
                src={originalImage} 
                alt="Detection visualization" 
                className="w-full h-auto"
              />
              {hasDetections && (
                <>
                  {/* Mock bounding boxes overlay with animations */}
                  <motion.div 
                    className="absolute top-[20%] left-[30%] w-[25%] h-[30%] border-4 border-red-500 rounded-lg shadow-2xl"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    <div className="absolute -top-8 left-0 bg-gradient-to-r from-red-500 to-rose-600 text-white px-3 py-1.5 text-xs rounded-lg shadow-lg">
                      {detection.detections[0]?.class} {(detection.detections[0]?.confidence * 100).toFixed(0)}%
                    </div>
                  </motion.div>
                  {detection.detections.length > 1 && (
                    <motion.div 
                      className="absolute top-[35%] right-[25%] w-[20%] h-[25%] border-4 border-yellow-500 rounded-lg shadow-2xl"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.5 }}
                    >
                      <div className="absolute -top-8 left-0 bg-gradient-to-r from-yellow-500 to-orange-500 text-white px-3 py-1.5 text-xs rounded-lg shadow-lg whitespace-nowrap">
                        {detection.detections[1]?.class} {(detection.detections[1]?.confidence * 100).toFixed(0)}%
                      </div>
                    </motion.div>
                  )}
                  {detection.detections.length > 2 && (
                    <motion.div 
                      className="absolute bottom-[25%] left-[20%] w-[22%] h-[20%] border-4 border-blue-500 rounded-lg shadow-2xl"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.7 }}
                    >
                      <div className="absolute -top-8 left-0 bg-gradient-to-r from-blue-500 to-cyan-600 text-white px-3 py-1.5 text-xs rounded-lg shadow-lg whitespace-nowrap">
                        {detection.detections[2]?.class} {(detection.detections[2]?.confidence * 100).toFixed(0)}%
                      </div>
                    </motion.div>
                  )}
                </>
              )}
            </motion.div>
          )}

          <p className="text-gray-600 text-xs mb-4 text-center bg-cyan-50 p-2 rounded-lg border border-cyan-200">
            📦 YOLOv8-M Detection Model (mAP: 0.4305)
          </p>

          {hasDetections ? (
            <div className="space-y-4">
              <div className="flex items-start gap-2 mb-4 bg-gradient-to-r from-yellow-50 to-orange-50 p-4 rounded-xl border-2 border-yellow-300">
                <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                <h3 className="text-gray-800">🎯 Detected Abnormalities</h3>
              </div>
              
              <div className="space-y-3">
                {detection.detections.slice(0, 5).map((det, index) => (
                  <motion.div 
                    key={index} 
                    className="bg-gradient-to-r from-gray-50 to-blue-50 p-4 rounded-xl border-2 border-blue-200 hover:border-blue-400 transition-all duration-300 shadow-md hover:shadow-lg"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + index * 0.1 }}
                    whileHover={{ scale: 1.02 }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-800 flex items-center gap-2">
                        <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center text-white shadow-md">
                          {index + 1}
                        </div>
                        <strong>{det.class}</strong>
                      </span>
                      <span className="text-gray-700 bg-white px-3 py-1 rounded-lg shadow-sm border border-gray-200">
                        {(det.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    <Progress value={det.confidence * 100} className="h-3 bg-gray-200" />
                  </motion.div>
                ))}
              </div>

              <motion.div 
                className="mt-6 pt-4 border-t-2 border-gray-200 space-y-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
              >
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gradient-to-br from-indigo-50 to-purple-50 p-4 rounded-xl border-2 border-indigo-200 shadow-md">
                    <div className="flex items-center gap-2 mb-1">
                      <BarChart3 className="w-4 h-4 text-indigo-600" />
                      <span className="text-gray-600 text-sm">Total Detections</span>
                    </div>
                    <span className="text-2xl text-indigo-700">{detection.totalDetections}</span>
                  </div>
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-4 rounded-xl border-2 border-blue-200 shadow-md">
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingUp className="w-4 h-4 text-blue-600" />
                      <span className="text-gray-600 text-sm">Avg Confidence</span>
                    </div>
                    <span className="text-2xl text-blue-700">{(avgConfidence * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </motion.div>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
            >
              <Alert className="bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-300 shadow-lg">
                <CheckCircle className="h-6 w-6 text-green-600" />
                <AlertDescription>
                  <p className="text-green-900 mb-2 flex items-center gap-2">
                    <strong className="text-lg">✅ No abnormalities detected</strong>
                  </p>
                  <p className="text-green-800 text-sm">
                    This X-ray appears normal based on the detection model. All systems clear!
                  </p>
                </AlertDescription>
              </Alert>
            </motion.div>
          )}
        </Card>
      </motion.div>
    </div>
  );
}
