import cv2

import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge


class PublisherNodeClass(Node):

    def __init__(self):
        super().__init__('publisher_node')
        
        # Auto-detect working camera
        self.cameraDeviceNumber = self.detect_camera()
        if self.cameraDeviceNumber is None:
            self.get_logger().error('No working camera found!')
            raise RuntimeError('No camera available')
        
        self.get_logger().info(f'Using camera index: {self.cameraDeviceNumber}')
        self.camera = cv2.VideoCapture(self.cameraDeviceNumber)
        self.bridgeObject = CvBridge()
        self.topicNameFrames='topic_camera_image'
        self.queueSize=20
        self.publisher = self.create_publisher(Image,self.topicNameFrames, self.queueSize)
        self.periodCommunication=0.02
        self.timer = self.create_timer(self.periodCommunication,self.timer_callbackFunction)
        self.i=0
    
    def detect_camera(self):
        """Try multiple camera indices to find a working one"""
        camera_indices = [6, 0, 2, 4]  # Try common indices
        
        for index in camera_indices:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return index
        return None
        
    def timer_callbackFunction(self):
        success,frame = self.camera.read()
        
        if not success or frame is None:
            self.get_logger().warn('Failed to read frame from camera')
            return
            
        frame = cv2.resize(frame,(820,640), interpolation=cv2.INTER_CUBIC)
        
        if success == True:
            ROS2ImageMessage= self.bridgeObject.cv2_to_imgmsg(frame)
            self.publisher.publish(ROS2ImageMessage)
        self.get_logger().info('Publishing image number %d' % self.i)
        self.i += 1
        
def main(args=None):
    rclpy.init(args=args)

    publisherObject = PublisherNodeClass()
    rclpy.spin(publisherObject)
    publisherObject.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()