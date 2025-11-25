#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math
import numpy as np

def quaternion_from_euler(ai, aj, ak):
    # simple ZYX rotation
    ci = math.cos(ai/2)
    si = math.sin(ai/2)
    cj = math.cos(aj/2)
    sj = math.sin(aj/2)
    ck = math.cos(ak/2)
    sk = math.sin(ak/2)

    w = ci*cj*ck + si*sj*sk
    x = si*cj*ck - ci*sj*sk
    y = ci*sj*ck + si*cj*sk
    z = ci*cj*sk - si*sj*ck
    return (x, y, z, w)

class FakeIMU(Node):
    def __init__(self):
        super().__init__('fake_imu_node')
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.timer = self.create_timer(0.05, self.publish_imu)  # 20Hz

    def publish_imu(self):
        imu = Imu()
        now = self.get_clock().now().to_msg()
        imu.header.stamp = now
        imu.header.frame_id = "base_link"

        # orientation quaternion (no rotation)
        x, y, z, w = quaternion_from_euler(0.0, 0.0, 0.0)
        imu.orientation.x = x
        imu.orientation.y = y
        imu.orientation.z = z
        imu.orientation.w = w

        # orientation covariance (small values or unknown)
        imu.orientation_covariance = [0.01, 0, 0,
                                      0, 0.01, 0,
                                      0, 0, 0.01]

        # angular velocity
        imu.angular_velocity.x = 0.0
        imu.angular_velocity.y = 0.0
        imu.angular_velocity.z = 0.0
        imu.angular_velocity_covariance = [0.01, 0, 0,
                                           0, 0.01, 0,
                                           0, 0, 0.01]

        # linear acceleration
        imu.linear_acceleration.x = 0.0
        imu.linear_acceleration.y = 0.0
        imu.linear_acceleration.z = 0.0
        imu.linear_acceleration_covariance = [0.01, 0, 0,
                                              0, 0.01, 0,
                                              0, 0, 0.01]

        self.imu_pub.publish(imu)

def main(args=None):
    rclpy.init(args=args)
    node = FakeIMU()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
