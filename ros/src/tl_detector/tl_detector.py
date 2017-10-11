#!/usr/bin/env python
import rospy
from std_msgs.msg import Header, Int32
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point
from styx_msgs.msg import Lane, TrafficLightArray, TrafficLight, LightImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.car_position = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_image = None
        self.last_wp = -1
        self.state_count = 0

        # Reading values from config file.
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_line_positions = self.config['stop_line_positions']
        self.image_width = self.config['camera_info']['image_width']
        self.image_height = self.config['camera_info']['image_height']
        self.camera_param = (2552.7, 2280.5, 366, 652.4)  # Manually estimated simulator camera parameters
        # For site driving.
        # fx = self.config['camera_info']['focal_length_x']
        # fy = self.config['camera_info']['focal_length_y']
        # cx = self.image_width/2
        # cy = self.image_height/2
        # self.camera_param = (fx, fy, cx, cy)

        # Subscribers
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        # Publishers
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.light_image_pub = rospy.Publisher('/image_color_light', LightImage, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.car_position = msg.pose.position

    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.camera_image = msg
        if self.car_position and self.waypoints and self.lights:
            light_wp, state = self.process_traffic_lights()
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1


    # Given center point, width and hight of object in car coordinate, return bounding box in image coordinate.
    def get_bounding_box(self, center, width, height):
        top_left = Point(center.x, center.y+width/2, center.z+height/2)
        bottom_right = Point(center.x, center.y-width/2, center.z-height/2)
        (left, top) = self.project_car_to_image(top_left)
        (right, bottom) = self.project_car_to_image(bottom_right)
        return left, right, top, bottom


    def project_car_to_image(self, point):
        """Project point from 3D car coordinates to 2D camera image location
           http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        """
        fx, fy, cx, cy = self.camera_param
        p = Point(-point.y, -point.z, point.x)
        x_img = int(fx*p.x/p.z + cx)
        y_img = int(fy*p.y/p.z + cy)
        return x_img, y_img


    def get_zoomed_light_image(self, light_car_coord):
        light_image_coord = self.project_car_to_image(light_car_coord)
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, 'bgr8')
        left, right, top, bottom = self.get_bounding_box(light_car_coord, 3.2, 3.2)
        if left<0 or right>self.image_width or top<0 or bottom>self.image_height:
            rospy.loginfo('Bonding box out of image.')
            return None
        cv_image = cv_image[top:bottom, left:right]
        cv_image = cv2.resize(cv_image, (32,32))
        return cv_image


    def project_world_to_car(self, point_world_coord):
        time = self.camera_image.header.stamp
        try:
            self.listener.waitForTransform('/base_link', '/world', time, rospy.Duration(1.0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr('Failed to find camera to map transform')
            return None
        point_stamped_msg = PointStamped(Header(0, time, '/world'), point_world_coord)
        point_car_coord = self.listener.transformPoint('/base_link', point_stamped_msg).point
        return point_car_coord


    def get_closest_waypoint(self, point):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_index = None
        closest_dist = float('inf')
        for i, wp in enumerate([wp.pose.pose.position for wp in self.waypoints]):
            dist = math.sqrt((wp.x-point.x)**2+(wp.y-point.y)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i
        return closest_index


    def get_closest_light(self):
        car_wp = self.get_closest_waypoint(self.car_position)
        closest_line = -1
        closest_linw_wp = -1
        closest_dist = len(self.waypoints)
        for i, line_pos in enumerate(self.stop_line_positions):
            line_wp = self.get_closest_waypoint(Point(line_pos[0],line_pos[1],0))
            dist = line_wp - car_wp
            if dist < 0:
                dist += len(self.waypoints)
            if dist < closest_dist:
                closest_dist = dist
                closest_line = i
                closest_line_wp = line_wp
        return self.lights[closest_line], line_wp


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light, light_wp = self.get_closest_light()
        light_world_coord = light.pose.pose.position
        light_car_coord = self.project_world_to_car(light_world_coord)
        # rospy.loginfo(light_car_coord)
        # light_car_coord = Point(41, 0, 5.5)
        if light_car_coord == None:
            return -1, TrafficLight.UNKNOWN
        elif light_car_coord.x > 150:
            rospy.loginfo('Traffic light too far from car.')
            return -1, TrafficLight.UNKNOWN

        zoomed_image = self.get_zoomed_light_image(light_car_coord)
        try:
            if zoomed_image == None:
                return -1, TrafficLight.UNKNOWN
        except:
            pass

        # Show zoomed image
        cv2.imshow('Zoomed image', zoomed_image)
        cv2.waitKey(1)
        # Publish zoomed image (for training classifier)
        if self.last_state == light.state:
            msg = LightImage(self.bridge.cv2_to_imgmsg(self.last_image, 'bgr8'), self.last_state)
            self.light_image_pub.publish(msg)
        self.last_image = zoomed_image

        # state = self.light_classifier.get_classification(zoomed_image)
        state = light.state
        return light_wp, state


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
