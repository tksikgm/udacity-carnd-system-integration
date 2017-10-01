#!/usr/bin/env python
import rospy
from std_msgs.msg import Header, Int32
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point
from styx_msgs.msg import Lane, TrafficLightArray, TrafficLight
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

        self.pose = None
        self.car_position = Point()
        self.waypoints = None
        self.camera_image = None
        self.lights = []

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

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_line_positions = self.config['stop_line_positions']

        # Camera parameters
        # fx = self.config['camera_info']['focal_length_x']
        # fy = self.config['camera_info']['focal_length_y']
        # image_width = self.config['camera_info']['image_width']
        # image_height = self.config['camera_info']['image_height']
        self.camera_param = (fx, fy, cx, cy) = (2552.7, 2280.5, 366, 652.4)  # Manually tweaked

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.camera_image_pub = rospy.Publisher('/image_color_info', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
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
        self.has_image = True
        self.camera_image = msg

        if self.pose and self.waypoints and self.lights:
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


    def project_car_to_image(self, point):
        """Project point from 3D car coordinates to 2D camera image location
           http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        """
        fx, fy, cx, cy = self.camera_param
        p = Point(-point.y, -point.z, point.x)
        x_img = int(fx*p.x/p.z + cx)
        y_img = int(fy*p.y/p.z + cy)
        return (x_img, y_img), p


    def bounding_box(self, center, width, height):
        # Given center point, width and hight of object in car coordinate, return bounding box in image coordinate.
        top_left = Point(center.x, center.y+width/2, center.z+height/2)
        bottom_right = Point(center.x, center.y-width/2, center.z-height/2)
        (left, top),_ = self.project_car_to_image(top_left)
        (right, bottom),_ = self.project_car_to_image(bottom_right)
        return left, right, top, bottom


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """
        # Get transform between pose of camera and world frame
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform('/base_link', '/world', now, rospy.Duration(1.0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr('Failed to find camera to map transform')
            return (0, 0)

        # Transform from world coodinate to car coordinate
        point_wc = PointStamped(Header(0, now, '/world'), point_in_world)
        point_cc = self.listener.transformPoint('/base_link', point_wc).point

        # Transform from vehicle camera coodinate to image plane coordinate
        point_img, point_cam = self.project_car_to_image(point_cc)

        # Bounding box for traffic light
        light_width, light_height = 0.8, 2.2  # In meter. Manually tweaked
        bb = self.bounding_box(point_cc, light_width, light_height)

        return (point_img, bb, point_cam)


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        point_img, bb, point_cam = self.project_to_image_plane(light.pose.pose.position)

        # Put additional information on image and publish for testing. (but latency is large)
        text = '(%.2f, %.2f, %.2f)' % (point_cam.x, point_cam.y, point_cam.z)
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        cv2.putText(cv_image, 'light in camera coordinate', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(cv_image, text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.circle(cv_image, point_img, 5, (255,255,255), 2)
        cv2.rectangle(cv_image, (bb[0],bb[2]), (bb[1],bb[3]), (255,255,255), 2)
        image_message = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        try:
            self.camera_image_pub.publish(image_message)
        except CvBridgeError as e:
            rospy.loginfo(e)

        #TODO use light location to zoom in on traffic light in image
        #Get classification
        return self.light_classifier.get_classification(cv_image)


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
        for i, wp in enumerate(self.waypoints):
            x_wp = wp.pose.pose.position.x
            y_wp = wp.pose.pose.position.y
            dist = math.sqrt((x_wp-point.x)**2+(y_wp-point.y)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i
        return closest_index


    def get_closest_light(self):
        car_wp = self.get_closest_waypoint(self.car_position)
        closest_line = -1
        closest_dist = len(self.waypoints)
        for i, line_pos in enumerate(self.stop_line_positions):
            line_wp = self.get_closest_waypoint(Point(line_pos[0],line_pos[1],0))
            dist = line_wp - car_wp
            if dist < 0:
                dist += len(self.waypoints)
            if dist < closest_dist:
                closest_dist = dist
                closest_line = i
        return self.lights[closest_line]


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = self.get_closest_light()
        if light:
            light_wp = self.get_closest_waypoint(light.pose.pose.position)
            state = self.get_light_state(light)
            return light_wp, state
        else:
            return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
