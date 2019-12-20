"""
tellotracker:
Allows manual operation of the drone and demo tracking mode.

Requires mplayer to record/save video.

Controls:
- tab to lift off
- WASD to move the drone
- space/shift to ascend/descent slowly
- Q/E to yaw slowly
- arrow keys to ascend, descend, or yaw quickly
- backspace to land, or P to palm-land
- enter to take a picture
- R to start recording video, R again to stop recording
  (video and photos will be saved to a timestamped file in ~/Pictures/)
- Z to toggle camera zoom state
  (zoomed-in widescreen or high FOV 4:3)
- T to toggle tracking
@author Leonie Buckley, Saksham Sinha and Jonathan Byrne
@copyright 2018 see license file for details
"""
import time

import datetime
import os
import tellopy
import numpy
import av
import cv2
from pynput import keyboard
from tracker import Tracker

from ObjectDetector_Mobilenet_SSD import ObjectDetector_Mobilenet_SSD

from simple_pid import PID#https://github.com/m-lundberg/simple-pid

image = None

def getFrame():
    global image
    return image

PHOTO_SNAP_INTERVAL_SECS = 3

def main():
    photo_snap_time = time.time()#Yeap.
    pid_track_time = 0
    global image
    """ Create a tello controller and show the video feed."""
    tellotrack = TelloCV()
    objDetector = ObjectDetector_Mobilenet_SSD('GPU', getFrame)

    human_detected = False
    human_not_detected_count = 0
    human_x, human_y = 0, 0
    searching_human = False#Searching -> found: 'assure' abrupt stop.
    #1/objDetector.inferFPS

    yaw_pid = PID(1, 0.12, 0.05, setpoint = 0, sample_time = None, output_limits = (-1, 1))#We go for 0 deg error dead ooon.
    throttle_pid = PID(1, 0.1, 0.01, setpoint = 0, sample_time = None, output_limits = (-1, 1))#We go for 0 deg error dead ooon.
    print(yaw_pid.components, throttle_pid.components)

    for packet in tellotrack.container.demux((tellotrack.vid_stream,)):
        for frame in packet.decode():
            image = tellotrack.process_frame(frame)
            # Our operations on the frame come here
            #renderImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            renderImg = image
            originH, originW, _ = renderImg.shape
            infer_time, detectedObjects = objDetector.getProcessedData()

            # << Human Tracking - PID stuff Starts Here >> #
            max_possb = 0
            min_dist = 99999999
            
            human_detected = False
            if detectedObjects is not None:
                for object in detectedObjects:
                    obj3, obj4, obj5, obj6, class_id, percent = object
                    xmin = int(obj3 * originW)
                    ymin = int(obj4 * originH)
                    xmax = int(obj5 * originW)
                    ymax = int(obj6 * originH)
                    if(objDetector.labels[class_id] == 'person'):
                        #if(percent > max_possb):
                        #    max_possb = percent
                        
                        #MIN=distance. Priority.
                        _human_x, _human_y = ((xmin + xmax)//2, int(ymin + (ymax - ymin)*0.2) )#Yeesssss. Aim for the head.
                        _dist = (_human_x - human_x)**2 + (_human_y - human_y)**2
                        if(_dist < min_dist):
                            min_dist = _dist#update
                            human_x, human_y = _human_x, _human_y#yeap.

                        human_detected = True
                        human_not_detected_count = 0#Init.
                if human_detected:
                    # object 표시
                    cv2.rectangle(renderImg, (xmin,ymin),
                                  (xmax, ymax), (0,0,0), 2)
                    cv2.circle(renderImg, (human_x, human_y), 10, (0,0,255), 2)
                    cv2.putText(renderImg, objDetector.labels[class_id] + str(percent), (xmin, ymin + 40), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (200, 10, 10), 1)
                        #When some value is returned from the neural net.
                if tellotrack.human_following:
                    if(human_detected):
                        human_x_norm = human_x / originW
                        human_y_norm = human_y / originH
                        human_x_err = (0.5 - human_x_norm)#Huh, 
                        human_y_err = (human_y_norm - 0.5)#Need to be DEAD on. Inverse dir. Considering Throttle result.
                        print('yaw e:', human_x_err, 'thtle e:', human_y_err)

                        yaw_output = yaw_pid(human_x_err)
                        throttle_output = throttle_pid(human_y_err)
                        print('yaw_out:', yaw_output, 'throttle_out:',throttle_output)

                        cv2.rectangle(renderImg, (originW//2, originH//2),
                             (originW//2 + int(yaw_output * 200), originH//2 - int(throttle_output * 200)), (0,0,0), 2)

                        tellotrack.drone.set_yaw(yaw_output)
                        #tellotrack.drone.set_throttle(throttle_output)
                        tellotrack.drone.set_roll(0.1)#SLIGHT.

                        #SNAP Photo.
                        cur_time = time.time()

                        print('PID loop at:', cur_time - pid_track_time)
                        pid_track_time = cur_time

                        
                        '''if(cur_time - photo_snap_time > PHOTO_SNAP_INTERVAL_SECS):
                            tellotrack.drone.take_picture()
                            photo_snap_time = cur_time'''

                    if not human_detected:
                        human_not_detected_count += 1
                    if(human_not_detected_count > 22):#Approx. 1.5sec.
                        #print('Human not detected too long. CCW rot 30 doin...')
                        tellotrack.drone.counter_clockwise(30)#Search...
                        searching_human = True
                    cv2.putText(renderImg, str(infer_time), (0, originH - 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (10, 200, 10), 1)
            cv2.circle(renderImg, (originW//2, originH//2), 10, (255,0,0),2)
            # << Human Tracking - PID stuff Ends Here >> #

            cv2.imshow('tello', renderImg)
            _ = cv2.waitKey(1) & 0xFF

class TelloCV(object):
    """
    TelloTracker builds keyboard controls on top of TelloPy as well
    as generating images from the video stream and enabling opencv support
    """

    def __init__(self):
        self.human_following = False

        self.prev_flight_data = None
        self.record = False
        self.tracking = False
        self.keydown = False
        self.date_fmt = '%Y-%m-%d_%H%M%S'
        self.speed = 50
        self.drone = tellopy.Tello()
        self.init_drone()
        self.init_controls()

        # container for processing the packets into frames
        self.container = av.open(self.drone.get_video_stream())
        self.vid_stream = self.container.streams.video[0]
        self.out_file = None
        self.out_stream = None
        self.out_name = None
        self.start_time = time.time()

        # tracking a color
        green_lower = (30, 50, 50)
        green_upper = (80, 255, 255)
        #red_lower = (0, 50, 50)
        # red_upper = (20, 255, 255)
        # blue_lower = (110, 50, 50)
        # upper_blue = (130, 255, 255)
        self.track_cmd = ""
        self.tracker = Tracker(self.vid_stream.height,
                               self.vid_stream.width,
                               green_lower, green_upper)

    def init_drone(self):
        """Connect, uneable streaming and subscribe to events"""
        # self.drone.log.set_level(2)
        self.drone.connect()
        self.drone.start_video()
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
                             self.flight_data_handler)
        self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED,
                             self.handle_flight_received)


    def on_press(self, keyname):
        """handler for keyboard listener"""
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.drone.quit()
                exit(0)
            if keyname in self.controls:
                key_handler = self.controls[keyname]
                if isinstance(key_handler, str):
                    getattr(self.drone, key_handler)(self.speed)
                else:
                    key_handler(self.speed)
        except AttributeError:
            print('special key {0} pressed'.format(keyname))

    def on_release(self, keyname):
        """Reset on key up from keyboard listener"""
        self.keydown = False
        keyname = str(keyname).strip('\'')
        print('-' + keyname)
        if keyname in self.controls:
            key_handler = self.controls[keyname]
            if isinstance(key_handler, str):
                getattr(self.drone, key_handler)(0)
            else:
                key_handler(0)

    def init_controls(self):
        """Define keys and add listener"""
        self.controls = {
            'f' : lambda speed: self.toggle_following(speed),#Junwoo HWANG Added. 191127_0353.
            'w': 'forward',
            's': 'backward',
            'a': 'left',
            'd': 'right',
            'Key.space': 'up',
            'Key.shift': 'down',
            'Key.shift_r': 'down',
            'q': 'counter_clockwise',
            'e': 'clockwise',
            'i': lambda speed: self.drone.flip_forward(),
            'k': lambda speed: self.drone.flip_back(),
            'j': lambda speed: self.drone.flip_left(),
            'l': lambda speed: self.drone.flip_right(),
            # arrow keys for fast turns and altitude adjustments
            'Key.left': lambda speed: self.drone.counter_clockwise(speed),
            'Key.right': lambda speed: self.drone.clockwise(speed),
            'Key.up': lambda speed: self.drone.up(speed),
            'Key.down': lambda speed: self.drone.down(speed),
            'Key.tab': lambda speed: self.drone.takeoff(),
            'Key.backspace': lambda speed: self.drone.land(),
            'p': lambda speed: self.palm_land(speed),
            't': lambda speed: self.toggle_tracking(speed),
            'r': lambda speed: self.toggle_recording(speed),
            'z': lambda speed: self.toggle_zoom(speed),
            'Key.enter': lambda speed: self.take_picture(speed)
        }
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()
        # self.key_listener.join()

    def process_frame(self, frame):
        """convert frame to cv2 image and show"""
        image = cv2.cvtColor(numpy.array(
            frame.to_image()), cv2.COLOR_RGB2BGR)
        image = self.write_hud(image)
        if self.record:
            self.record_vid(frame)

        distance = 100
        cmd = ""
        if self.tracking:
            xoff, yoff = self.tracker.track(image)
            image = self.tracker.draw_arrows(image)

            if xoff < -distance:
                cmd = "counter_clockwise"
            elif xoff > distance:
                cmd = "clockwise"
            elif yoff < -distance:
                cmd = "down"
            elif yoff > distance:
                cmd = "up"
            else:
                if self.track_cmd is not "":
                    getattr(self.drone, self.track_cmd)(0)
                    self.track_cmd = ""


        if cmd is not self.track_cmd:
            if cmd is not "":
                print("track command:", cmd)
                getattr(self.drone, cmd)(self.speed)
                self.track_cmd = cmd

        return image

    def write_hud(self, frame):
        """Draw drone info, tracking and record on frame"""
        stats = self.prev_flight_data.split('|')
        stats.append("Tracking:" + str(self.tracking))
        if self.drone.zoom:
            stats.append("VID")
        else:
            stats.append("PIC")
        if self.record:
            diff = int(time.time() - self.start_time)
            mins, secs = divmod(diff, 60)
            stats.append("REC {:02d}:{:02d}".format(mins, secs))

        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(frame, text, (0, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), lineType=30)
        return frame
    def toggle_following(self, speed):
        if speed == 0:
            return
        self.human_following = not self.human_following
        print('humman following:', self.human_following)
        if not self.human_following:
            self.drone.set_yaw(0)
            self.drone.set_throttle(0)
            self.drone.set_roll(0)
            #Init controls back.
            print('following Nop. Controls return to 0.')
    def toggle_recording(self, speed):
        """Handle recording keypress, creates output stream and file"""
        if speed == 0:
            return
        self.record = not self.record

        if self.record:
            datename = [os.getenv('HOME'), datetime.datetime.now().strftime(self.date_fmt)]
            self.out_name = '{}/Pictures/tello-{}.mp4'.format(*datename)
            print("Outputting video to:", self.out_name)
            self.out_file = av.open(self.out_name, 'w')
            self.start_time = time.time()
            self.out_stream = self.out_file.add_stream(
                'mpeg4', self.vid_stream.rate)
            self.out_stream.pix_fmt = 'yuv420p'
            self.out_stream.width = self.vid_stream.width
            self.out_stream.height = self.vid_stream.height

        if not self.record:
            print("Video saved to ", self.out_name)
            self.out_file.close()
            self.out_stream = None

    def record_vid(self, frame):
        """
        convert frames to packets and write to file
        """
        new_frame = av.VideoFrame(
            width=frame.width, height=frame.height, format=frame.format.name)
        for i in range(len(frame.planes)):
            new_frame.planes[i].update(frame.planes[i])
        pkt = None
        try:
            pkt = self.out_stream.encode(new_frame)
        except IOError as err:
            print("encoding failed: {0}".format(err))
        if pkt is not None:
            try:
                self.out_file.mux(pkt)
            except IOError:
                print('mux failed: ' + str(pkt))

    def take_picture(self, speed):
        """Tell drone to take picture, image sent to file handler"""
        if speed == 0:
            return
        self.drone.take_picture()

    def palm_land(self, speed):
        """Tell drone to land"""
        if speed == 0:
            return
        self.drone.palm_land()

    def toggle_tracking(self, speed):
        """ Handle tracking keypress"""
        if speed == 0:  # handle key up event
            return
        self.tracking = not self.tracking
        print("tracking:", self.tracking)
        return

    def toggle_zoom(self, speed):
        """
        In "video" mode the self.drone sends 1280x720 frames.
        In "photo" mode it sends 2592x1936 (952x720) frames.
        The video will always be centered in the window.
        In photo mode, if we keep the window at 1280x720 that gives us ~160px on
        each side for status information, which is ample.
        Video mode is harder because then we need to abandon the 16:9 display size
        if we want to put the HUD next to the video.
        """
        if speed == 0:
            return
        self.drone.set_video_mode(not self.drone.zoom)

    def flight_data_handler(self, event, sender, data):
        """Listener to flight data from the drone."""
        text = str(data)
        if self.prev_flight_data != text:
            self.prev_flight_data = text
            #print(text)

    def handle_flight_received(self, event, sender, data):
        """Create a file in ~/Pictures/ to receive image from the drone"""
        path = '%s/Pictures/tello-%s.jpeg' % (
            os.getenv('HOME'),
            datetime.datetime.now().strftime(self.date_fmt))
        with open(path, 'wb') as out_file:
            out_file.write(data)
        print('Saved photo to %s' % path)


if __name__ == '__main__':
    main()
