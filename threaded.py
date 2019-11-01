import numpy as np
from threading import Thread
import cv2
from skimage import segmentation, color, filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import copy

from utils import FPSTracker


def image_superpixel_segmentation(img):
    """
    Perform SLIC superpixel segmentation on image
    :param img:
    :return:
    """
    slic_labels = segmentation.slic(img, compactness=10, n_segments=128)
    seg_image = color.label2rgb(slic_labels, img, kind='avg')

    return seg_image


@adapt_rgb(each_channel)
def image_edge_detection(img):
    """
    Perform edge detection on image and return gradient magnitude image
    :param img:
    :return:
    """
    return filters.sobel(img).astype(np.float32)


class WebcamCapture:
    """
    Thread to capture frames from webcam
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        (self.ret, self.frame) = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.grab_frame, args=()).start()

    def grab_frame(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                (self.ret, self.frame) = self.cap.read()

    def stop(self):
        self.stopped = True
        # When everything done, release the capture
        self.cap.release()


class ProcessFrames:
    """
    Thread to process frames
    """

    def __init__(self, frame=None):
        self.input_frame = frame
        self.output_frame = frame
        self.stopped = False
        self.fps_tracker = FPSTracker()


    def start(self):
        self.fps_tracker.start()
        Thread(target=self.process, args=()).start()

    def process(self):
        while not self.stopped:
            # Our operations on the frame come here
            rgb = cv2.cvtColor(self.input_frame, cv2.COLOR_BGR2RGB)
            slic_seg_image = image_superpixel_segmentation(rgb)
            # slic_seg_image = image_edge_detection(rgb)

            # cv2.imshow() expects BGR
            self.output_frame = cv2.cvtColor(slic_seg_image, cv2.COLOR_RGB2BGR)

            # Add FPS Text to frame
            cv2.putText(self.output_frame, "{:.2f} FPS".format(self.fps_tracker.get_fps()),
                        (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))

            self.fps_tracker.tick()

    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    webcam_stream = WebcamCapture()
    webcam_stream.start()

    process_stream = ProcessFrames(webcam_stream.frame)
    process_stream.start()

    # display_stream = DisplayFrames(webcam_stream.frame)
    # display_stream.start()

    fps_tracker = FPSTracker()
    fps_tracker.start()

    while True:

        # if display_stream.stopped or webcam_stream.stopped or process_stream.stopped:
        if webcam_stream.stopped or process_stream.stopped or cv2.waitKey(1) & 0xFF == ord('q'):
            webcam_stream.stop()
            process_stream.stop()
            break

        # Capture frame-by-frame
        input_frame = webcam_stream.frame
        input_copy = copy.deepcopy(input_frame).astype(np.float32)
        input_copy /= 255.0

        # Process the frame
        process_stream.input_frame = input_frame

        # Display the resulting frame
        # display_stream.frame = process_stream.output_frame

        cv2.putText(input_copy, "{:.2f} FPS".format(fps_tracker.get_fps()),
                    (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))


        output_frame = process_stream.output_frame / 255.0

        # print(np.median(input_copy), np.median(output_frame))

        joined_disp = np.hstack((input_copy, output_frame))
        cv2.imshow('frame', joined_disp)
        fps_tracker.tick()

    cv2.destroyAllWindows()
