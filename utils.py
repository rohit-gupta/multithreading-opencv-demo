from datetime import datetime


class FPSTracker:
    """
    Track FPS
    """

    def __init__(self):
        self.start_time = None
        self.frame_count = 0

    def start(self):
        self.start_time = datetime.now()
        return self

    def tick(self):
        self.frame_count += 1

    def get_fps(self):
        if self.start_time is not None:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            return self.frame_count / elapsed_time
        else:
            return 0
