from datetime import datetime


class FPSTracker:
    """
    Track FPS
    """

    def __init__(self):
        self._start_time = None
        self._frame_count = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def tick(self):
        self._frame_count += 1

    def get_fps(self):
        if self._start_time is not None:
            elapsed_time = (datetime.now() - self._start_time).total_seconds()
            return self._frame_count / elapsed_time
        else:
            return 0