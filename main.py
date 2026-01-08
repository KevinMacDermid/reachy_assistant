from src.wakeword import listen_for_wakeword
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves
from dotenv import load_dotenv
import logging


logger = logging.getLogger(__name__)
EMOTION_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
DANCE_MOVES = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
_ = load_dotenv()


def main():
    """
    Main conversation App
    """
    # Tried starting the daemon here, but it wouldn't work, start it separately
    reachy = ReachyMini()
    reachy.goto_sleep()
    # ToDo: should force this to use the robot's audio channel
    #listen_for_wakeword()
    #print("Wakeword Received!")
    reachy.wake_up()
    audio_frame = reachy.media.get_audio_sample()


if __name__ == '__main__':
    main()

