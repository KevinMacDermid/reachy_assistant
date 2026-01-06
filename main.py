from src.wakeword import listen_for_wakeword
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves
import logging


logger = logging.getLogger(__name__)
EMOTION_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
DANCE_MOVES = RecordedMoves("pollen-robotics/reachy-mini-dances-library")


def main():
    """
    Main conversation App

    ** Start the daemon before running this **

    """
    wake_move = EMOTION_MOVES.get("welcoming1")
    reachy = ReachyMini()
    # ToDo: should force this to use the robot's audio
    listen_for_wakeword()
    print("Wakeword Received!")
    reachy.play_move(wake_move)


if __name__ == '__main__':
    main()

