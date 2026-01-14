from mcp.server.fastmcp import FastMCP
from reachy_mini.motion.recorded_move import RecordedMoves

EMOTIONS_MCP = FastMCP("Emotions")
EMOTION_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

# This is a subset of all available moves, to simplify + remove really long ones
EMOTION_NAMES = [
    "laughing1",
    "understanding1",
    "frustrated1",
    "thoughtful1",
    "welcoming2",
    "uncomfortable1",
    "understanding2",  # this is also a yes, but short
    "calming1",
    "inquiring2",
    "thoughtful2",
    "tired1",
    "proud3",
    "oops1",
    "lost1",
    "shy1",
    "success1",
    "reprimand1",
    "sad1",
    "no1",
    "enthusiastic1",
    "boredom1",
    "incomprehensible2",
    "yes_sad1"
]

@EMOTIONS_MCP.tool()
def show_emotion(move_name: str):
    pass