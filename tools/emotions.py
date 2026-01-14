from enum import Enum
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from reachy_mini.motion.recorded_move import RecordedMoves

EMOTIONS_MCP = FastMCP("Emotions")
EMOTION_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

# Subset of all available moves - simplify + keep shorter moves
EMOTION_NAMES = (
    "laughing1",
    "understanding1",
    "frustrated1",
    "thoughtful1",
    "welcoming2",
    "uncomfortable1",
    "understanding2",  # also a "yes", but short
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
    "yes_sad1",
)
EMOTION_DESC = "\n".join(f"- {name}: {EMOTION_MOVES.get(name).description}" for name in EMOTION_NAMES)


@EMOTIONS_MCP.tool()
async def show_emotion(
    move_name: Annotated[
        Enum("EmotionName", {name: name for name in EMOTION_NAMES}),
        Field(
            description=(
                "Which emotion to show. Pick one of the enum values.\n"
                "Descriptions:\n"
                f"{EMOTION_DESC}"
            )
        ),
    ],
) -> dict:
    """
    Select an emotion move to display on the robot.
    """
    #This tool only returns the selected move name; the host app should decide whether/when
    #to actually play the move on the robot.
    return {"move_name": move_name}

