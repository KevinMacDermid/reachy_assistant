from multiprocessing.resource_sharer import DupFd

from mcp.server.fastmcp import FastMCP
import subprocess
from dataclasses import dataclass
from pydantic import BaseModel

mcp = FastMCP("Tools Server")
DEFAULT_IP = "192.168.1.201"

class LightState(BaseModel):
    """ Tracks the state of an LED. Defaults are the default on state"""
    on_or_off: str = "on"  # this is a string to match the CLI
    cct: bool = True  # this is the nice warm white
    brightness: int = 100
    color: tuple[int, int, int] = (0, 0, 0)

@mcp.tool()
def get_light_state() -> LightState:
    """
    Get the state of the (LED) lights
    """
    # Found the cmd line operation worked better than the library calls...
    cmd = f'flux_led --info {DEFAULT_IP}'
    result = str(subprocess.check_output(cmd, shell=True))
    # Extract required data
    on_or_off = "on" if " ON " in result else "off"
    cct = True if "[CCT:" in result else False
    brightness = 100
    if "Brightness:" in result:
        brightness = int(result.split("Brightness: ")[1][:3].replace("%", "").strip())
    color = (0, 0, 0)
    if "[Color:" in result:
        color = tuple([int(x) for x in result.split("[Color: (")[1].split(")")[0].split(",")])
    return LightState(on_or_off, cct, brightness, color)


@mcp.tool()
def set_light_state(light_state: LightState) -> str:
    """
    Set the state of the (LED) lights

    Args:
        light_state: The desired state of the LED lights

    Returns:
        A message indicating the result of the operation
    """
    # Build the command based on the light state
    cmd_parts = [f'flux_led {DEFAULT_IP}']

    if light_state.on_or_off == "off":
        cmd_parts.append('--off')
    else:
        cmd_parts.append('--on')

        if light_state.cct:
            # Use CCT mode with brightness
            cmd_parts.append(f'-w {light_state.brightness}')
        else:
            # Use RGB color mode
            r, g, b = light_state.color
            cmd_parts.append(f'-c {r},{g},{b}')

    cmd = ' '.join(cmd_parts)

    try:
        subprocess.check_output(cmd, shell=True)

        if light_state.on_or_off == "off":
            return "Lights turned off successfully"
        elif light_state.cct:
            return f"Lights set to warm white at {light_state.brightness}% brightness"
        else:
            return f"Lights set to color RGB {light_state.color}"
    except subprocess.CalledProcessError as e:
        return f"Error setting light state: {str(e)}"


if __name__ == "__main__":
    mcp.run()
