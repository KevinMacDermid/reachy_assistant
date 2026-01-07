from multiprocessing.resource_sharer import DupFd

from mcp.server.fastmcp import FastMCP
import subprocess
from dataclasses import dataclass

mcp = FastMCP("Tools Server")
DEFAULT_IP = "192.168.1.201"

@dataclass
class LightState:
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
def set_light_state(
        on_or_off: str  = LiteralString["on", "off"],
        cct:bool
):
    def process_request(self, messages: list[dict]) -> str:
        # Got inconsistent behaviour unless I kept only the last message
        props = self._get_properties(get_message_content(messages[-1]))
        if type(props) == str:
            return props

        return_msg = "I was unsure what to do with the lights."
        state = self.prev_state
        if "on_or_off" in props and props["on_or_off"] == "off":
            state.on_or_off = "off"
            return_msg = "I have turned off the lights as you requested."

        # Turn on with no details, assumes default
        if props == {"on_or_off": "on"} or "default" in props and props["default"]:
            state.on_or_off = "on"
            state.cct = True
            state.brightness = 100
            return_msg = "I have turned on your usual lights."

        if "brightness" in props:
            state.on_or_off = "on"
            state.brightness = props["brightness"]
            new_color = tuple([int(props["brightness"]/100 * x) for x in state.color])
            state.color = new_color
            return_msg = "I have adjusted the lights as you requested"

        if "color" in props:
            state.cct = False
            state.on_or_off = "on"  # on is implied when asking for a color
            r, g, b = props["color"]
            state.color = (r, g, b)
            return_msg = "I have adjusted the lights as you requested"

        self.prev_state = state    # no need to cache here, it's just used for the listening call
        self._set_led_state(state)
        return return_msg + ". Is there anything else I can assist with?"



if __name__ == "__main__":
    mcp.run()
