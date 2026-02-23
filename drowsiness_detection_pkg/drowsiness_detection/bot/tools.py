"""
Driver Alert Tools:
- Voice alerts (text-to-speech)
- Steering wheel vibration
"""

import time
import logging
import threading
from pydantic import BaseModel, Field
from langchain.tools import tool

# from .controls. import VoiceControl, WheelControlVibration

# Initialize control objects once
# voice = VoiceControl()
# steering = WheelControlVibration()

# Configure logging globally
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# schemas
class VoiceAlertSchema(BaseModel):
    """Schema for voice alert tool."""

    text: str = Field(..., description="The text to convert to speech.")


class VibrateSteeringSchema(BaseModel):
    """Schema for steering wheel vibration tool."""

    duration: float = Field(..., gt=0, le=3, description="Duration in seconds (max 3)")
    intensity: int = Field(..., ge=0, le=60, description="Vibration intensity (max 60)")


def run_in_thread(func, *args, **kwargs) -> None:
    """Run a function in a daemon thread."""
    thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    thread.start()


# tools
@tool(args_schema=VoiceAlertSchema)
def voice_alert(text: str) -> str:
    """Alert the driver via text-to-speech."""

    def _speak():
        try:
            voice.text_to_speech(text)
            logging.info(f"Voice alert started for text: '{text}'")
        except Exception as e:
            logging.error(f"[Voice Alert Error] {e}")

    run_in_thread(_speak)
    return f"Voice alert triggered with text: '{text}'"


@tool(args_schema=VibrateSteeringSchema)
def vibrate_steering_wheel(duration: float, intensity: int) -> str:
    """Vibrate the steering wheel to alert the driver."""

    def _vibrate():
        try:
            steering.vibrate(duration=duration, intensity=intensity)
            time.sleep(duration)
            steering.vibrate(duration=0)  # stop vibration
            logging.info(
                f"Steering wheel vibrated at {intensity} intensity for {duration}s"
            )
        except Exception as e:
            logging.error(f"[Steering Wheel Error] {e}")

    run_in_thread(_vibrate)
    return f"Steering wheel vibration triggered (intensity {intensity}, duration {duration}s)"
