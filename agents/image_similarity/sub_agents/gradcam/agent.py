import os
from dotenv import load_dotenv

from google.adk.agents import Agent

from .prompts import return_instructions_gradcam
from .tools import compute_pixel_attribution

load_dotenv()


root_agent = Agent(
    model=os.getenv("MODEL"),
    name="gradcam_agent",
    description="Compares two images by visualizing and analyzing the regions most important for a model's prediction using Grad-CAM. Useful for assessing whether key visual semantics are preserved or altered, especially in derivative or AI-generated works.",
    instruction=return_instructions_gradcam(),
    tools=[compute_pixel_attribution]
)