import os
from dotenv import load_dotenv

from google.adk.agents import Agent

from .prompts import return_instructions_lpips
from .tools import compute_lpips_similarity

load_dotenv()


root_agent = Agent(
    model=os.getenv("MODEL"),
    name="lpips_agent",
    description="Compares two images to measure their visual similarity using LPIPS (Learned Perceptual Image Patch Similarity). This agent is useful for detecting AI-generated content similarity or verifying derivative works.",
    instruction=return_instructions_lpips(),
    tools=[compute_lpips_similarity]
)



