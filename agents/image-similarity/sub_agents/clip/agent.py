import os
from dotenv import load_dotenv

from google.adk.agents import Agent

from .prompts import return_instructions_clip
from .tools import compute_clip_similarity

load_dotenv()


root_agent = Agent(
    model=os.getenv("MODEL"),
    name="clip_agent",
    description="Compares two images based on semantic similarity using CLIP embeddings. Useful for checking whether an image is a conceptual or AI-generated variant of another.",
    instruction=return_instructions_clip(),
    tools=[compute_clip_similarity]
)



