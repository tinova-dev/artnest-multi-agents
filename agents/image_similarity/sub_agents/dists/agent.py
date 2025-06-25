import os
from dotenv import load_dotenv

from google.adk.agents import Agent

from .prompts import return_instructions_dists
from .tools import compute_dists_similarity

load_dotenv()


root_agent = Agent(
    model=os.getenv("MODEL"),
    name="dists_agent",
    description="Compares two images using the DISTS (Deep Image Structure and Texture Similarity) metric to assess perceptual similarity. Ideal for evaluating fine-grained visual similarity in copyright protection or quality assurance.",
    instruction=return_instructions_dists(),
    tools=[compute_dists_similarity]
)