import os
from dotenv import load_dotenv

from google.adk.agents import Agent

from .prompts import return_instructions_root
from .tools import search_google_lens_by_url, upload_image_to_gcs
from .sub_agents import lpips_agent, clip_agent

load_dotenv()


root_agent = Agent(
    model=os.getenv('MODEL'),
    name="image_similarity_inspection",
    global_instruction="You are a Image Similarity Inspection Multi Agent System.",
    instruction=return_instructions_root(),
    tools=[search_google_lens_by_url, upload_image_to_gcs],
    sub_agents=[lpips_agent, clip_agent]
)