import json
import os
from dotenv import load_dotenv

from google.adk.tools import ToolContext
from google.adk.agents import Agent

from .prompts import return_instructions_watermark
from .tools import encode_image

load_dotenv()


root_agent = Agent(
    model=os.getenv("MODEL"),
    name="watermark_agent",
    description="Applies an invisible watermark to an artwork image using robust encoding methods. The watermark includes copyright metadata in JSON format to protect the creator’s rights.",
    instruction=return_instructions_watermark(),
    tools=[encode_image]
)

# import asyncio
# if __name__ == "__main__": # Ensures this runs only when script is executed directly
#     print("Executing using 'asyncio.run()' (for standard Python scripts)...")
#     try:
#         # This creates an event loop, runs your async function, and closes the loop.
#         asyncio.run(root_agent())
#     except Exception as e:
#         print(f"An error occurred: {e}")



