import os
from dotenv import load_dotenv

from google.adk.agents import Agent

from .prompts import return_instructions_adversarial_noise
from .tools import run_fgsm_attack

load_dotenv()


root_agent = Agent(
    model=os.getenv("MODEL"),
    name="adversarial_noise_agents",
    description="An agent that generates adversarial images using the FGSM method to evaluate model robustness and simulate potential AI attacks.",
    instruction=return_instructions_adversarial_noise(),
    tools=[run_fgsm_attack]
)

# import asyncio
# if __name__ == "__main__": # Ensures this runs only when script is executed directly
#     print("Executing using 'asyncio.run()' (for standard Python scripts)...")
#     try:
#         # This creates an event loop, runs your async function, and closes the loop.
#         asyncio.run(root_agent())
#     except Exception as e:
#         print(f"An error occurred: {e}")



