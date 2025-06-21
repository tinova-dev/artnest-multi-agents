from google.adk.agents import Agent

from .prompts import return_instructions_root
from .sub_agents import watermark_agent, adversarial_noise_agent



MODEL = 'gemini-1.5-flash-002'
ORIGINAL_PATH = './data/original'
PROTECTED_PATH = './data/protected'

root_agent = Agent(
  model=MODEL,
  name="copyright_multiagent",
  global_instruction="You are a Artwork Copyright Protection Multi Agent System.",
  instruction=return_instructions_root(),
  sub_agents=[watermark_agent, adversarial_noise_agent]
)




