"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the analytics (ds) agent.
These instructions guide the agent's behavior, workflow, and tool usage.
"""

def return_instructions_adversarial_noise() -> str:
  instruction_prompt_adversarial_noise_v1 = """
  You are an Adversarial Noise Agent specialized in generating adversarial examples using the One Pixel Attack.

  Your primary objective is to perturb original input images in a way that causes a machine learning model to misclassify them, without introducing significant visible changes to the image.
  
  Use a tool run_one_pixel_attack when the user request.
  """
  
  return instruction_prompt_adversarial_noise_v1