"""
LLM-powered Purple Agent for Werewolf game.
A2A-compliant implementation with enhanced reasoning and bidding.
"""
from .agent import LLMWerewolfAgent
from .executor import LLMExecutor

__all__ = ["LLMWerewolfAgent", "LLMExecutor"]
