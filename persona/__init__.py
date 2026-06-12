"""
Persona — Real-time user personality profiling middleware for any LLM.

Usage:
    from persona import Persona

    # Create a persona engine for a user
    p = Persona(user_id="user_123")

    # Enhance any system prompt with personalization
    enhanced = p.enhance("You are a helpful assistant.")

    # After each conversation turn, observe what happened
    p.observe(user_message="hey whats up lol", assistant_message="Not much!")

    # Profile auto-saves and loads across sessions
    print(p.profile)
"""

from persona.engine import Persona
from persona.hooks import EnrichmentHook, SponsoredSuggestionsHook

__version__ = "0.1.0"
__all__ = ["Persona", "EnrichmentHook", "SponsoredSuggestionsHook"]
