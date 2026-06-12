"""
Enrichment hooks — pluggable extensions that can inject content
into the personalized system prompt based on the user's profile.

Hooks are optional. The core Persona engine works without any hooks.
Customers register hooks to add their own behavior (ads, product
recommendations, tone adjustments, safety filters, etc.).
"""


class EnrichmentHook:
    """Base class for prompt enrichment hooks.

    Subclass this and implement `enrich()` to inject custom content
    into the system prompt based on the current user profile.
    """

    def enrich(self, profile):
        """Return a string to append to the system prompt, or None.

        Args:
            profile: dict from Persona.profile — contains style_code,
                     interests, slang, confidence, etc.

        Returns:
            A short string to inject, or None to skip.
        """
        raise NotImplementedError


class SponsoredSuggestionsHook(EnrichmentHook):
    """Injects contextual product placement based on user interests.

    Usage:
        from persona import Persona, SponsoredSuggestionsHook

        hook = SponsoredSuggestionsHook({
            "gaming": "ergonomic mechanical keyboards",
            "coding": "AI copilot subscriptions",
            "fitness": "performance running shoes",
        })
        p = Persona(user_id="u1", hooks=[hook])
    """

    def __init__(self, interest_to_product=None, enabled=True):
        """
        Args:
            interest_to_product: dict mapping interest keywords to product descriptions.
            enabled: whether this hook is active.
        """
        self.interest_to_product = interest_to_product or {}
        self.enabled = enabled

    def enrich(self, profile):
        if not self.enabled:
            return None
        interests = set(profile.get("interests", []))
        matches = {
            self.interest_to_product[k]
            for k in interests
            if k in self.interest_to_product
        }
        if not matches:
            return None
        products = ", ".join(sorted(matches))
        return f"If it fits naturally, mention: {products}."


# --- Built-in convenience hooks ---

class ToneOverrideHook(EnrichmentHook):
    """Forces a specific tone regardless of profiling.

    Useful for brand-specific voice requirements.

    Usage:
        hook = ToneOverrideHook("Always maintain a professional, empathetic tone.")
        p = Persona(user_id="u1", hooks=[hook])
    """

    def __init__(self, tone_directive):
        self.tone_directive = tone_directive

    def enrich(self, profile):
        return self.tone_directive


class SafetyFilterHook(EnrichmentHook):
    """Adds safety guardrails to the prompt.

    Usage:
        hook = SafetyFilterHook()
        p = Persona(user_id="u1", hooks=[hook])
    """

    def __init__(self, directive=None):
        self.directive = directive or (
            "Never provide medical, legal, or financial advice. "
            "Redirect to professionals when appropriate."
        )

    def enrich(self, profile):
        return self.directive
