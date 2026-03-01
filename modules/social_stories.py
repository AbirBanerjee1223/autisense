"""
Social Story Generator using Google Gemini.
Follows Carol Gray's Social Stories™ criteria (2010 revision):
  - Written in 1st person (child's perspective)
  - Uses descriptive, perspective, directive, affirmative, cooperative sentences
  - Maintains 2-5 descriptive/perspective sentences per directive sentence
  - Age-appropriate vocabulary and sentence length

Reference: Gray, C. (2010). The New Social Story Book. Future Horizons.
"""

import os
from typing import Optional, List, Dict
from datetime import datetime

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ─────────────────────────────────────────────
# PRE-BUILT TOPIC LIBRARY
# ─────────────────────────────────────────────
STORY_TOPICS = {
    "Daily Routines": [
        {
            "title": "Going to the Doctor",
            "prompt_hint": "visiting the doctor for a checkup, what happens in the waiting room, examination",
            "icon": "🏥",
        },
        {
            "title": "Getting a Haircut",
            "prompt_hint": "sitting in the barber/salon chair, sounds of clippers, getting hair washed",
            "icon": "💇",
        },
        {
            "title": "Going to the Grocery Store",
            "prompt_hint": "bright lights, many people, staying with parent, waiting in line to pay",
            "icon": "🛒",
        },
        {
            "title": "Taking a Bath",
            "prompt_hint": "getting undressed, water temperature, washing hair, drying off",
            "icon": "🛁",
        },
        {
            "title": "Bedtime Routine",
            "prompt_hint": "brushing teeth, putting on pajamas, reading a story, turning off lights",
            "icon": "🌙",
        },
    ],
    "Social Situations": [
        {
            "title": "Making a Friend at School",
            "prompt_hint": "approaching another child, saying hello, asking to play, sharing toys",
            "icon": "🤝",
        },
        {
            "title": "Waiting for My Turn",
            "prompt_hint": "taking turns in a game, waiting in line, being patient",
            "icon": "⏳",
        },
        {
            "title": "When Someone Says Hello",
            "prompt_hint": "recognizing a greeting, making eye contact, saying hello back, waving",
            "icon": "👋",
        },
        {
            "title": "Playing with Others",
            "prompt_hint": "joining group play, sharing, following game rules, handling disagreements",
            "icon": "🎮",
        },
        {
            "title": "Asking for Help",
            "prompt_hint": "recognizing when help is needed, who to ask, how to ask politely",
            "icon": "🙋",
        },
    ],
    "Emotional Regulation": [
        {
            "title": "When Plans Change Unexpectedly",
            "prompt_hint": "something planned gets cancelled or changed, feeling upset, coping strategies",
            "icon": "🔄",
        },
        {
            "title": "Feeling Angry",
            "prompt_hint": "recognizing anger in my body, calming strategies, breathing, asking for space",
            "icon": "😤",
        },
        {
            "title": "Feeling Worried or Scared",
            "prompt_hint": "noticing worry feelings, talking to a safe adult, coping techniques",
            "icon": "😰",
        },
        {
            "title": "When I Feel Overwhelmed",
            "prompt_hint": "too much noise or stimulation, finding a quiet space, self-regulation",
            "icon": "🌊",
        },
        {
            "title": "Saying Goodbye",
            "prompt_hint": "when a parent leaves, feeling sad, knowing they will come back",
            "icon": "👋",
        },
    ],
    "Transitions": [
        {
            "title": "Starting a New School",
            "prompt_hint": "new classroom, new teacher, new kids, everything will be okay",
            "icon": "🏫",
        },
        {
            "title": "Moving to a New House",
            "prompt_hint": "packing, new room, new neighborhood, keeping my things",
            "icon": "🏠",
        },
        {
            "title": "When the Activity Changes",
            "prompt_hint": "stopping one activity to start another, transition warning, timer",
            "icon": "⏰",
        },
    ],
}


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SOCIAL_STORY_SYSTEM_PROMPT = """You are a certified behavioral therapist and Carol Gray Social Stories™ trained professional.

Generate a social story following Carol Gray's Social Stories 10.2 criteria exactly:

SENTENCE TYPES (you MUST use all of these):
1. DESCRIPTIVE sentences: Truthfully describe the context, setting, people, and what happens. (e.g., "The doctor's office has a waiting room with chairs.")
2. PERSPECTIVE sentences: Describe the feelings, thoughts, or motivations of other people. (e.g., "My mom feels happy when I try new things.")
3. DIRECTIVE sentences: Gently suggest a desired response. Use "I will try to..." or "I can..." NOT commands. (e.g., "I will try to sit quietly while I wait.")
4. AFFIRMATIVE sentences: Express a commonly shared value or truth. (e.g., "It is okay to feel nervous about new things.")
5. COOPERATIVE sentences: Identify who will help. (e.g., "My teacher will help me if I feel confused.")

RATIO RULE: For every 1 directive sentence, include 2-5 descriptive/perspective/affirmative sentences.

FORMATTING RULES:
- Write in FIRST PERSON from the child's perspective ("I", "me", "my")
- Use simple, concrete, literal language appropriate for the specified age
- Keep sentences short (under 15 words for young children)
- Total length: 150-250 words
- Use a positive, reassuring tone throughout
- End with an affirmative or positive statement
- NO metaphors, idioms, or abstract language
- NO sarcasm or exaggeration
- Add a title line at the top

OUTPUT FORMAT:
Title: [Story Title]

[Story paragraphs, each 2-4 sentences, separated by blank lines]
"""


# ─────────────────────────────────────────────
# GENERATOR CLASS
# ─────────────────────────────────────────────
class SocialStoryGenerator:
    """Generate personalized social stories using Google Gemini."""

    def __init__(self):
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Gemini model if API key is available."""
        if not GEMINI_AVAILABLE:
            return

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=SOCIAL_STORY_SYSTEM_PROMPT,
            )
        except Exception:
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def generate_story(
        self,
        topic: str,
        child_age: int = 4,
        additional_context: str = "",
        screening_domains: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate a social story for a given topic.
        
        Parameters:
            topic: The situation/topic for the story
            child_age: Child's age in years (affects vocabulary complexity)
            additional_context: Any extra context from the parent
            screening_domains: Flagged domains from screening to personalize content
            
        Returns:
            Dict with 'title', 'story', 'metadata', and 'success' flag
        """
        if not self.is_available:
            return self._fallback_story(topic, child_age)

        # Build the generation prompt
        prompt = self._build_prompt(topic, child_age, additional_context, screening_domains)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=800,
                    top_p=0.9,
                ),
            )

            story_text = response.text.strip()
            title, body = self._parse_story(story_text, topic)

            return {
                "success": True,
                "title": title,
                "story": body,
                "full_text": story_text,
                "metadata": {
                    "topic": topic,
                    "child_age": child_age,
                    "generated_at": datetime.now().isoformat(),
                    "model": "gemini-2.0-flash",
                    "method": "carol_gray_10.2",
                },
            }

        except Exception as e:
            return {
                "success": False,
                "title": f"Story: {topic}",
                "story": f"Story generation encountered an error: {str(e)}. Please try again.",
                "full_text": "",
                "metadata": {"error": str(e)},
            }

    def generate_from_preset(
        self,
        category: str,
        topic_index: int,
        child_age: int = 4,
        screening_domains: Optional[List[str]] = None,
    ) -> Dict:
        """Generate a story from the pre-built topic library."""
        if category not in STORY_TOPICS:
            return {"success": False, "title": "Error", "story": f"Category '{category}' not found."}

        topics = STORY_TOPICS[category]
        if topic_index < 0 or topic_index >= len(topics):
            return {"success": False, "title": "Error", "story": f"Topic index {topic_index} out of range."}

        topic_data = topics[topic_index]
        return self.generate_story(
            topic=topic_data["title"],
            child_age=child_age,
            additional_context=topic_data["prompt_hint"],
            screening_domains=screening_domains,
        )

    def _build_prompt(
        self,
        topic: str,
        child_age: int,
        additional_context: str,
        screening_domains: Optional[List[str]],
    ) -> str:
        age_guidance = ""
        if child_age <= 3:
            age_guidance = "Use very simple words (1-2 syllables). Max 8 words per sentence. Use concrete nouns only."
        elif child_age <= 5:
            age_guidance = "Use simple vocabulary. Max 12 words per sentence. Avoid abstract concepts."
        elif child_age <= 8:
            age_guidance = "Use age-appropriate vocabulary. Max 15 words per sentence. Simple cause-and-effect is okay."
        else:
            age_guidance = "Use clear but slightly more complex language appropriate for older children."

        prompt = f"""Generate a social story about: {topic}

Child's age: {child_age} years old
Language guidance: {age_guidance}
"""
        if additional_context:
            prompt += f"\nAdditional context about the situation: {additional_context}\n"

        if screening_domains:
            prompt += (
                f"\nThis child has been flagged for concerns in these developmental areas: "
                f"{', '.join(screening_domains)}. "
                f"Subtly incorporate support for these areas where relevant "
                f"(e.g., if 'eye contact' is flagged, gently include a line about looking at faces).\n"
            )

        prompt += "\nGenerate the social story now:"
        return prompt

    def _parse_story(self, raw_text: str, fallback_title: str) -> tuple:
        """Parse the title and body from Gemini's response."""
        lines = raw_text.strip().split("\n")
        title = fallback_title
        body_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower().startswith("title:"):
                title = stripped[6:].strip().strip('"').strip("*")
            elif stripped.startswith("# "):
                title = stripped[2:].strip()
            elif stripped.startswith("**") and stripped.endswith("**") and i == 0:
                title = stripped.strip("*").strip()
            else:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        if not body:
            body = raw_text

        return title, body

    def _fallback_story(self, topic: str, child_age: int) -> Dict:
        """Return a pre-written template if Gemini is not available."""
        story = (
            f"Sometimes I go to new places or do new things.\n\n"
            f"Today I am learning about: {topic}.\n\n"
            f"It is okay to feel a little nervous about new things. "
            f"Many children feel this way.\n\n"
            f"I can take a deep breath if I feel worried. "
            f"I can count to five slowly: 1... 2... 3... 4... 5.\n\n"
            f"There are people around me who want to help. "
            f"I can ask my parent or teacher if I need help.\n\n"
            f"I will try my best, and that is always enough. "
            f"I am brave and I am learning new things every day."
        )
        return {
            "success": True,
            "title": f"My Story About {topic}",
            "story": story,
            "full_text": story,
            "metadata": {
                "topic": topic,
                "child_age": child_age,
                "generated_at": datetime.now().isoformat(),
                "model": "fallback_template",
                "method": "pre_written",
            },
        }

    @staticmethod
    def get_topic_library() -> Dict:
        """Return the full pre-built topic library for UI rendering."""
        return STORY_TOPICS