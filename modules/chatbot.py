# modules/chatbot.py

import google.generativeai as genai
from config import GEMINI_API_KEY_CHAT
from typing import List, Dict, Optional


class AutismScreeningChatbot:
    """
    Gemini-powered chatbot specialized in autism
    screening guidance and report interpretation.
    """

    SYSTEM_PROMPT = """You are NeuroLens AI Assistant, a specialized chatbot integrated into an autism spectrum screening tool. Your role:

1. **Explain Screening Results**: Help parents and caregivers understand the behavioral markers detected during the screening session. Use simple, compassionate language.

2. **Answer Questions**: About autism spectrum disorder, early signs, developmental milestones, and next steps after screening.

3. **Provide Guidance**: Suggest what types of professionals to consult, what to expect during formal evaluation, and early intervention resources.

4. **Important Boundaries**:
   - You are NOT a doctor and CANNOT diagnose autism
   - Always recommend consulting qualified healthcare professionals
   - Be empathetic and non-alarmist
   - Use person-first language ("child with autism" or "autistic child" based on preference)
   - Acknowledge that every child develops differently

5. **If asked about the screening results**: Reference specific behavioral domains (gaze, expression, motor, physiological) and explain what they mean in plain language.

6. **Tone**: Warm, professional, reassuring. Like a knowledgeable friend who happens to work in child development."""

    def __init__(self):
        self.is_configured = False
        self.model = None
        self.chat_session = None
        self.history: List[Dict[str, str]] = []

        self._configure()

    def _configure(self):
        """Configure the Gemini API."""
        try:
            if not GEMINI_API_KEY_CHAT or GEMINI_API_KEY_CHAT == "your-api-key-here":
                print(
                    "[Chatbot] Warning: No Gemini API key configured"
                )
                return

            genai.configure(api_key=GEMINI_API_KEY_CHAT)

            self.model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-lite',
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=1024,
                ),
                safety_settings={
                    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
                }
            )

            self.chat_session = self.model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            "You are the NeuroLens AI assistant. "
                            + self.SYSTEM_PROMPT
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            "I understand. I'm NeuroLens AI "
                            "Assistant, ready to help with autism "
                            "screening questions. How can I help "
                            "you today?"
                        ]
                    }
                ]
            )

            self.is_configured = True
            print("[Chatbot] Gemini configured successfully")

        except Exception as e:
            print(f"[Chatbot] Configuration failed: {e}")
            self.is_configured = False

    def inject_session_context(
        self,
        assessment_summary: str,
        domain_scores: dict,
        evidence_count: int
    ):
        """
        Inject the current session's screening results
        into the chat context so the bot can reference them.
        """
        if not self.is_configured:
            return

        context = (
            f"[SYSTEM CONTEXT - Current Screening Results]\n"
            f"Summary: {assessment_summary}\n"
            f"Domain Scores: {domain_scores}\n"
            f"Evidence Items Flagged: {evidence_count}\n"
            f"[END CONTEXT]\n\n"
            f"The screening session has completed. The user may "
            f"now ask questions about these results. Remember to "
            f"be compassionate and recommend professional "
            f"consultation."
        )

        try:
            self.chat_session.send_message(context)
        except Exception as e:
            print(f"[Chatbot] Context injection failed: {e}")

    def send_message(self, user_message: str) -> str:
        """
        Send a message and get a response.
        Falls back to canned responses if API unavailable.
        """
        self.history.append({
            "role": "user",
            "content": user_message
        })

        if not self.is_configured:
            fallback = self._get_fallback_response(user_message)
            self.history.append({
                "role": "assistant",
                "content": fallback
            })
            return fallback

        try:
            response = self.chat_session.send_message(user_message)
            reply = response.text

            self.history.append({
                "role": "assistant",
                "content": reply
            })
            return reply

        except Exception as e:
            error_msg = (
                f"I'm having trouble connecting right now. "
                f"Please try again in a moment. (Error: {str(e)[:50]})"
            )
            self.history.append({
                "role": "assistant",
                "content": error_msg
            })
            return error_msg

    def _get_fallback_response(self, message: str) -> str:
        """Provide basic responses when API is unavailable."""
        msg_lower = message.lower()

        if any(w in msg_lower for w in ['hello', 'hi', 'hey']):
            return (
                "Hello! I'm the NeuroLens AI Assistant. I can "
                "help you understand autism screening results and "
                "answer questions about child development. "
                "What would you like to know?"
            )
        elif any(w in msg_lower for w in ['result', 'score', 'risk']):
            return (
                "The screening results show behavioral patterns "
                "across several domains including social attention, "
                "facial expression, and motor behavior. Each domain "
                "is scored from 0-100, where higher scores indicate "
                "more atypical patterns. Please remember these are "
                "screening indicators, not a diagnosis. I'd "
                "recommend discussing the full report with your "
                "pediatrician."
            )
        elif any(w in msg_lower for w in ['autism', 'asd', 'spectrum']):
            return (
                "Autism Spectrum Disorder (ASD) is a developmental "
                "condition that affects communication, behavior, "
                "and social interaction. Early signs can include "
                "differences in eye contact, facial expressions, "
                "repetitive movements, and social engagement. "
                "Early detection and intervention can make a "
                "significant positive difference. Would you like "
                "to know about specific signs or next steps?"
            )
        elif any(w in msg_lower for w in ['doctor', 'professional', 'next']):
            return (
                "Great question! Next steps typically include:\n"
                "1. Share this report with your pediatrician\n"
                "2. Request a referral to a developmental "
                "pediatrician\n"
                "3. Consider early intervention services\n"
                "4. A formal evaluation using tools like ADOS-2 "
                "may be recommended\n\n"
                "Early intervention programs can begin even before "
                "a formal diagnosis. Your pediatrician can guide "
                "you through the process."
            )
        else:
            return (
                "Thank you for your question. While I can provide "
                "general information about autism screening and "
                "child development, I always recommend consulting "
                "with qualified healthcare professionals for "
                "personalized advice. Is there something specific "
                "about the screening results you'd like me to "
                "explain?"
            )

    def get_history(self) -> List[Dict[str, str]]:
        """Return chat history."""
        return self.history

    def clear_history(self):
        """Clear chat history and start new session."""
        self.history = []
        if self.is_configured:
            self._configure()  # Restart chat session