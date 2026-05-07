import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

"""
Before submitting the assignment, describe here in a few sentences what you would have built next
if you spent 2 more hours on this project:

If I had 2 more hours, I would:
1. Add text-to-speech output so the story can be read aloud to the child automatically.
2. Build a "Story Library" that saves generated stories to a local JSON file so users can
   re-read favorites.
3. Add illustration prompts — after generating the story, produce DALL-E prompts for each
   story scene so a parent could optionally generate picture-book images.
4. Implement a multi-turn "campfire mode" where the child can ask "what happens next?" and
   the story continues chapter by chapter.
"""

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL = "gpt-3.5-turbo"
MAX_JUDGE_LOOPS = 3
PASSING_SCORE = 7

# ─────────────────────────────────────────────
# Genre-specific prompt templates
# ─────────────────────────────────────────────
GENRE_PROMPTS = {
    "adventure": (
        "You are an expert children's adventure author. Write an exciting bedtime story for ages 5-10. "
        "Use vivid action scenes, brave characters, and exotic settings. "
        "Keep the danger playful and non-scary — the hero always finds a clever way through."
    ),
    "funny": (
        "You are a comedic children's author famous for silly stories. Write a hilarious bedtime story for ages 5-10. "
        "Use wordplay, absurd situations, funny sounds, and goofy characters. "
        "Make the child giggle but still wind down for sleep by the end."
    ),
    "educational": (
        "You are an educational children's author. Write a bedtime story for ages 5-10 that sneaks in a real-world lesson "
        "(science, history, nature, kindness, etc.). Weave the learning naturally into the plot — never lecture."
    ),
    "sleepy": (
        "You are a gentle children's author known for calming bedtime stories. Write a soft, soothing story for ages 5-10. "
        "Use slow pacing, cozy settings (warm blankets, starry skies, gentle rain), repetitive rhythms, and a quiet ending "
        "that eases the child into sleep."
    ),
    "fantasy": (
        "You are a children's fantasy author. Write a magical bedtime story for ages 5-10 with enchanted creatures, "
        "mystical lands, and wonder. Keep magic whimsical and light — no dark sorcery."
    ),
    "moral": (
        "You are a children's author who writes heartwarming stories with a moral lesson. "
        "Write a bedtime story for ages 5-10 where the characters learn about friendship, honesty, courage, "
        "sharing, or empathy through their experiences. Show, don't tell — let the moral emerge from the plot."
    ),
}

STORY_ARC_INSTRUCTION = (
    "\n\nStructure the story using this arc:\n"
    "1. **Opening** — Set the scene and introduce the main character in a warm, inviting way.\n"
    "2. **Inciting Incident** — Something unexpected happens that kicks off the adventure.\n"
    "3. **Rising Action** — The character faces a fun challenge or puzzle.\n"
    "4. **Climax** — The most exciting moment, resolved through cleverness, kindness, or teamwork.\n"
    "5. **Resolution** — Everything works out; new friendships or lessons are celebrated.\n"
    "6. **Goodnight Moment** — A cozy, calming final paragraph that helps the child drift to sleep.\n"
    "\nAim for about 400-500 words. Start the story with a creative title on its own line."
)


# ─────────────────────────────────────────────
# Core API Helper
# ─────────────────────────────────────────────
def call_model(system_prompt: str, user_message: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """Call the OpenAI Chat API with error handling and retries."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API Key missing! Please add OPENAI_API_KEY to your .env file.")

    openai.api_key = api_key

    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message["content"]  # type: ignore
        if not content or not content.strip():
            raise ValueError("Model returned an empty response.")
        return content.strip()
    except openai.error.RateLimitError:
        print("⚠️  Rate limited by OpenAI. Please wait a moment and try again.")
        raise
    except openai.error.AuthenticationError:
        print("⚠️  Invalid API key. Check your .env file.")
        raise
    except Exception as e:
        print(f"⚠️  API call failed: {e}")
        raise


# ─────────────────────────────────────────────
# Agent 1: Intent Classifier
# ─────────────────────────────────────────────
def classify_intent(user_input: str) -> str:
    """Classify the user's story request into a genre category."""
    classifier_prompt = (
        "You are a story-request classifier. Given a child's story request, classify it into "
        "exactly ONE of these categories:\n"
        "  adventure, funny, educational, sleepy, fantasy, moral\n\n"
        "Rules:\n"
        "- If the request mentions jokes, humor, silliness → funny\n"
        "- If it mentions magic, dragons, fairies, enchanted → fantasy\n"
        "- If it mentions exploring, quests, treasure, space → adventure\n"
        "- If it mentions learning, science, history, nature facts → educational\n"
        "- If it mentions calm, sleepy, cozy, goodnight, relaxing → sleepy\n"
        "- If it mentions friendship, sharing, honesty, lessons, kindness → moral\n"
        "- If ambiguous, pick the BEST fit.\n\n"
        "Respond with ONLY the single lowercase category word. Nothing else."
    )
    result = call_model(classifier_prompt, user_input, max_tokens=20, temperature=0.0)
    genre = result.strip().lower().rstrip(".")

    # Fallback if the model returns something unexpected
    if genre not in GENRE_PROMPTS:
        print(f"   (Classifier returned '{genre}', defaulting to 'adventure')")
        genre = "adventure"
    return genre


# ─────────────────────────────────────────────
# Story Completeness Validator
# ─────────────────────────────────────────────
def validate_completeness(story: str) -> bool:
    """
    Check if a story appears complete (not truncated mid-sentence).
    Returns True if the story looks complete, False if it may be cut off.
    """
    stripped = story.strip()
    if not stripped:
        return False
    # A complete story should end with sentence-ending punctuation or a closing quote/asterisk
    last_char = stripped[-1]
    valid_endings = {'.', '!', '?', '"', "'", '*', '~'}
    if last_char not in valid_endings:
        return False
    # Check for obvious mid-sentence truncation patterns
    lines = stripped.split('\n')
    last_line = lines[-1].strip()
    # If the last non-empty line is very short and doesn't end properly, suspect truncation
    incomplete_markers = [' the', ' a', ' an', ' of', ' and', ' but', ' or', ' to', ' in', ' with']
    for marker in incomplete_markers:
        if last_line.lower().endswith(marker):
            return False
    return True


# ─────────────────────────────────────────────
# Agent 2: Storyteller
# ─────────────────────────────────────────────
def generate_draft(topic: str, genre: str) -> str:
    """Generate a story draft using the genre-tailored prompt + story arc."""
    system_prompt = GENRE_PROMPTS[genre] + STORY_ARC_INSTRUCTION
    return call_model(system_prompt, f"Story request: {topic}", temperature=0.85)


# ─────────────────────────────────────────────
# Agent 3: Judge (with scoring)
# ─────────────────────────────────────────────
def judge_story(draft: str) -> tuple[int, str]:
    """
    Evaluate the draft and return (score, feedback).
    Score is 1-10. Feedback is actionable bullet points.
    """
    judge_prompt = (
        "You are a strict but fair editor for children's bedtime stories (target audience: ages 5-10).\n\n"
        "Evaluate the story draft on these criteria:\n"
        "  1. Age Appropriateness — No scary, violent, or complex themes.\n"
        "  2. Story Arc — Clear beginning, middle, climax, and resolution.\n"
        "  3. Engagement — Vivid characters, fun dialogue, imaginative scenes.\n"
        "  4. Bedtime Suitability — Calming ending, appropriate pacing, not over-stimulating.\n"
        "  5. Language — Simple vocabulary, short sentences, easy to read aloud.\n\n"
        "Respond in EXACTLY this JSON format and nothing else:\n"
        '{\n'
        '  "score": <integer 1-10>,\n'
        '  "strengths": ["<strength1>", "<strength2>"],\n'
        '  "improvements": ["<improvement1>", "<improvement2>", "<improvement3>"]\n'
        '}'
    )
    raw = call_model(judge_prompt, f"Story draft to evaluate:\n\n{draft}", temperature=0.2)

    # Parse the JSON response robustly
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned.strip())
        score = int(data.get("score", 5))
        strengths = data.get("strengths", [])
        improvements = data.get("improvements", [])
        feedback = ""
        if strengths:
            feedback += "Strengths:\n" + "\n".join(f"  ✅ {s}" for s in strengths)
        if improvements:
            feedback += "\nAreas to Improve:\n" + "\n".join(f"  🔧 {i}" for i in improvements)
        return (score, feedback)
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: treat entire response as feedback, assign neutral score
        print("   (Judge response was not valid JSON — using raw feedback)")
        return (5, raw)


# ─────────────────────────────────────────────
# Agent 4: Revisor
# ─────────────────────────────────────────────
def revise_story(draft: str, feedback: str, genre: str) -> str:
    """Rewrite the story by incorporating the judge's feedback."""
    revisor_prompt = (
        f"You are an expert children's {genre} author revising a bedtime story for ages 5-10.\n"
        "You will receive the original draft and an editor's feedback.\n"
        "Rewrite the story, strictly addressing every piece of feedback.\n"
        "Maintain the same story arc structure:\n"
        "  Opening → Inciting Incident → Rising Action → Climax → Resolution → Goodnight Moment.\n"
        "Keep the title. Aim for 400-500 words. Output ONLY the revised story."
    )
    user_msg = (
        f"Original Draft:\n{draft}\n\n"
        f"Editor Feedback:\n{feedback}\n\n"
        "Please provide the complete revised story."
    )
    return call_model(revisor_prompt, user_msg, temperature=0.7)


# ─────────────────────────────────────────────
# Agent 5: Extras Generator (Surprise factor)
# ─────────────────────────────────────────────
def generate_extras(story: str) -> str:
    """Generate bonus content: moral, parent read-aloud tips, and discussion questions."""
    extras_prompt = (
        "You are a children's literacy expert. Given the following bedtime story, produce:\n"
        "1. **Moral of the Story** — One sentence a parent can share with the child.\n"
        "2. **Read-Aloud Tips** — 2 short tips for the parent (e.g., 'Use a squeaky voice for Milo').\n"
        "3. **Discussion Questions** — 2 simple questions to ask the child after the story.\n"
        "4. **Reading Level** — Estimate the grade level (e.g., 'Kindergarten to 2nd Grade').\n\n"
        "Format your response with clear headers for each section. Keep it concise."
    )
    return call_model(extras_prompt, f"Story:\n{story}", max_tokens=500, temperature=0.5)


# ─────────────────────────────────────────────
# Orchestrator: Full Pipeline
# ─────────────────────────────────────────────
def run_story_pipeline(user_input: str) -> tuple[str, str, str]:
    """
    Full maker-checker pipeline:
      1. Classify intent → pick genre
      2. Generate draft with genre-specific prompt + story arc
      3. Judge loop: score → revise → re-judge (up to MAX_JUDGE_LOOPS)
      4. Generate bonus extras
      5. Return (final_story, extras, genre)
    """

    # ── Step 1: Classify ──
    print("\n[Step 1/4] 🏷️  Classifying your story request...")
    genre = classify_intent(user_input)
    print(f"   Detected genre: {genre.upper()}")

    # ── Step 2: First Draft ──
    print(f"\n[Step 2/4] ✍️  Storyteller ({genre}) is writing the first draft...")
    current_draft = generate_draft(user_input, genre)

    # Validate completeness — regenerate once if truncated
    if not validate_completeness(current_draft):
        print("   ⚠️  Draft appears truncated. Regenerating...")
        current_draft = generate_draft(user_input, genre)

    # ── Step 3: Judge Loop ──
    print(f"\n[Step 3/4] ⚖️  Judge is reviewing (up to {MAX_JUDGE_LOOPS} rounds)...")
    for iteration in range(1, MAX_JUDGE_LOOPS + 1):
        score, feedback = judge_story(current_draft)
        print(f"\n   --- Round {iteration} | Score: {score}/10 ---")
        print(f"{feedback}")

        if score >= PASSING_SCORE:
            print(f"\n   ✅ Story passed the Judge (score {score}/10). Moving on!")
            break
        else:
            if iteration < MAX_JUDGE_LOOPS:
                print(f"\n   📝 Score below {PASSING_SCORE}. Revising...")
                current_draft = revise_story(current_draft, feedback, genre)
                if not validate_completeness(current_draft):
                    print("   ⚠️  Revision appears truncated. Requesting rewrite...")
                    current_draft = revise_story(current_draft, feedback + "\n\nCRITICAL: The previous revision was cut off mid-sentence. Ensure the story is COMPLETE.", genre)
            else:
                print(f"\n   ⚠️  Max revisions reached. Proceeding with best version.")

    # ── Step 4: Extras ──
    print("\n[Step 4/4] 🎁 Generating bonus content for parents...")
    extras = generate_extras(current_draft)

    return current_draft, extras, genre


# ─────────────────────────────────────────────
# Human-in-the-Loop: Post-Story Feedback
# ─────────────────────────────────────────────
def handle_user_feedback(story: str, genre: str) -> str:
    """Allow the user to request changes to the story iteratively."""
    while True:
        print("\n💬 Would you like any changes to the story?")
        print("   (Type your feedback, or 'done' to keep the story as-is)")
        user_feedback = input("   > ").strip()

        # Detect acceptance phrases — don't treat "perfect" as revision feedback
        positive_exits = {
            "done", "no", "nope", "exit", "quit", "perfect", "love it",
            "looks good", "looks great", "great", "awesome", "wonderful",
            "it's perfect", "its perfect", "i love it", "that's great",
            "thats great", "all good", "no changes", "keep it", "good",
            "amazing", "beautiful", "yes", "lovely",
        }
        if not user_feedback or user_feedback.lower().strip("!. ") in positive_exits:
            print("   Great! Keeping the story as-is. 🌙")
            return story

        print("\n   🛠️  Revising based on your feedback...")
        revisor_prompt = (
            f"You are an expert children's {genre} author. A parent has requested changes to a bedtime story.\n"
            "Rewrite the story incorporating their feedback while maintaining quality for ages 5-10.\n"
            "Keep the story arc structure and title. Output ONLY the revised story."
        )
        user_msg = (
            f"Current Story:\n{story}\n\n"
            f"Parent's Requested Changes:\n{user_feedback}\n\n"
            "Please provide the revised story."
        )
        story = call_model(revisor_prompt, user_msg, temperature=0.7)

        print("\n" + "=" * 52)
        print("          ✨ REVISED STORY ✨")
        print("=" * 52 + "\n")
        print(story)
        print("\n" + "=" * 52)

    return story


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 52)
    print("       🌙 AI Bedtime Storyteller 🌙")
    print("=" * 52)
    print("  Powered by a multi-agent story engine:")
    print("  Classifier → Storyteller → Judge Loop → You!")
    print("=" * 52)

    user_input = input("\nWhat kind of story would you like to hear?\n> ").strip()

    if not user_input:
        print("No story request provided. Goodnight! 🌙")
        return

    # Run the full pipeline
    story, extras, genre = run_story_pipeline(user_input)

    # Display the final story
    print("\n" + "=" * 52)
    print("          🌟 YOUR BEDTIME STORY 🌟")
    print("=" * 52 + "\n")
    print(story)
    print("\n" + "=" * 52)

    # Human-in-the-loop feedback
    story = handle_user_feedback(story, genre)

    # Display extras
    print("\n" + "=" * 52)
    print("          📚 PARENT'S COMPANION GUIDE 📚")
    print("=" * 52 + "\n")
    print(extras)
    print("\n" + "=" * 52)
    print("          Sweet dreams! 🌙✨")
    print("=" * 52)


if __name__ == "__main__":
    main()