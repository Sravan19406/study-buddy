import os
import requests
import streamlit as st
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.8-flash"

st.set_page_config(page_title="Study Buddy Agent", page_icon="📚")
st.title("📚 Study Buddy Agent")
st.caption("Type a topic. I'll summarize it, quiz you, then grade your answers.")


# ---------- Settings ----------
def secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def require_password():
    pw = secret("APP_PASSWORD")
    if not pw or st.session_state.get("authed"):
        return
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == pw:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


require_password()

api_key = secret("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY. Add it in the app's Secrets settings.")
    st.stop()

client = genai.Client(api_key=str(api_key).split()[0].strip())
MODEL = secret("GEMINI_MODEL", DEFAULT_MODEL)

SYSTEM = """You are Study Buddy, a friendly tutor.
When the student gives a topic:
1. Call search_wikipedia, then write a short, simple summary (max 6 sentences).
2. Create exactly 5 quiz questions and call save_quiz with the questions and correct answers.
3. Show ONLY the questions to the student (never the answers) and ask them to reply with their answers.
When the student replies with answers:
- For each question, call get_reference_answer, judge fairly (accept correct meaning, not exact wording),
  call record_result, and explain briefly why wrong answers are wrong.
- Finish with the final score and an encouraging tip. Then invite a new topic."""


# ---------- Tools (created per visitor session) ----------
def make_tools(quiz):
    def search_wikipedia(topic: str) -> str:
        """Search Wikipedia for background information on a topic."""
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "generator": "search", "gsrsearch": topic,
                    "gsrlimit": 1, "prop": "extracts", "exintro": 1,
                    "explaintext": 1, "format": "json"},
            headers={"User-Agent": "StudyBuddyWorkshop/1.0"}, timeout=10,
        )
        pages = r.json().get("query", {}).get("pages", {})
        if not pages:
            return "No results found."
        page = next(iter(pages.values()))
        return f"{page['title']}: {page['extract'][:3000]}"

    def save_quiz(questions: list[str], answers: list[str]) -> str:
        """Save the quiz. 'questions' and 'answers' are parallel lists of equal length. Answers are hidden from the student."""
        quiz.update(questions=questions, answers=answers, score=0, answered=0)
        return f"Saved {len(questions)} questions."

    def get_reference_answer(question_number: int) -> str:
        """Get the correct answer for a quiz question number (1-based)."""
        return quiz["answers"][question_number - 1]

    def record_result(correct: bool) -> str:
        """Record whether the student's answer was correct. Returns the running score."""
        quiz["answered"] += 1
        quiz["score"] += int(correct)
        return f"Score so far: {quiz['score']}/{quiz['answered']} answered (out of {len(quiz['questions'])})"

    return [search_wikipedia, save_quiz, get_reference_answer, record_result]


# ---------- Agent loop ----------
def run_agent(system, tools, contents, max_steps=12, on_tool=None):
    """Call LLM -> run requested tools -> repeat until the LLM gives a final answer."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    funcs = {f.__name__: f for f in tools}

    for _ in range(max_steps):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        contents.append(resp.candidates[0].content)

        calls = resp.function_calls
        if not calls:
            return resp.text or ""

        parts = []
        for call in calls:
            args = dict(call.args)
            if on_tool:
                on_tool(call.name, args)
            try:
                output = funcs[call.name](**args)
            except Exception as e:
                output = f"Tool error: {e}"
            parts.append(types.Part.from_function_response(
                name=call.name, response={"result": str(output)}))
        contents.append(types.Content(role="user", parts=parts))

    return "Stopped: too many steps."


# ---------- Session state ----------
def reset():
    quiz = {"questions": [], "answers": [], "score": 0, "answered": 0}
    st.session_state.tools = make_tools(quiz)
    st.session_state.contents = []
    st.session_state.display = []


if "contents" not in st.session_state:
    reset()

with st.sidebar:
    if st.button("🔄 Start over"):
        reset()
        st.rerun()

# ---------- Chat UI ----------
for role, text, logs in st.session_state.display:
    with st.chat_message(role):
        if logs:
            with st.expander("🔧 Tool calls"):
                for line in logs:
                    st.code(line)
        st.markdown(text)

prompt = st.chat_input("Type a topic, or answer the quiz...")
if prompt:
    st.session_state.display.append(("user", prompt, []))
    with st.chat_message("user"):
        st.markdown(prompt)

    contents = st.session_state.contents
    checkpoint = len(contents)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    logs = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = run_agent(SYSTEM, st.session_state.tools, contents,
                                  on_tool=lambda n, a: logs.append(f"{n}({str(a)[:150]})"))
            except Exception as e:
                del contents[checkpoint:]  # drop the half-finished turn
                reply = f"⚠️ Something went wrong: {e}"
        if logs:
            with st.expander("🔧 Tool calls"):
                for line in logs:
                    st.code(line)
        st.markdown(reply)
    st.session_state.display.append(("assistant", reply, logs))
