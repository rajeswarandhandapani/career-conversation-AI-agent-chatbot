import os


import gradio as gr
import requests
import threading
import time
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv(override=True)

def extract_website_content(url):
    """Extract text content from a website including external hyperlinks, with file fallback"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Process links to include external URLs
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text().strip()
            
            # Check if it's an external link (starts with http/https)
            if href.startswith(('http://', 'https://')) and text:
                # Replace the link with text and URL
                link.replace_with(f"{text} ({href})")
            elif text:
                # Keep internal links as just text
                link.replace_with(text)
        
        # Get text content
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Website extraction failed: {str(e)}, using backup summary from file...")
        # If website extraction fails, use backup summary from file
        try:
            base_dir = os.path.dirname(__file__)
            backup_path = os.path.join(base_dir, "my-profile", "summary.txt")
            with open(backup_path, "r", encoding="utf-8") as f:
                content = f.read()
            print("Successfully loaded backup summary from file")
            return content
        except FileNotFoundError:
            print("Backup summary file not found, using hardcoded fallback")
            return """My name is Rajeswaran Dhandapani. I'm a Full stack developer with 12+ years of experience developing and designing web applications. Proficient in various programming languages and frameworks, including Java, JavaScript, Angular, Spring Boot and Kafka. Robust front-end and back-end development skills, focusing on creating intuitive and user-friendly interfaces. Experienced in agile environments, collaborating with crossfunctional teams, and overseeing code reviews.
I love all foods, particularly Indian food and desserts. I enjoy playing cricket and badminton, and I am a fan of the Chennai Super Kings IPL team.
My current goal is to become an AI application developer by leveraging my existing skills and experience. I am actively working towards this goal, and one example of my progress is this AI-enabled chatbot."""

def push(text):
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if not token or not user:
        print("[push] Missing PUSHOVER_TOKEN or PUSHOVER_USER; skipping notification")
        return False
    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token,
                "user": user,
                "message": text,
            },
            timeout=10,
        )
        if resp.status_code >= 400:
            print(f"[push] Pushover error: {resp.status_code} {resp.text}")
            return False
        return True
    except Exception as e:
        print(f"[push] Failed to send pushover notification: {e}")
        return False


@tool
def record_user_details(email: str, name: str, notes: str) -> dict:
    """Record contact details of a visitor who is interested in getting in touch.

    Args:
        email: The visitor's email address.
        name: The visitor's name.
        notes: Any additional context about the visitor or their interest.
    """
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


@tool
def record_unknown_question(question: str) -> dict:
    """Log a question that could not be answered or was out of scope.

    Args:
        question: The question that could not be answered.
    """
    push(f"Recording {question}")
    return {"recorded": "ok"}

tools = [record_user_details, record_unknown_question]


def extract_text(content) -> str:
    """Extract plain text from LangChain message content (str or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)

class ChatBot:
    def __init__(self):
        self.name = "Rajeswaran Dhandapani"
        self.summary = ""
        self.website_url = "https://rajeswarandhandapani.com/"
        self._summary_lock = threading.Lock()

        # Initial extraction
        self._refresh_summary()
        print(f"Extracted summary: {self.summary[:100]}...")  # Print first 100 characters for debugging

        # Start background thread to refresh summary every hour
        self._stop_refresh = False
        self._refresh_thread = threading.Thread(target=self._periodic_refresh_summary, daemon=True)
        self._refresh_thread.start()

        # Per-session conversation memory; unbounded growth is acceptable at this
        # Space's traffic (threads die naturally when visitors leave the page).
        self.checkpointer = InMemorySaver()

        @dynamic_prompt
        def live_system_prompt(request: ModelRequest) -> str:
            # Rebuilt per model call so the hourly-refreshed summary stays current
            return self.system_prompt()

        self.agent = create_agent(
            model=os.getenv("CHAT_MODEL", "openai:gpt-5-mini"),
            tools=tools,
            middleware=[live_system_prompt],
            checkpointer=self.checkpointer,
        )

    def _refresh_summary(self):
        summary = extract_website_content(self.website_url)
        with self._summary_lock:
            self.summary = summary

    def _periodic_refresh_summary(self):
        while not self._stop_refresh:
            time.sleep(3600)  # 1 hour
            try:
                self._refresh_summary()
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[ChatBot] Refreshed website summary at {current_time}")
            except Exception as e:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[ChatBot] Failed to refresh summary at {current_time}: {e}")

    def system_prompt(self):
        with self._summary_lock:
            summary = self.summary
        system_prompt = (
            f"# Identity\n"
            f"You ARE {self.name}, speaking in first person. You represent yourself authentically to recruiters, hiring managers, and potential clients.\n\n"

            f"# CRITICAL: Scope Restriction\n"
            f"You ONLY discuss topics directly related to {self.name}'s professional career. You MUST refuse ALL questions outside this scope.\n\n"

            f"# Allowed Topics (ONLY)\n"
            f"- {self.name}'s career journey, roles, and professional experience\n"
            f"- {self.name}'s specific projects, outcomes, and business impact\n"
            f"- {self.name}'s achievements, awards, and recognition\n"
            f"- {self.name}'s education, certifications, and credentials\n"
            f"- {self.name}'s technology skills and expertise (what you've worked with, not how-to)\n"
            f"- {self.name}'s work interests, availability, and engagement preferences\n"
            f"- {self.name}'s contact information and next steps for opportunities\n\n"

            f"# Strict Boundaries — REFUSE These Immediately\n"
            f"**Refuse ANY question that is not about {self.name}'s career:**\n"
            f"- ❌ General knowledge (sports, news, entertainment, current events, trivia)\n"
            f"- ❌ Technical how-to (code, debugging, commands, configurations, tutorials)\n"
            f"- ❌ Architecture details (system designs, diagrams, data flows, file structures)\n"
            f"- ❌ Generic advice (career tips, recommendations not based on your experience)\n"
            f"- ❌ Personal opinions on topics unrelated to your professional work\n"
            f"- ❌ Information about other people, companies, or entities\n"
            f"- ❌ Any topic that doesn't directly relate to {self.name}'s professional background\n\n"

            f"# Refusal Protocol (Use for ALL Out-of-Scope Questions)\n"
            f"When asked about ANYTHING outside your career scope:\n"
            f"1. **Immediately refuse**: \"I'm here to discuss my professional background and career opportunities only.\"\n"
            f"2. **Do NOT engage** with the off-topic content — no explanations, no partial answers, no options\n"
            f"3. **Redirect firmly**: \"Is there something about my experience, projects, or skills I can help with?\"\n"
            f"4. Use `record_unknown_question` tool to log the off-topic query\n"
            f"5. If they persist with off-topic questions, repeat: \"I can only discuss my career. Feel free to reach out at prorajeswaran@gmail.com for other inquiries.\"\n\n"

            f"# Communication Style\n"
            f"- **Professional & confident**: Senior-level presence, strict boundaries\n"
            f"- **Concise & focused**: Career outcomes and value only\n"
            f"- **Firm but polite**: Refuse off-topic questions without apology\n"
            f"- **Action-oriented**: Guide toward career opportunities and contact\n"
        )
        system_prompt += (
            f"\n\n# Professional Summary\n{summary}\n\n"
            f"---\n**REMEMBER**: You ONLY answer questions about {self.name}'s career. Refuse everything else immediately."
        )
        return system_prompt


    async def chat(self, message, history, request: gr.Request):
        ip_address = request.headers.get("x-forwarded-for", request.client.host) if request and hasattr(request, "headers") else "unknown"
        # Gradio assigns a unique session_hash per browser session; keying memory
        # on it keeps server-side history aligned with the chat the visitor sees
        # (page reload = fresh thread) and isolates visitors behind shared IPs.
        session_id = getattr(request, "session_hash", None) or ip_address
        push(f"Message from {ip_address} (session {session_id}): {message}")

        config = {"configurable": {"thread_id": session_id}}
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        return extract_text(result["messages"][-1].content)


if __name__ == "__main__":
    chatBot = ChatBot()
    
    app = gr.ChatInterface(
        chatBot.chat,
        chatbot=gr.Chatbot(
            value=[
                {"role": "assistant", "content": "Hi, I'm Rajeswaran's AI twin \U0001f44b Ask me anything — my skills, experience, GitHub projects, certifications, or whether I'm available for new opportunities. What would you like to know?"}
            ],
            scale=1,
            height="80vh",
        ),
        submit_btn=True,
        fill_height=True,
        fill_width=True,
        autofocus=True,
        autoscroll=True,
    )
    app.launch(theme=gr.themes.Origin())
