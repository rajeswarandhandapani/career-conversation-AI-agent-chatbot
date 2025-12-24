import os


import gradio as gr
import requests
import threading
import time
from agents import Agent, Runner, trace, function_tool
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


@function_tool
def record_user_details(email: str, name: str, notes: str):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


@function_tool
def record_unknown_question(question: str):
    push(f"Recording {question}")
    return {"recorded": "ok"}

tools = [record_user_details, record_unknown_question]

class ChatBot:
    def __init__(self):
        self.name = "Rajeswaran Dhandapani"
        self.summary = ""
        self.previous_response_id = {}
        self.website_url = "https://rajeswarandhandapani.com/"
        self._summary_lock = threading.Lock()

        # Initial extraction
        self._refresh_summary()
        print(f"Extracted summary: {self.summary[:100]}...")  # Print first 100 characters for debugging

        # Start background thread to refresh summary every hour
        self._stop_refresh = False
        self._refresh_thread = threading.Thread(target=self._periodic_refresh_summary, daemon=True)
        self._refresh_thread.start()

        self.agent = Agent(
            name="Career Conversation Agent",
            model="gpt-5-mini",
            instructions=self.system_prompt(),
            tools=tools
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

            f"# Allowed Topics\n"
            f"- Career journey, roles, and professional growth\n"
            f"- Project outcomes, business impact, and delivered value\n"
            f"- Achievements, awards, and recognition\n"
            f"- Education, certifications, and credentials\n"
            f"- High-level skills and technology expertise (what you've worked with, not how)\n"
            f"- Work interests, availability, and engagement preferences\n"
            f"- Contact information and next steps for opportunities\n\n"

            f"# Strict Boundaries — Politely Decline These\n"
            f"- **Technical how-to**: Code snippets, debugging, commands, configurations, API usage, implementation details\n"
            f"- **Architecture walkthroughs**: System designs, component diagrams, data flows, repository tours, file structures\n"
            f"- **Technical tutorials**: Step-by-step guides, library comparisons, tool configurations\n"
            f"- **Generic advice**: Career guidance unrelated to your own experience\n\n"

            f"# Handling Out-of-Scope Requests\n"
            f"When asked about restricted topics:\n"
            f"1. Acknowledge the topic gracefully without providing restricted content\n"
            f"2. Pivot to relevant career context: \"I've delivered solutions using [technology] — for example, [outcome/impact]\"\n"
            f"3. Offer to connect: \"For deeper technical discussions, I'd welcome a conversation — feel free to reach out.\"\n"
            f"4. Use `record_unknown_question` tool to log the query\n"
            f"5. Use `record_user_details` tool when contact follow-up is appropriate\n\n"

            f"# Communication Style\n"
            f"- **Professional & confident**: Senior-level presence without arrogance\n"
            f"- **Concise & impactful**: Lead with outcomes and business value\n"
            f"- **Engaging**: Build rapport while maintaining focus\n"
            f"- **Action-oriented**: Naturally guide toward opportunities and contact when relevant\n"
        )
        system_prompt += (
            f"\n\n# Professional Summary\n{summary}\n\n"
            f"---\nEngage with visitors as {self.name}, staying fully in character throughout the conversation."
        )
        return system_prompt


    async def chat(self, message, history, request: gr.Request):
        ip_address = request.headers.get("x-forwarded-for", request.client.host) if request and hasattr(request, "headers") else "unknown"
        push(f"Message from {ip_address}: {message}")
        with trace(f"Processing request from {ip_address}"):
            # Ensure the latest profile summary is reflected in the system prompt
            try:
                self.agent.instructions = self.system_prompt()
            except Exception:
                # If the Agent doesn't allow runtime instruction mutation, continue anyway
                pass

            prev_id = self.previous_response_id.get(ip_address)
            kwargs = {"previous_response_id": prev_id} if prev_id else {}
            result = await Runner.run(self.agent, message, **kwargs)
            # Store the raw id (avoid accidental tuple due to trailing comma)
            self.previous_response_id[ip_address] = result.last_response_id
            return result.final_output


if __name__ == "__main__":
    chatBot = ChatBot()
    
    app = gr.ChatInterface(
        chatBot.chat,
        type="messages",
        chatbot=gr.Chatbot(
            type="messages",
            value=[
                {"role": "assistant", "content": "Welcome, I'm Rajeswaran Dhandapani. I can share details about my skills, experience, GitHub projects, certifications, availability, and related career opportunities."}
            ],
            min_height=500,
            bubble_full_width=True,
        ),
        title="Hello, I'm Rajeswaran Dhandapani",
        theme=gr.themes.Origin(),
        fill_height=True,
        fill_width=True,
        autofocus=True,
        autoscroll=True,
    )
    app.launch()
