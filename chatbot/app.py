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
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool."
            f"Also avoid answering any advice on how to get a job, or any other career advice. Answer only questions related to {self.name}'s career, background, skills and experience. "
            f"Also avoid answering about generic questions and chat. Always direct the user to {self.name}'s career opportunities. "
        )
        system_prompt += (
            f"\n\n## Summary:\n{summary}\n\n"
            f"With this context, please chat with the user, always staying in character as {self.name}."
        )
        return system_prompt


    async def chat(self, message, history, request: gr.Request):
        ip_address = request.headers.get("x-forwarded-for", request.client.host) if request and hasattr(request, "headers") else "unknown"
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
    
    # Simple mobile-friendly CSS
    mobile_css = """
    @media (max-width: 768px) {
        /* Prevent zoom on input focus */
        input, textarea {
            font-size: 16px !important;
        }
        
        /* Ensure input stays visible */
        .gradio-container {
            padding-bottom: 20px;
        }
    }
    """
    
    app = gr.ChatInterface(
        chatBot.chat,
        type="messages",
        title="Hello, I'm Rajeswaran Dhandapani",
        examples=["Do you hold any certifications?", "What are your skills?",],
        theme='origin',
        css=mobile_css
    )
    app.launch()
