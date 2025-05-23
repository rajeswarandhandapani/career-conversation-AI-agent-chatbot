import os

import gradio as gr
import requests
from agents import Agent, Runner, trace, function_tool
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


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
        self.linkedin = ""
        self.summary = ""
        self.previous_response_id = {}
        reader = PdfReader("my-profile/linkedin.pdf")
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("my-profile/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
        self.agent = Agent(
            name="Career Conversation Agent",
            model="gpt-4o-mini",
            instructions=self.system_prompt(),
            tools=tools
        )

    def system_prompt(self):
        system_prompt = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool."
            f"Also avoid answering any advice on how to get a job, or any other career advice. Answer only questions related to {self.name}'s career, background, skills and experience. "
            f"Also avoid answering about generic questions and chat. Always direct the user to {self.name}'s career opportunities. "
        )
        system_prompt += (
            f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
            f"With this context, please chat with the user, always staying in character as {self.name}."
        )
        return system_prompt


    async def chat(self, message, history, request: gr.Request):
        ip_address = request.headers.get("x-forwarded-for", request.client.host) if request and hasattr(request, "headers") else "unknown"
        with trace(f"Processing request from {ip_address}"):
            prev_id = self.previous_response_id.get(ip_address)
            if prev_id:
                result = await Runner.run(self.agent, message, previous_response_id=prev_id[0])
            else:
                result = await Runner.run(self.agent, message)
            self.previous_response_id[ip_address]=result.last_response_id,
            return result.final_output


if __name__ == "__main__":
    chatBot = ChatBot()
    app = gr.ChatInterface(
        chatBot.chat,
        type="messages",
        title="Hello, I'm Rajeswaran Dhandapani",
        examples=["Do you hold any certifications?", "What is your background?", "What are your skills?", "What are your hobbies?", "What is your experience?", "What are your career goals?", "What are your projects?"],
        theme='soft'
    )
    app.launch()
