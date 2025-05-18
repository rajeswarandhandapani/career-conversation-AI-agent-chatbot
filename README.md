1. Visit https://huggingface.co and set up an account
2. From the Avatar menu on the top right, choose Access Tokens. Choose "Create New Token". Give it WRITE permissions.
3. Take this token and add it to your .env file: `HF_TOKEN=hf_xxx`
4. From the 1_foundations folder, enter: `uv run gradio deploy` and if for some reason this still wants you to enter your HF token, then interrupt it with ctrl+c and run this instead: `uv run dotenv -f ../.env run -- uv run gradio deploy` which forces your keys to all be set as environment variables
5. Follow its instructions: name it "career_conversation", specify app.py, choose cpu-basic as the hardware, say Yes to needing to supply secrets, provide your openai api key, your pushover user and token, and say "no" to github actions.  