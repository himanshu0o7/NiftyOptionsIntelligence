import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Upload files to assistant
def upload_files(file_list):
    file_ids = []
    for path in file_list:
        if os.path.exists(path):
            print(f"📤 Uploading {path}")
            file_obj = client.files.create(file=open(path, "rb"), purpose="assistants")
            file_ids.append(file_obj.id)
    return file_ids

# ✅ Initialize assistant with Code Interpreter
def create_kp5bot_assistant(file_ids):
    return client.beta.assistants.create(
        instructions="""
        You are KP5Bot's AI Dev Assistant.
        Your task: Fix code errors, write modular Python files, generate pytest test cases, run them, and export logs/plots/CSVs.
        Do not modify core Phase 1 logic. Always run validation after fixes.
        """,
        model="gpt-4o",
        tools=[{"type": "code_interpreter"}],
        tool_resources={"code_interpreter": {"file_ids": file_ids}},
    )

# ✅ Create a new thread and start a Run
def run_code_thread(assistant_id, user_prompt):
    thread = client.beta.threads.create(
        messages=[{
            "role": "user",
            "content": user_prompt
        }]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    return thread.id, run.id

# ✅ Monitor run until complete
def wait_for_completion(thread_id, run_id):
    print("⏳ Waiting for Codex run to complete...")
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status in ["completed", "failed", "cancelled"]:
            print(f"✅ Run status: {run.status}")
            return run.status
        time.sleep(2)

# ✅ MAIN EXECUTION
if __name__ == "__main__":
    files = ["bot_modules/greeks_handler.py", "data/sample_option_chain.csv"]
    file_ids = upload_files(files)

    assistant = create_kp5bot_assistant(file_ids)
    print(f"✅ Assistant Created: {assistant.id}")

    thread_id, run_id = run_code_thread(assistant.id, "Fix any issues in greeks_handler.py and generate test_greeks_handler.py using pytest.")
    print(f"🧵 Thread: {thread_id}\n🏃‍♂️ Run: {run_id}")

    wait_for_completion(thread_id, run_id)

