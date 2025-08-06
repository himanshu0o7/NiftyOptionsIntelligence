import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Optional: For YouTube transcript (uses 3rd-party unofficial)
# pip install youtube-transcript-api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except:
    YouTubeTranscriptApi = None

LOG_FILE = "logs/self_learning_log.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110 Safari/537.36"
}


def fetch_page_summary(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string if soup.title else "No title"
        paras = soup.find_all("p")
        content = " ".join(p.get_text() for p in paras[:10])
        return title, content[:1000]  # Return max 1000 chars
    except Exception as e:
        return "Error", f"Failed to fetch: {e}"


def fetch_youtube_transcript(youtube_url):
    if YouTubeTranscriptApi is None:
        return "YouTube Transcript API not installed", ""
    try:
        video_id = urlparse(youtube_url).query.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return "YouTube Video", text[:1000]
    except Exception as e:
        return "Transcript Error", str(e)


def log_learning(topic, url, summary):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "results": [
            {
                "url": url,
                "summary": summary
            }
        ]
    }
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


def learn_from_url(topic, url):
    if "youtube.com" in url:
        title, summary = fetch_youtube_transcript(url)
    else:
        title, summary = fetch_page_summary(url)
    print(f"📚 Learning from: {title}\n➡️ {url}\n🧠 {summary[:300]}...\n")
    log_learning(topic, url, summary)


# EXAMPLES (Can be extended to cronjob, Telegram-trigger, or codex-agent)
if __name__ == "__main__":
    learn_from_url("BankNifty breakout news", "https://www.trendlyne.com/markets-today/")
    learn_from_url("BankNifty technical update", "https://economictimes.indiatimes.com/markets")
    learn_from_url("Nifty Options strategy", "https://www.youtube.com/watch?v=WhRE6nO1UpM")
