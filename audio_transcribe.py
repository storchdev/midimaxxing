import base64
import subprocess
import sys
import threading
from pathlib import Path

import requests

# --- CONFIGURATION ---
INPUT_FILE = sys.argv[1]
_stem = Path(INPUT_FILE).stem
OUTPUT_FILENAME = sys.argv[2] if len(sys.argv) > 2 else f"midi/{_stem}_audio.mid"
CONTAINER_NAME = "piano-worker"  # Ensure this matches your --name
API_URL = "http://localhost:5000/predictions"
TRIGGER_PHRASE = "Write out to transcription.mid"

Path(OUTPUT_FILENAME).parent.mkdir(parents=True, exist_ok=True)

# 1. Prepare Audio
print(f"Loading {INPUT_FILE}...")
with open(INPUT_FILE, "rb") as file:
    encoded_string = base64.b64encode(file.read()).decode("utf-8")
    audio_uri = f"data:audio/mp3;base64,{encoded_string}"


# 2. Define the Request Function (background thread)
def send_request():
    try:
        # We assume the server is ready. If it's cold, this might timeout,
        # but the log watcher will just wait.
        requests.post(API_URL, json={"input": {"audio_input": audio_uri}})
    except:
        pass


# 3. Start Request
req_thread = threading.Thread(target=send_request)
req_thread.start()

# 4. Watch Docker Logs
print("--> Watching logs...")

# --since 0s ensures we don't read old logs from previous runs
process = subprocess.Popen(
    ["docker", "logs", "-f", "--since", "0s", CONTAINER_NAME],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

found = False

try:
    for line in process.stdout:
        clean_line = line.strip()

        # \r moves cursor to start, \033[K clears the rest of the line
        # We truncate to 100 chars to prevent messy wrapping on small terminals
        display_line = (
            (clean_line[:100] + "..") if len(clean_line) > 100 else clean_line
        )
        sys.stdout.write(f"\r[Container]: {display_line}\033[K")
        sys.stdout.flush()

        if TRIGGER_PHRASE in clean_line:
            # Print newline so the "Success" message doesn't overwrite the last log
            print("\n")
            print(f"✅ Trigger detected: '{TRIGGER_PHRASE}'")
            print("--> Extracting MIDI file...")

            # 5. Copy file out
            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{CONTAINER_NAME}:/src/transcription.mid",
                    OUTPUT_FILENAME,
                ],
                check=True,
            )
            print(f"✅ Saved to: {OUTPUT_FILENAME}")
            found = True
            break

    # 6. Stop video generation
    if found:
        print("--> Stopping video generation (restarting container)...")
        subprocess.run(["docker", "restart", CONTAINER_NAME], stdout=subprocess.DEVNULL)
        print("✅ Container reset. Done.")

except KeyboardInterrupt:
    print("\nStopping...")

# Clean up
process.terminate()
sys.exit(0 if found else 1)
