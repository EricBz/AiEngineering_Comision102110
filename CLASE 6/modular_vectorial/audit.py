from datetime import datetime

LOG_FILE = "storage/audit.log"

def log_transition(state_name):
    with open(LOG_FILE, "a", encoding="utf8") as file:
        file.write(
            f"{datetime.now()} -> Estado: {state_name}\n"
        )