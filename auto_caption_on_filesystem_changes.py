#!/home/offipso/Programs/venvs/redpill_explorer_2/bin/python
import os
import sys
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

CAPTION_SCRIPT = "/home/offipso/git/redpill_explorer_2/caption_images_with_cogvlm2.py"
PIDFILE = "/tmp/caption_images.pid"

class NewFileHandler(FileSystemEventHandler):
    """Handles events for new files created in the monitored directory."""

    def on_created(self, event):
        # Triggered when a new file or directory is created
        if event.is_directory:
            return

        # If a new image file is created, start the caption script
        if any(event.src_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            print(f"New image file detected: {event.src_path}")
            if not is_caption_script_running():
                run_caption_script(event.src_path)
            else:
                print("Caption script is already running. Waiting for it to finish...")

def is_caption_script_running():
    """Checks if the caption script is already running by looking for the PID file."""
    return os.path.exists(PIDFILE)

def run_caption_script(file_path):
    """Run the caption script with the new file as an argument."""
    print(f"Starting caption script for file: {file_path}")
    # Run the caption script with the new file as an argument
    with open(PIDFILE, 'w') as f:
        process = subprocess.Popen([sys.executable, CAPTION_SCRIPT, file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        f.write(str(process.pid))

    # Wait for the caption script to complete
    process.wait()
    print(f"Caption script completed for file: {file_path}")

    # Remove the PID file after completion
    if os.path.exists(PIDFILE):
        os.remove(PIDFILE)

def monitor_directory(directory):
    """Monitors the specified directory for new files and triggers the caption script."""
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=directory, recursive=True)
    observer.start()
    print(f"Monitoring directory: {directory}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_caption_on_filesystem_changes.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    monitor_directory(directory)
