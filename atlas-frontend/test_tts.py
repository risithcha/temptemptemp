import sys
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from tts_engine import TTSEngine


def main():
    app = QApplication(sys.argv)
    
    tts = TTSEngine()
    
    print("TTS Engine initialized")
    print()
    
    tts.say("Hello world")
    
    counter = [0]
    
    def print_counter():
        counter[0] += 1
        print(f"Counter: {counter[0]}")
        
        if counter[0] == 3:
            print("\nInterrupting with new message...")
            tts.say("This should interput the previous message and start speaking this one.")
        
        if counter[0] >= 10:
            print("\nShutting down TTS engine")
            tts.shutdown()
            app.quit()
    
    timer = QTimer()
    timer.timeout.connect(print_counter)
    timer.start(1000)
    
    print("Starting event loop...")
    print()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
