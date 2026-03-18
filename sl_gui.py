import tkinter as tk
from tkinter import Frame, Label, Listbox, Button
import threading
import cv2
import time
from collections import deque
from sign_language_detector import SignLanguageDetector
from tts_engine import TextToSpeechEngine


class SignLanguageGUI:
    """Main application class for sign language detection"""

    def __init__(self, root):
        self.root = root
        self.is_running = False
        self.cap = None
        self.hand_detector = SignLanguageDetector()
        self.tts_engine = TextToSpeechEngine()
        self.hand_sign_queue = deque()
        self.temp_signage = None
        self.last_signage = None
        self.delay = 0.6 #to delay before detection
        self.detected_signs_label = None
        self.tts_listbox = None

        self.setup_gui()

    def setup_gui(self):
        """Setup the modern GUI components"""
        self.root.title("Sign Language Detector")
        self.root.geometry("900x600")
        self.root.configure(bg="#1E1E2F")  # Dark background, yay! :D

        # Bind keyboard shortcuts
        self.root.bind('<space>', self.on_space_pressed)
        self.root.bind('s', self.on_s_pressed)

        # Top Frame for title
        top_frame = tk.Frame(self.root, bg="#1E1E2F")
        top_frame.pack(pady=20)

        title = tk.Label(
            top_frame,
            text="Sign Language Detector",
            font=("Helvetica", 26, "bold"),
            fg="#FFFFFF",
            bg="#1E1E2F"
        )
        title.pack(pady=(0, 10))

        subtitle = tk.Label(
            top_frame,
            text="Select your preferred Text-to-Speech model",
            font=("Helvetica", 14),
            fg="#CCCCCC",
            bg="#1E1E2F"
        )
        subtitle.pack()

        shortcuts = tk.Label(
            top_frame,
            text="Keyboard Shortcuts: [SPACE] to Start | [S] to Stop",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#1E1E2F"
        )
        shortcuts.pack(pady=(5, 0))

        # Center Frame for detection and TTS selection
        center_frame = tk.Frame(self.root, bg="#2E2E3E", padx=20, pady=20)
        center_frame.pack(pady=20, fill="both", expand=True)

        self.detected_signs_label = tk.Label(
            center_frame,
            text="Detected Signs",
            font=("Helvetica", 16, "bold"),
            fg="#00FF00",
            bg="#2E2E3E"
        )
        self.detected_signs_label.pack(pady=15)

        # TTS selection listbox
        self.tts_listbox = tk.Listbox(
            center_frame,
            width=40,
            height=4,
            font=("Helvetica", 12),
            bg="#1E1E2F",
            fg="#FFFFFF",
            selectbackground="#0080FF",
            selectforeground="#FFFFFF",
            highlightthickness=0,
            borderwidth=0
        )
        self.tts_listbox.insert(1, "Google Text-to-Speech (ONLINE)")
        self.tts_listbox.insert(2, "Python Text-to-Speech (OFFLINE)")
        self.tts_listbox.pack(pady=10)

        # Bottom Frame for buttons
        button_frame = tk.Frame(center_frame, bg="#2E2E3E")
        button_frame.pack(pady=20)

        # Function to create a hover effect
        def on_enter(e, button, color):
            button['bg'] = color

        def on_leave(e, button, color):
            button['bg'] = color

        # Start button
        start_detection = tk.Button(
            button_frame,
            text="Start Detection",
            font=("Helvetica", 14, "bold"),
            bg="#008000",
            fg="#FFFFFF",
            padx=20,
            pady=10,
            borderwidth=0,
            activebackground="#00AA00",
            command=self.start_detection_command
        )
        start_detection.pack(side="left", padx=20)

        start_detection.bind("<Enter>", lambda e: on_enter(e, start_detection, "#00AA00"))
        start_detection.bind("<Leave>", lambda e: on_leave(e, start_detection, "#008000"))

        # Stop button
        stop_detection = tk.Button(
            button_frame,
            text="Stop Detection",
            font=("Helvetica", 14, "bold"),
            bg="#D9534F",
            fg="#FFFFFF",
            padx=20,
            pady=10,
            borderwidth=0,
            activebackground="#FF4C4C",
            command=self.stop_detection_command
        )
        stop_detection.pack(side="left", padx=20)

        stop_detection.bind("<Enter>", lambda e: on_enter(e, stop_detection, "#FF4C4C"))
        stop_detection.bind("<Leave>", lambda e: on_leave(e, stop_detection, "#D9534F"))

    def start_detection_command(self):
        """Start detection in a separate thread"""
        start_detection_thread = threading.Thread(target=self.sign_language_detector, daemon=True)
        start_detection_thread.start()

    def stop_detection_command(self):
        """Stop detection"""
        self.is_running = False
        print("Stopping sign language detection...")

    def update_detected_signs_label(self, text):
        """Update the detected signs label"""
        self.detected_signs_label.config(text=text)

    def sign_language_detector(self):
        """Main sign language detection loop"""
        self.is_running = True
        self.hand_sign_queue = deque()
        self.temp_signage = None
        self.last_signage = None

        self.cap = cv2.VideoCapture(0)
        print("Detection started. Press 's' or  Stop to exit")

        while self.is_running:
            success, img = self.cap.read()
            if not success:
                break

            img_output = img.copy()
            detected_sign, img, img_crop, img_white, bbox = self.hand_detector.detect_hand_sign(img)

            if detected_sign is not None:
                current_time = time.time()

                if self.temp_signage is None:
                    self.temp_signage = detected_sign
                    self.last_signage = current_time
                elif self.temp_signage == detected_sign:
                    if current_time - self.last_signage > self.delay:
                        self.hand_sign_queue.append(detected_sign)
                        print(f"Detected sign: {detected_sign}")
                        self.root.after(0, lambda sign=detected_sign:
                                       self.update_detected_signs_label(f"Detected Sign: {sign}"))


                        self.temp_signage = None
                else:
                    self.temp_signage = detected_sign
                    self.last_signage = current_time
                    print(f"Sign was changed to {detected_sign}")
                    self.root.after(0, lambda sign=detected_sign:
                                   self.update_detected_signs_label(f"Sign was changed to: {sign}"))

                self.hand_detector.draw_hand_bbox(img_output, bbox)

                cv2.imshow("ImageCrop", img_crop)
                cv2.imshow("ImageWhite", img_white)
                cv2.imshow("Image", img_output)
            else:
                self.temp_signage = None

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to close manually
                self.is_running = False

        self.cap.release()
        cv2.destroyAllWindows()

        # speak the final word
        try:
            selected_tts = self.tts_listbox.get(self.tts_listbox.curselection())
        except:
            selected_tts = "Python Text-to-Speech (OFFLINE)"

        if self.hand_sign_queue:
            final_word = ''.join(self.hand_sign_queue)
            print(f"\nFinal signed word: {final_word}")
            self.root.after(0, lambda word=final_word:
                           self.update_detected_signs_label(f"Final Word: {word}"))
            text_to_speech = final_word
        else:
            print("\nNo signs were captured.")
            text_to_speech = "No signs were captured"

        self.tts_engine.speak(text_to_speech, selected_tts)

    def on_space_pressed(self, event):
        """Start detection when spacebar is pressed"""
        self.start_detection_command()
        return "break"

    def on_s_pressed(self, event):
        """Stop detection when 's' key is pressed"""
        self.stop_detection_command()
        return "break"


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()