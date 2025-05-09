import customtkinter as ctk
import threading
import time
from Sl_Travel_Bot_Chat import get_response, bot_name
from database import ChatBotDB

db = ChatBotDB()

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class ChatApplication:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Sri Lanka Tourism Chatbot")
        self.window.geometry("360x600")
        self.window.resizable(False, False)

        self.learning_mode = False
        self.last_question = ""
        self._setup_loading_screen()
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_loading_screen(self):
        self.loading_frame = ctk.CTkFrame(self.window)
        self.loading_frame.pack(fill="both", expand=True)

        self.loading_label = ctk.CTkLabel(
            self.loading_frame, 
            text="Initializing... Please wait.", 
            font=ctk.CTkFont(family="Helvetica", size=20, weight="bold")
        )
        self.loading_label.pack(pady=20)

        self.progress = ctk.CTkProgressBar(
            self.loading_frame, 
            orientation="horizontal", 
            mode="indeterminate", 
            width=300
        )
        self.progress.pack(pady=20)
        self.progress.start()

        threading.Thread(target=self._initialize_chat, daemon=True).start()

    def _initialize_chat(self):
        steps = ["Setting up database...", "Loading chatbot model...", "Preparing chat interface..."]
        for step in steps:
            time.sleep(2)
            self._update_loading_message(step)
        self._setup_main_window()

    def _update_loading_message(self, message):
        try:
            self.loading_label.configure(text=message)
        except Exception as e:
            print(f"Error updating loading message: {e}")

    def _setup_main_window(self):
        try:
            self.loading_frame.pack_forget()

            self.title_label = ctk.CTkLabel(
                self.window, 
                text="Sri Lanka Tourism Bot", 
                font=ctk.CTkFont(family="Helvetica", size=24, weight="bold")
            )
            self.title_label.pack(pady=10)

            # Main chat display with modern styling
            self.text_display = ctk.CTkTextbox(
                self.window, 
                width=340, 
                height=500, 
                corner_radius=10,
                fg_color="#2b2b2b",
                font=ctk.CTkFont(family="Helvetica", size=14),
                wrap="word",
                spacing3=8  # Add spacing between paragraphs
            )
            self.text_display.pack(pady=(0, 10))
            self.text_display.configure(state="disabled")

            # Input frame
            entry_frame = ctk.CTkFrame(self.window, fg_color="transparent")
            entry_frame.pack(pady=10, fill="x", padx=20)

            self.msg_entry = ctk.CTkEntry(
                entry_frame, 
                width=240, 
                height=40, 
                placeholder_text="Type your message...",
                font=ctk.CTkFont(family="Helvetica", size=14),
                corner_radius=20
            )
            self.msg_entry.pack(side="left", padx=(0, 10), expand=True, fill="x")
            self.msg_entry.bind("<Return>", self._on_enter_pressed)

            self.send_button = ctk.CTkButton(
                entry_frame, 
                text="Send", 
                command=self._on_enter_pressed, 
                width=100,
                height=55,
                # corner_radius=1,
                font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
                fg_color="#1f6aa5",
                hover_color="#144870"
            )
            self.send_button.pack(side="right")
        except Exception as e:
            self._display_error(f"Error setting up main window: {e}")

    def _on_enter_pressed(self, event=None):
        msg = self.msg_entry.get().strip().lower()
        if not msg or not any(c.isalnum() for c in msg):
            self._display_error("Please enter a valid message.")
            return
        if len(msg) > 500:
            self._display_error("Message is too long. Please keep it under 500 characters.")
            return
        self.msg_entry.delete(0, "end")
        threading.Thread(target=self._insert_message, args=(msg, "You")).start()

    def _insert_message(self, msg, sender):
        try:
            self.text_display.configure(state="normal")
            
            # Add spacing between messages
            self.text_display.insert("end", "\n")
            
            # Different styling for user and bot messages
            if sender == "You":
                # User message bubble (right-aligned)
                self.text_display.tag_config("user", 
                    foreground="#ffffff", 
                    background="#1f6aa5", 
                    lmargin1=100, 
                    lmargin2=100, 
                    rmargin=20,
                    relief="raised", 
                    borderwidth=2,
                    spacing2=5,
                    wrap="word"
                )
                self.text_display.insert("end", f"{sender}: {msg}\n", "user")
            else:
                # Bot message bubble (left-aligned)
                self.text_display.tag_config("bot", 
                    foreground="#ffffff", 
                    background="#2d862d", 
                    lmargin1=20, 
                    lmargin2=20, 
                    rmargin=100,
                    relief="raised", 
                    borderwidth=2,
                    spacing2=5,
                    wrap="word"
                )
                self.text_display.insert("end", f"{bot_name}: {msg}\n", "bot")

            if self.learning_mode:
                try:
                    db.insert_response(self.last_question, msg)
                    self.text_display.insert("end", f"\n{bot_name}: Thank you! I've learned this new information about travel.\n", "bot")
                    self.learning_mode = False
                except (ValueError, RuntimeError) as e:
                    self.text_display.insert("end", f"\n{bot_name}: Error saving response: {e}\n", "bot")
            else:
                response = get_response(msg)
                self.text_display.insert("end", f"\n{bot_name}: {response}\n", "bot")
                if response == "let me know?":
                    self.learning_mode = True
                    self.last_question = msg
                elif response.startswith("Model retrained"):
                    self.text_display.insert("end", f"\n{bot_name}: I've updated my knowledge base with new data!\n", "bot")

            # Add separator with less visual weight
            self.text_display.insert("end", "\n", "separator")
            self.text_display.tag_config("separator", 
                foreground="#555555", 
                spacing1=5, 
                spacing3=5
            )
            
            self.text_display.configure(state="disabled")
            self.text_display.see("end")
        except Exception as e:
            self._display_error(f"Error processing message: {e}")

    def _display_error(self, message):
        try:
            self.text_display.configure(state="normal")
            self.text_display.tag_config("error", 
                foreground="#ffffff", 
                background="#cc0000", 
                lmargin1=20, 
                lmargin2=20, 
                rmargin=100,
                spacing2=5
            )
            self.text_display.insert("end", f"\n{bot_name}: {message}\n", "error")
            self.text_display.insert("end", "\n", "separator")
            self.text_display.configure(state="disabled")
            self.text_display.see("end")
        except Exception as e:
            print(f"Error displaying error message: {e}")

    def _on_closing(self):
        try:
            db.close_connection()
            self.window.destroy()
        except Exception as e:
            print(f"Error during shutdown: {e}")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    try:
        app = ChatApplication()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")