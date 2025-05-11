import customtkinter as ctk
import threading
import time
from PIL import Image
from Sl_Travel_Bot_Chat import get_response, bot_name
from database import ChatBotDB

db = ChatBotDB()

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class ChatApplication:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Sri Lanka Tourism Chatbot")
        self.window.geometry("360x600")  # Slightly larger window
        self.window.resizable(False, False)

        # Load icons using CTkImage
        try:
            self.user_icon = ctk.CTkImage(
                Image.open(r"C:\Users\REDTECH\Desktop\ToursimChat\User.ico"), 
                size=(30, 30)
            )
            self.bot_icon = ctk.CTkImage(
                Image.open(r"C:\Users\REDTECH\Desktop\ToursimChat\ChatBot.ico"), 
                size=(30, 30)
            )
        except Exception as e:
            print(f"Error loading icons: {e}")
            self.user_icon = None
            self.bot_icon = None

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

            # Create a frame for the chat display
            self.chat_frame = ctk.CTkScrollableFrame(
                self.window,
                width=380,  # Wider chat area
                height=500,
                fg_color="#2b2b2b"
            )
            self.chat_frame.pack(pady=(0, 10), padx=10)

            # Input frame with improved layout
            entry_frame = ctk.CTkFrame(self.window, fg_color="transparent", height=60)
            entry_frame.pack(pady=(0, 10), fill="x", padx=10)

            # Message entry field - larger and better proportioned
            self.msg_entry = ctk.CTkEntry(
                entry_frame,
                width=250,  # Wider entry field
                height=35,  # Taller entry field
                placeholder_text="Type your message...",
                font=ctk.CTkFont(family="Helvetica", size=14),
                corner_radius=15
            )
            self.msg_entry.pack(side="left", expand=True, fill="x", padx=(0, 10))

            # Send button - larger and better proportioned
            self.send_button = ctk.CTkButton(
                entry_frame,
                text="Send",
                command=self._on_enter_pressed,
                width=80,
                height=35,  # Matches entry field height
                font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
                fg_color="#1f6aa5",
                hover_color="#144870",
                corner_radius=15
            )
            self.send_button.pack(side="right")
        except Exception as e:
            self._display_error(f"Error setting up main window: {e}")

    def _on_enter_pressed(self, event=None):
        msg = self.msg_entry.get().strip()
        if not msg or not any(c.isalnum() for c in msg):
            self._display_error("Please enter a valid message.")
            return
        if len(msg) > 500:
            self._display_error("Message is too long. Please keep it under 500 characters.")
            return
        self.msg_entry.delete(0, "end")
        threading.Thread(target=self._process_message, args=(msg,)).start()

    def _process_message(self, msg):
        try:
            self._insert_message(msg, "You")
            
            if self.learning_mode:
                try:
                    db.insert_response(self.last_question, msg)
                    self._insert_message("Thank you! I've learned this new information.", bot_name)
                    self.learning_mode = False
                except (ValueError, RuntimeError) as e:
                    self._insert_message(f"Error saving response: {e}", bot_name)
            else:
                response = get_response(msg)
                self._insert_message(response, bot_name)
                
                if response.lower() == "let me know?" and not self.learning_mode:
                    self.learning_mode = True
                    self.last_question = msg
                elif response.startswith("Model retrained"):
                    self._insert_message("I've updated my knowledge base with new data!", bot_name)

        except Exception as e:
            self._display_error(f"Error processing message: {e}")

    def _insert_message(self, msg, sender):
        try:
            message_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
            
            if sender == "You":
                message_frame.pack(anchor="e", padx=5, pady=5, fill="x")
                
                text_label = ctk.CTkLabel(
                    message_frame,
                    text=msg,
                    font=ctk.CTkFont(family="Helvetica", size=14),
                    fg_color="#1f6aa5",
                    corner_radius=10,
                    padx=10,
                    pady=5,
                    justify="left",
                    wraplength=250
                )
                text_label.pack(side="right", padx=(5, 0))
                
                if self.user_icon:
                    icon_label = ctk.CTkLabel(
                        message_frame,
                        image=self.user_icon,
                        text="",
                        width=30,
                        height=30
                    )
                    icon_label.pack(side="right")
            else:
                message_frame.pack(anchor="w", padx=5, pady=5, fill="x")
                
                if self.bot_icon:
                    icon_label = ctk.CTkLabel(
                        message_frame,
                        image=self.bot_icon,
                        text="",
                        width=30,
                        height=30
                    )
                    icon_label.pack(side="left")
                
                text_label = ctk.CTkLabel(
                    message_frame,
                    text=msg,
                    font=ctk.CTkFont(family="Helvetica", size=14,slant="italic", weight="bold"),
                    fg_color="#2d862d",
                    corner_radius=10,
                    padx=10,
                    pady=5,
                    justify="left",
                    wraplength=250
                )
                text_label.pack(side="left", padx=(5, 0))

            self.chat_frame._parent_canvas.yview_moveto(1.0)
        except Exception as e:
            print(f"Error inserting message: {e}")

    def _display_error(self, message):
        try:
            error_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
            error_frame.pack(fill="x", padx=5, pady=5)
            
            error_label = ctk.CTkLabel(
                error_frame,
                text=message,
                font=ctk.CTkFont(family="Helvetica", size=14),
                fg_color="#cc0000",
                corner_radius=10,
                padx=10,
                pady=5,
                justify="left",
                wraplength=300
            )
            error_label.pack(fill="x")
            
            self.chat_frame._parent_canvas.yview_moveto(1.0)
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
