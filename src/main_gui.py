import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import datetime
import os
import time
import sys

# ensure src is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.crime_detector import CrimeDetectionPipeline


class Layout:
    def __init__(self, window, main_window):
        self.window = window
        self.main_window = main_window
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Create control frame
        self.control_frame = tk.Frame(self.window)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Start/Stop Detection Button
        self.b1 = tk.Button(
            self.control_frame,
            activebackground='black',
            activeforeground='red',
            bg='red',
            fg='white',
            text="Start Detection",
            height=max(1, int(screen_height / 80)),
            width=max(10, int(screen_width / 20)),
            border=3,
            command=self.toggle_detection,
        )

        # Report False Positive Button
        self.b2 = tk.Button(
            self.control_frame,
            activebackground='black',
            activeforeground='yellow',
            bg='orange',
            fg='white',
            text="Report False Alert",
            height=max(1, int(screen_height / 80)),
            width=max(10, int(screen_width / 20)),
            border=3,
            command=self.report_false_positive,
            state=tk.DISABLED,
        )

        # Save Current Clip Button
        self.b3 = tk.Button(
            self.control_frame,
            text="Save Clip",
            activebackground='black',
            activeforeground='green',
            bg='blue',
            fg='white',
            height=max(1, int(screen_height / 80)),
            width=max(10, int(screen_width / 20)),
            border=3,
            command=self.save_clip,
        )

        # Pack buttons
        self.b1.pack(padx=5, pady=5, side=tk.LEFT)
        self.b2.pack(padx=5, pady=5, side=tk.LEFT)
        self.b3.pack(padx=5, pady=5, side=tk.LEFT)

        # Status label
        self.status_label = tk.Label(
            self.control_frame,
            text="Status: Camera Ready",
            fg='green',
            font=('Arial', 12, 'bold'),
        )
        self.status_label.pack(padx=20, pady=5, side=tk.LEFT)

    def toggle_detection(self):
        """Start or stop crime detection"""
        if self.main_window.pipeline.is_detecting():
            self.main_window.pipeline.stop_detection()
            self.b1.config(text="Start Detection", bg='red')
            self.status_label.config(text="Status: Detection Stopped", fg='red')
            self.b2.config(state=tk.DISABLED)
        else:
            self.main_window.pipeline.start_detection()
            self.b1.config(text="Stop Detection", bg='green')
            self.status_label.config(text="Status: Detection Active", fg='green')

    def report_false_positive(self):
        """Report false positive detection"""
        response = messagebox.askyesno(
            "Report False Positive",
            "Was the last detection a false alarm?\n\nThis helps improve the model accuracy.",
        )
        if response:
            messagebox.showinfo("Reported", "Thank you! False positive reported.")
            # In a real system, this would log the feedback for retraining
            print(f"False positive reported at {datetime.datetime.now()}")
        self.b2.config(state=tk.DISABLED)

    def save_clip(self):
        """Save current video clip"""
        # Use main_window.recent_frames and actual frame size
        if hasattr(self.main_window, 'recent_frames') and self.main_window.recent_frames:
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                clips_dir = "saved_clips"
                if not os.path.exists(clips_dir):
                    os.makedirs(clips_dir)

                clip_path = os.path.join(clips_dir, f"clip_{timestamp}.avi")

                # Determine actual frame size from recent frames
                first_frame = self.main_window.recent_frames[0]
                h, w = first_frame.shape[:2]

                # Save last N frames
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(clip_path, fourcc, 20.0, (w, h))

                for frame in self.main_window.recent_frames:
                    # make sure frame size matches
                    if frame.shape[1] != w or frame.shape[0] != h:
                        frame = cv2.resize(frame, (w, h))
                    out.write(frame)
                out.release()

                messagebox.showinfo("Saved", f"Clip saved as:\n{clip_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save clip:\n{str(e)}")
        else:
            messagebox.showwarning("No Data", "No recent footage to save.")


class MainWindow:
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure reasonable dimensions
        if self.width <= 0 or self.height <= 0:
            self.width, self.height = 640, 480

        self.interval = 30  # ms

        # Initialize crime detection pipeline
        self.pipeline = CrimeDetectionPipeline()

        # Create main display frame
        self.display_frame = tk.Frame(self.window)
        self.display_frame.pack(side=tk.TOP, padx=5, pady=5)

        # Video canvas
        self.canvas = tk.Canvas(
            self.display_frame,
            width=self.width,
            height=self.height,
            bg='black',
        )
        self.canvas.pack(side=tk.LEFT, padx=5)

        # Info panel
        self.info_panel = tk.Frame(self.display_frame, width=300, height=self.height)
        self.info_panel.pack(side=tk.RIGHT, padx=5, fill=tk.Y)
        self.info_panel.pack_propagate(False)

        # Detection status
        self.detection_status = tk.Label(
            self.info_panel,
            text="CRIME DETECTION\nSTATUS",
            font=('Arial', 14, 'bold'),
            fg='blue',
        )
        self.detection_status.pack(pady=10)

        # Current detection info
        self.current_detection = tk.Label(
            self.info_panel,
            text="No crimes detected",
            font=('Arial', 11),
            fg='green',
            wraplength=280,
            justify=tk.LEFT,
        )
        self.current_detection.pack(pady=5)

        # Detection history
        self.history_label = tk.Label(
            self.info_panel,
            text="Recent Detections:",
            font=('Arial', 10, 'bold'),
        )
        self.history_label.pack(pady=(20, 5))

        self.detection_history = tk.Text(self.info_panel, height=15, width=35, font=('Arial', 9))
        self.detection_history.pack(pady=5, fill=tk.BOTH, expand=True)

        # Motion indicator
        self.motion_indicator = tk.Label(self.info_panel, text="Motion: None", font=('Arial', 10), fg='gray')
        self.motion_indicator.pack(pady=5)

        # Buffer for saving clips
        self.recent_frames = []
        self.max_frames_buffer = 200  # ~10 seconds at 20fps

        # Start video processing
        self.update_image()

    def update_image(self):
        # Read frame safely from capture
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # If no frame is returned, schedule another attempt
            self.window.after(self.interval, self.update_image)
            return

        try:
            # Resize frame for consistent processing (don't change aspect ratio forcibly)
            frame_resized = cv2.resize(frame, (self.width, self.height))

            # Add to recent frames buffer for clip saving
            self.recent_frames.append(frame_resized.copy())
            if len(self.recent_frames) > self.max_frames_buffer:
                self.recent_frames.pop(0)

            # Process frame through crime detection pipeline
            crime_detected, crime_type, confidence, motion_areas = self.pipeline.process_frame(frame_resized)

            # Create display frame with overlays
            display_frame = frame_resized.copy()

            # Draw motion detection boxes
            if motion_areas:
                self.motion_indicator.config(text=f"Motion: {len(motion_areas)} areas", fg='orange')
                for (x, y, w, h) in motion_areas:
                    # ensure rectangle fits in display_frame dimensions
                    x2 = min(x + w, display_frame.shape[1] - 1)
                    y2 = min(y + h, display_frame.shape[0] - 1)
                    cv2.rectangle(display_frame, (x, y), (x2, y2), (0, 255, 0), 2)
            else:
                self.motion_indicator.config(text="Motion: None", fg='gray')

            # Handle crime detection
            if crime_detected and confidence > 0.6:  # Only high-confidence detections
                self._handle_crime_detection(crime_type, confidence, display_frame)
            else:
                self.current_detection.config(text="No crimes detected", fg='green')

            # Add timestamp to frame
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add detection status to frame
            status_text = "DETECTING" if self.pipeline.is_detecting() else "MONITORING"
            status_color = (0, 255, 0) if self.pipeline.is_detecting() else (255, 255, 0)
            cv2.putText(display_frame, status_text, (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Convert to RGB and display
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(display_rgb)
            self.image = ImageTk.PhotoImage(self.image)

            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        except Exception as e:
            print(f"Error in update_image: {e}")

        # Schedule next update
        self.window.after(self.interval, self.update_image)

    def _handle_crime_detection(self, crime_type, confidence, frame):
        """Handle detected crime - update UI and log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Update current detection display
        detection_text = f"⚠️ CRIME DETECTED!\n\nType: {crime_type}\nConfidence: {confidence:.2%}\nTime: {timestamp}"
        self.current_detection.config(text=detection_text, fg='red', font=('Arial', 11, 'bold'))

        # Add to history
        history_entry = f"[{timestamp}] {crime_type} ({confidence:.1%})\n"
        self.detection_history.insert(tk.END, history_entry)
        self.detection_history.see(tk.END)

        # Draw crime detection overlay on frame
        try:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, f"CRIME ALERT: {crime_type.upper()}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        except Exception:
            pass

        # Enable false positive reporting
        if hasattr(self, 'layout'):
            self.layout.b2.config(state=tk.NORMAL)

        # Flash the window (optional alert)
        try:
            self.window.bell()  # System beep
        except Exception:
            pass

        # Log detection
        print(f"CRIME DETECTED: {crime_type} at {timestamp} with confidence {confidence:.2%}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Real-Time Crime Detection and Classification System")
    root.geometry("1200x800")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror(
            "Camera Error",
            "Cannot access camera!\n\nPlease check:\n1. Camera is connected\n2. No other apps are using it\n3. Camera permissions are granted",
        )
        root.destroy()
        exit(1)

    try:
        # Create main window and layout
        main_window = MainWindow(root, cap)
        layout = Layout(root, main_window)

        # Store reference for cross-component communication
        main_window.layout = layout

        # Show startup message
        messagebox.showinfo(
            "System Ready",
            "Real-Time Crime Detection System initialized!\n\n"
            "• Click 'Start Detection' to begin monitoring\n"
            "• Motion will be tracked automatically\n"
            "• High-confidence crime detections will trigger alerts\n"
            "• Use 'Save Clip' to save recent footage",
        )

        # Start the application
        root.mainloop()

    except Exception as e:
        messagebox.showerror("System Error", f"Failed to initialize system:\n{str(e)}")
    finally:
        # Cleanup
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
