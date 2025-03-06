import os
import cv2
import numpy as np
import torch
import whisper
from moviepy.editor import VideoFileClip
import json
import re
import pytesseract
import difflib
from transformers import pipeline
from PIL import Image
import time
from sentence_transformers import SentenceTransformer, util

# Helper: Map an average RGB color to a basic color name
def approximate_color(rgb_tuple):
    # Define a small palette of basic colors (RGB)
    colors = {
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128)
    }
    r, g, b = rgb_tuple
    min_dist = float("inf")
    closest_color = None
    for color_name, (cr, cg, cb) in colors.items():
        dist = np.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    return closest_color

class UIElementDetector:
    """Detect UI elements in screen recordings using OCR and basic color analysis"""
    def __init__(self):
        # In a real implementation, you might load a trained model here
        pass

    def detect(self, frame):
        """Detect UI elements in a frame using contour detection, OCR, and color analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Threshold to extract potential UI elements (adjust threshold as needed)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ui_elements = []

        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Minimum size threshold
                continue
            x, y, w, h = cv2.boundingRect(contour)
            element_img = frame[y:y+h, x:x+w]
            # Use pytesseract to extract text from the UI element image
            ocr_result = pytesseract.image_to_string(element_img, config="--psm 7").strip()
            # Compute average color in BGR then convert to RGB
            avg_color = cv2.mean(element_img)[:3]
            avg_color = tuple(int(c) for c in avg_color[::-1])  # convert BGR -> RGB
            color_name = approximate_color(avg_color)
            # Classify the element based on aspect ratio (simple heuristic)
            element_type = self._classify_element(w, h)
            ui_elements.append({
                'type': element_type,
                'text': ocr_result if ocr_result else "N/A",
                'bbox': (x, y, w, h),
                'color': color_name
            })
        return ui_elements

    def _classify_element(self, width, height):
        """Classify UI element based on shape"""
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 5:
            return "Menu"
        elif 2 < aspect_ratio <= 5:
            return "Text field"
        elif 0.75 < aspect_ratio <= 2:
            return "Button"
        elif aspect_ratio <= 0.75:
            return "Icon"
        else:
            return "UI Element"


class CursorTracker:
    """Track cursor position and interactions using template matching (placeholder implementation)"""
    def __init__(self):
        # In a real implementation, you might load actual cursor templates
        pass

    def track(self, frame):
        """Simulate cursor detection"""
        import random
        h, w = frame.shape[:2]
        if random.random() < 0.7:  # 70% chance of detecting cursor
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            return {
                'detected': True,
                'position': (x, y),
                'confidence': random.uniform(0.7, 0.95)
            }
        else:
            return {
                'detected': False,
                'position': None,
                'confidence': 0
            }


class OnboardingVideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.temp_dir = "temp_processing"
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.audio_path = os.path.join(self.temp_dir, "audio.wav")

        # Create temp directories
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        # Load models
        print("Loading models...")
        self.transcription_model = whisper.load_model("medium")
        self.nlp_summarizer = pipeline("summarization")
        self.nlp_qa = pipeline("question-answering")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ui_detector = UIElementDetector()
        self.cursor_tracker = CursorTracker()
        print("Models loaded successfully")

    def process(self):
        print(f"Processing video: {self.video_path}")
        self.extract_audio()
        self.extract_frames()
        transcript = self.transcribe_audio()
        frames_data = self.analyze_frames()
        steps = self.segment_into_steps(transcript, frames_data)
        structured_output = self.generate_structured_output(steps)
        return structured_output

    def extract_audio(self):
        print(f"Extracting audio from {self.video_path}...")
        video = VideoFileClip(self.video_path)
        audio = video.audio
        audio.write_audiofile(self.audio_path)
        self.video_duration = video.duration
        self.fps = video.fps
        video.close()

    def extract_frames(self, frame_interval=5):
        print(f"Extracting frames from {self.video_path}...")
        cap = cv2.VideoCapture(self.video_path)
        self.frame_data = []
        frame_count = 0
        previous_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = frame_count / self.fps
                frame_path = os.path.join(self.frames_dir, f"frame_{frame_count:06d}_{timestamp:.2f}s.jpg")
                cv2.imwrite(frame_path, frame)
                self.frame_data.append({
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'path': frame_path,
                    'motion_detected': False
                })
                if previous_frame is not None:
                    motion = self.detect_motion(previous_frame, frame)
                    self.frame_data[-1]['motion_detected'] = motion > 0.002  # Threshold
                previous_frame = frame.copy()
            frame_count += 1
        cap.release()
        print(f"Extracted {len(self.frame_data)} frames")
        return self.frame_data

    def detect_motion(self, prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        motion_score = np.sum(frame_diff) / (prev_gray.shape[0] * prev_gray.shape[1] * 255)
        return motion_score

    def transcribe_audio(self):
        print(f"Transcribing audio from {self.audio_path}...")
        result = self.transcription_model.transcribe(
            self.audio_path,
            verbose=True,
            word_timestamps=True
        )
        self.transcript_segments = result["segments"]
        self.transcript_text = result["text"]
        print(f"Transcription complete: {len(self.transcript_segments)} segments")
        return self.transcript_segments

    def analyze_frames(self):
        print("Analyzing frames for UI elements and actions...")
        analyzed_frames = []
        for i, frame_data in enumerate(self.frame_data):
            print(f"Analyzing frame {i+1}/{len(self.frame_data)}")
            frame = cv2.imread(frame_data['path'])
            ui_elements = self.ui_detector.detect(frame)
            cursor_info = self.cursor_tracker.track(frame)
            click_detected = False
            clicked_element = None
            if cursor_info['detected'] and frame_data['motion_detected']:
                for ui_elem in ui_elements:
                    x, y, w, h = ui_elem['bbox']
                    if (x <= cursor_info['position'][0] <= x + w and
                        y <= cursor_info['position'][1] <= y + h):
                        click_detected = True
                        clicked_element = ui_elem
                        break
            analyzed_frame = {
                **frame_data,
                'ui_elements': ui_elements,
                'cursor': cursor_info,
                'click_detected': click_detected,
                'clicked_element': clicked_element
            }
            analyzed_frames.append(analyzed_frame)
        return analyzed_frames

    def segment_into_steps(self, transcript_segments, frames_data):
        print("Segmenting into logical steps...")
        potential_steps = []
        # 1. Speech markers
        step_markers = [
            "step", "first", "second", "third", "fourth", "fifth",
            "start by", "begin by", "next", "then", "after that",
            "now", "click on", "go to", "navigate to", "open", "select"
        ]
        for segment in transcript_segments:
            text = segment['text'].lower()
            for marker in step_markers:
                if marker in text:
                    potential_steps.append({
                        'type': 'speech_marker',
                        'timestamp': segment['start'],
                        'segment': segment
                    })
                    break
        # 2. UI interactions
        for frame in frames_data:
            if frame['click_detected'] and frame['clicked_element'] is not None:
                potential_steps.append({
                    'type': 'ui_interaction',
                    'timestamp': frame['timestamp'],
                    'frame': frame
                })
        # 3. Screen transitions based on changes in UI elements
        for i in range(1, len(frames_data)):
            prev_elements = len(frames_data[i-1]['ui_elements'])
            curr_elements = len(frames_data[i]['ui_elements'])
            if abs(prev_elements - curr_elements) > 3:
                potential_steps.append({
                    'type': 'screen_transition',
                    'timestamp': frames_data[i]['timestamp'],
                    'frame': frames_data[i]
                })
        potential_steps.sort(key=lambda x: x['timestamp'])
        # Merge steps that are too close in time (within 2 seconds)
        merged_steps = []
        for step in potential_steps:
            if not merged_steps or step['timestamp'] - merged_steps[-1]['timestamp'] > 2.0:
                merged_steps.append(step)
            else:
                if step['type'] == 'speech_marker' and merged_steps[-1]['type'] != 'speech_marker':
                    merged_steps[-1] = step
        # Create steps from boundaries
        steps = []
        for i in range(len(merged_steps)):
            start_time = merged_steps[i]['timestamp']
            end_time = merged_steps[i+1]['timestamp'] if i+1 < len(merged_steps) else self.video_duration
            step_segments = [seg for seg in transcript_segments if seg['start'] >= start_time and seg['start'] < end_time]
            step_frames = [frame for frame in frames_data if frame['timestamp'] >= start_time and frame['timestamp'] < end_time]
            step_info = self.extract_step_info(step_segments, step_frames, i+1)
            steps.append(step_info)
        return steps

    def extract_step_info(self, transcript_segments, frames, step_number):
        step_text = " ".join([segment['text'] for segment in transcript_segments])
        interacted_elements = []
        for frame in frames:
            if frame['click_detected'] and frame['clicked_element'] is not None:
                interacted_elements.append(frame['clicked_element'])
        key_ui_elements = self.extract_key_ui_elements(frames, step_text)
        inputs = self.extract_inputs(transcript_segments, frames)
        title = self.generate_step_title(step_text, interacted_elements, key_ui_elements)
        description = self.generate_step_description(step_text, interacted_elements, key_ui_elements, inputs)
        success_criteria = self.generate_success_criteria(interacted_elements, key_ui_elements, inputs, step_text)
        return {
            'step_number': step_number,
            'title': title,
            'description': description,
            'ui_elements': key_ui_elements,
            'inputs': inputs,
            'success_criteria': success_criteria,
            'transcript': step_text,
            'interacted_elements': interacted_elements
        }

    def extract_key_ui_elements(self, frames, step_text):
        all_elements = []
        for frame in frames:
            all_elements.extend(frame['ui_elements'])
        unique_elements = {}
        for elem in all_elements:
            if elem['text'] != "N/A" and elem['text'] not in unique_elements:
                unique_elements[elem['text']] = elem
        scored_elements = []
        for elem_text, elem in unique_elements.items():
            mentioned = elem_text.lower() in step_text.lower()
            interacted = any(frame['click_detected'] and frame['clicked_element'] and
                             frame['clicked_element']['text'] == elem_text
                             for frame in frames)
            score = (1 if mentioned else 0) + (2 if interacted else 0)
            scored_elements.append({
                'element': elem,
                'score': score
            })
        scored_elements.sort(key=lambda x: x['score'], reverse=True)
        top_elements = [item['element'] for item in scored_elements[:5]]
        return top_elements

    def extract_inputs(self, transcript_segments, frames):
        inputs = []
        input_patterns = [
            r'(?:type|enter|input|fill in|write)(?:\s+in)?(?:\s+the)?[:\s]+["\']?([^"\'.,;!?]+)["\']?',
            r'(?:typing|entering)(?:\s+in)?(?:\s+the)?[:\s]+["\']?([^"\'.,;!?]+)["\']?'
        ]
        transcript_text = " ".join([seg['text'] for seg in transcript_segments])
        for pattern in input_patterns:
            matches = re.finditer(pattern, transcript_text, re.IGNORECASE)
            for match in matches:
                inputs.append(match.group(1).strip())
        return inputs

    def correct_transcription(self, original_term, candidates):
        # Use fuzzy matching to see if any candidate is similar enough
        matches = difflib.get_close_matches(original_term, candidates, n=1, cutoff=0.8)
        return matches[0] if matches else original_term

    def generate_step_title(self, transcript, interacted_elements, key_ui_elements):
        action_patterns = [
            r'(?:click|tap|press)(?:\s+on)?(?:\s+the)?\s+([^\s.,;!?]+(?:\s+[^\s.,;!?]+){0,3})',
            r'(?:go|navigate)(?:\s+to)?(?:\s+the)?\s+([^\s.,;!?]+(?:\s+[^\s.,;!?]+){0,3})',
            r'(?:open|launch|start)(?:\s+up)?(?:\s+the)?\s+([^\s.,;!?]+(?:\s+[^\s.,;!?]+){0,3})',
            r'(?:select|choose)(?:\s+the)?\s+([^\s.,;!?]+(?:\s+[^\s.,;!?]+){0,3})',
            r'(?:enter|type|input)(?:\s+in)?(?:\s+the)?\s+([^\s.,;!?]+(?:\s+[^\s.,;!?]+){0,3})'
        ]
        action_types = {
            r'(?:click|tap|press)': "Click",
            r'(?:go|navigate)': "Navigate to",
            r'(?:open|launch|start)': "Open",
            r'(?:select|choose)': "Select",
            r'(?:enter|type|input)': "Enter"
        }
        for pattern in action_patterns:
            matches = re.search(pattern, transcript, re.IGNORECASE)
            if matches:
                action_object = matches.group(1)
                # If we have UI elements from OCR, try to correct transcription errors
                ocr_texts = [elem['text'] for elem in key_ui_elements if elem['text'] != "N/A"]
                corrected = self.correct_transcription(action_object, ocr_texts)
                # Find the corresponding action type
                for pattern_prefix, action_type in action_types.items():
                    if pattern.startswith(pattern_prefix):
                        return f"{action_type} {corrected}"
                return f"Interact with {corrected}"
        if interacted_elements:
            elem = interacted_elements[0]
            elem_type = elem['type'].lower()
            elem_text = elem['text']
            color_info = f" ({elem['color']})" if elem.get('color') else ""
            if elem_type == 'button':
                return f"Click on the button labeled '{elem_text}'{color_info}"
            elif elem_type == 'link':
                return f"Click on the link labeled '{elem_text}'{color_info}"
            elif elem_type in ['textbox', 'input', 'text field']:
                return f"Enter information in the '{elem_text}' field{color_info}"
            else:
                return f"Interact with the '{elem_text}' element{color_info}"
        if len(transcript) > 20:
            try:
                result = self.nlp_summarizer(transcript, max_length=8, min_length=3, do_sample=False)
                if result:
                    return result[0]['summary_text'].capitalize()
            except Exception as e:
                print(f"Error during summarization: {e}")
        return "Complete the step"

    def generate_step_description(self, transcript, interacted_elements, key_ui_elements, inputs):
        description = self.clean_transcript(transcript)
        if interacted_elements:
            interaction_details = []
            for elem in interacted_elements:
                elem_type = elem['type'].lower()
                elem_text = elem['text']
                color_info = f" ({elem['color']})" if elem.get('color') else ""
                if elem_type == 'button':
                    interaction_details.append(f"Click on the button labeled '{elem_text}'{color_info}.")
                elif elem_type == 'link':
                    interaction_details.append(f"Click on the link labeled '{elem_text}'{color_info}.")
                elif elem_type in ['textbox', 'input', 'text field']:
                    matching_input = None
                    for inp in inputs:
                        if (elem_text.lower() in inp.lower() or
                            any(word in elem_text.lower() for word in inp.lower().split())):
                            matching_input = inp
                            break
                    if matching_input:
                        interaction_details.append(f"Enter '{matching_input}' in the '{elem_text}' field{color_info}.")
                    else:
                        interaction_details.append(f"Enter information in the '{elem_text}' field{color_info}.")
                else:
                    interaction_details.append(f"Interact with the '{elem_text}' element{color_info}.")
            if interaction_details:
                if not any(detail.lower() in description.lower() for detail in interaction_details):
                    description += "\n\nSpecific actions:\n- " + "\n- ".join(interaction_details)
        return description

    def clean_transcript(self, transcript):
        fillers = ['um', 'uh', 'er', 'mm', 'like', 'you know', 'sort of', 'kind of', 'basically']
        clean_text = transcript
        for filler in fillers:
            clean_text = re.sub(r'\b' + filler + r'\b', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        sentences = [s.capitalize() for s in sentences if s]
        clean_text = ' '.join(sentences)
        return clean_text

    def generate_success_criteria(self, interacted_elements, key_ui_elements, inputs, transcript):
        criteria = []
        for elem in interacted_elements:
            elem_type = elem['type'].lower()
            elem_text = elem['text']
            if elem_type == 'button':
                criteria.append(f"The button labeled '{elem_text}' has been clicked")
            elif elem_type == 'link':
                criteria.append(f"The link labeled '{elem_text}' has been clicked")
            elif elem_type in ['textbox', 'input', 'text field'] and inputs:
                criteria.append(f"Information has been entered in the '{elem_text}' field")
        success_patterns = [
            (r'you should see', "You can see"),
            (r'should appear', "The element appears on screen"),
            (r'will show', "The element is visible"),
            (r'should load', "The page has loaded"),
            (r'will take you to', "You have been redirected to the correct page")
        ]
        for pattern, result in success_patterns:
            if re.search(pattern, transcript, re.IGNORECASE):
                criteria.append(result)
        if not criteria:
            if any(elem['type'].lower() == 'button' for elem in key_ui_elements):
                criteria.append("The button has been clicked successfully")
            elif inputs:
                criteria.append("Information has been entered correctly")
            else:
                criteria.append("The step has been completed successfully")
        return criteria

    def generate_structured_output(self, steps):
        output = ""
        for step in steps:
            output += f"## Step {step['step_number']}: {step['title']}\n\n"
            output += f"1. **Description**: {step['description']}\n\n"
            output += "2. **UI Elements to Look For**:\n"
            if step['ui_elements']:
                for elem in step['ui_elements']:
                    element_desc = f"{elem['type']}: {elem['text']}"
                    if 'bbox' in elem:
                        x, y, w, h = elem['bbox']
                        position = self.describe_position(x, y, w, h)
                        color_info = f", {elem['color']}" if elem.get('color') else ""
                        element_desc += f" ({position}{color_info})"
                    output += f"   - {element_desc}\n"
            else:
                output += "   - No specific UI elements identified\n"
            output += "\n3. **Inputs/Selections Required**:\n"
            if step['inputs']:
                for input_item in step['inputs']:
                    output += f"   - {input_item}\n"
            else:
                output += "   - No specific inputs required\n"
            output += "\n4. **Success Criteria**:\n"
            for criteria in step['success_criteria']:
                output += f"   - {criteria}\n"
            output += "\n---\n\n"
        return output

    def describe_position(self, x, y, w, h, img_width=1920, img_height=1080):
        if x < img_width * 0.33:
            h_pos = "left"
        elif x > img_width * 0.66:
            h_pos = "right"
        else:
            h_pos = "center"
        if y < img_height * 0.33:
            v_pos = "top"
        elif y > img_height * 0.66:
            v_pos = "bottom"
        else:
            v_pos = "middle"
        if h_pos == "center" and v_pos == "middle":
            return "in the center of the screen"
        else:
            return f"in the {v_pos}-{h_pos} of the screen"


def process_video(video_path):
    analyzer = OnboardingVideoAnalyzer(video_path)
    structured_output = analyzer.process()
    return structured_output


if __name__ == "__main__":
    video_path = "recording.webm"
    try:
        import torch
        import whisper
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "torch", "openai-whisper", "transformers", "sentence-transformers", "pytesseract"])
    output = process_video(video_path)
    with open("onboarding_steps.md", "w") as f:
        f.write(output)
    print("Processing complete. Results saved to onboarding_steps.md")
