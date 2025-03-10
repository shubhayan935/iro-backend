import os
import cv2
import numpy as np
import torch
import whisper
from moviepy.editor import VideoFileClip
import json
import re
from openai import OpenAI
from transformers import pipeline
from PIL import Image
import time
from sentence_transformers import SentenceTransformer

########################################
# Helper Classes for Visual Processing #
########################################

class UIElementDetector:
    """A placeholder detector that simulates detecting UI elements in a frame."""
    def __init__(self):
        pass

    def detect(self, frame):
        # Convert to grayscale and threshold to simulate detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ui_elements = []

        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            element_type = self._classify_element(w, h)
            element_text = self._extract_text(frame[y:y+h, x:x+w])
            ui_elements.append({
                'type': element_type,
                'text': element_text,
                'bbox': (x, y, w, h)
            })
        return ui_elements

    def _classify_element(self, width, height):
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

    def _extract_text(self, element_img):
        # In a real solution you might use OCR.
        brightness = np.mean(element_img)
        if brightness > 200:
            return "Light UI element"
        elif brightness < 100:
            return "Dark UI element"
        else:
            return "UI element"


class CursorTracker:
    """A placeholder cursor tracker that simulates cursor detection."""
    def __init__(self):
        pass

    def track(self, frame):
        # Simulate cursor detection with a 70% chance of detecting a cursor.
        import random
        if random.random() < 0.7:
            h, w = frame.shape[:2]
            x = random.randint(0, w)
            y = random.randint(0, h)
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


#############################################
# Onboarding Video Analyzer (Multimodal)    #
#############################################

class OnboardingVideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.temp_dir = "temp_processing"
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.audio_path = os.path.join(self.temp_dir, "audio.wav")

        # Create temporary directories
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        print("Loading models...")
        self.transcription_model = whisper.load_model("medium")
        self.nlp_summarizer = pipeline("summarization")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ui_detector = UIElementDetector()
        self.cursor_tracker = CursorTracker()
        print("Models loaded successfully")

    def process(self):
        """Main processing function."""
        print(f"Processing video: {self.video_path}")

        # Extract audio and frames
        self.extract_audio()
        self.extract_frames()

        # Transcribe audio
        transcript = self.transcribe_audio()

        # Analyze frames (detect UI elements and cursor actions)
        frames_data = self.analyze_frames()

        # Segment into meaningful steps
        steps = self.segment_into_steps(transcript, frames_data)

        # Optionally, refine the extracted steps using GPT
        refined_steps = self.refine_steps_with_gpt(steps)

        return refined_steps

    def extract_audio(self):
        """Extract audio from the video file."""
        print(f"Extracting audio from {self.video_path}...")
        video = VideoFileClip(self.video_path)
        audio = video.audio
        audio.write_audiofile(self.audio_path)
        self.video_duration = video.duration
        self.fps = video.fps
        video.close()

    def extract_frames(self, frame_interval=5):
        """Extract frames at regular intervals with timestamps."""
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

                # Compute motion between frames
                motion = 0
                if previous_frame is not None:
                    motion = self.detect_motion(previous_frame, frame)

                self.frame_data.append({
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'path': frame_path,
                    'motion_detected': motion > 0.002  # threshold
                })

                previous_frame = frame.copy()
            frame_count += 1

        cap.release()
        print(f"Extracted {len(self.frame_data)} frames")
        return self.frame_data

    def detect_motion(self, prev_frame, curr_frame):
        """Detect motion between two frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        motion_score = np.sum(diff) / (prev_gray.shape[0] * prev_gray.shape[1] * 255)
        return motion_score

    def transcribe_audio(self):
        """Transcribe the audio using Whisper with word timestamps."""
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
        """Analyze frames for UI elements, cursor position, and user interactions."""
        print("Analyzing frames for UI elements and actions...")
        analyzed_frames = []
        for i, frame_data in enumerate(self.frame_data):
            print(f"Analyzing frame {i+1}/{len(self.frame_data)}")
            frame = cv2.imread(frame_data['path'])

            ui_elements = self.ui_detector.detect(frame)
            cursor_info = self.cursor_tracker.track(frame)

            # Detect a click based on motion and cursor overlap with UI element
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
        """Segment the video into logical onboarding steps using both transcript and visual cues."""
        print("Segmenting video into logical steps...")

        potential_steps = []

        # 1. Look for speech markers in the transcript
        for segment in transcript_segments:
            text = segment['text'].lower()
            step_markers = ["step", "first", "second", "third", "fourth", "fifth", "next", "then", "click on", "open", "navigate"]
            for marker in step_markers:
                if marker in text:
                    potential_steps.append({
                        'type': 'speech_marker',
                        'timestamp': segment['start'],
                        'segment': segment
                    })
                    break

        # 2. Look for significant UI interactions (clicks)
        for frame in frames_data:
            if frame['click_detected'] and frame['clicked_element'] is not None:
                potential_steps.append({
                    'type': 'ui_interaction',
                    'timestamp': frame['timestamp'],
                    'frame': frame
                })

        # 3. Look for screen transitions (changes in the number of UI elements)
        for i in range(1, len(frames_data)):
            prev_elements = len(frames_data[i-1]['ui_elements'])
            curr_elements = len(frames_data[i]['ui_elements'])
            if abs(prev_elements - curr_elements) > 3:
                potential_steps.append({
                    'type': 'screen_transition',
                    'timestamp': frames_data[i]['timestamp'],
                    'frame': frames_data[i]
                })

        # Sort and merge steps that are very close in time
        potential_steps.sort(key=lambda x: x['timestamp'])
        merged_steps = []
        for step in potential_steps:
            if not merged_steps or step['timestamp'] - merged_steps[-1]['timestamp'] > 2.0:
                merged_steps.append(step)
            else:
                if step['type'] == 'speech_marker' and merged_steps[-1]['type'] != 'speech_marker':
                    merged_steps[-1] = step

        # Create step boundaries
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
        """Extract key information for a single step."""
        step_text = " ".join([seg['text'] for seg in transcript_segments])
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
            if elem['text'] and elem['text'] not in unique_elements:
                unique_elements[elem['text']] = elem

        scored_elements = []
        for elem_text, elem in unique_elements.items():
            mentioned = elem_text.lower() in step_text.lower()
            interacted = any(frame['click_detected'] and frame['clicked_element'] and frame['clicked_element']['text'] == elem_text for frame in frames)
            score = 0
            if mentioned:
                score += 1
            if interacted:
                score += 2
            scored_elements.append({'element': elem, 'score': score})

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
                for pattern_prefix, action_type in action_types.items():
                    if pattern.startswith(pattern_prefix):
                        return f"{action_type} {action_object}"
                return f"Interact with {action_object}"
        if interacted_elements:
            elem = interacted_elements[0]
            elem_type = elem['type'].lower()
            elem_text = elem['text']
            if elem_type == 'button':
                return f"Click the {elem_text} button"
            elif elem_type == 'link':
                return f"Navigate to {elem_text}"
            elif elem_type in ['textbox', 'input', 'text field']:
                return f"Enter information in {elem_text} field"
            else:
                return f"Interact with {elem_text}"
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
                if elem_type == 'button':
                    interaction_details.append(f"Click on the '{elem_text}' button.")
                elif elem_type == 'link':
                    interaction_details.append(f"Click on the '{elem_text}' link.")
                elif elem_type in ['textbox', 'input', 'text field']:
                    matching_input = None
                    for inp in inputs:
                        if (elem_text.lower() in inp.lower() or
                            any(word in elem_text.lower() for word in inp.lower().split())):
                            matching_input = inp
                            break
                    if matching_input:
                        interaction_details.append(f"Enter '{matching_input}' in the '{elem_text}' field.")
                    else:
                        interaction_details.append(f"Enter information in the '{elem_text}' field.")
                else:
                    interaction_details.append(f"Interact with the '{elem_text}' element.")
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
                criteria.append(f"The '{elem_text}' button has been clicked")
            elif elem_type == 'link':
                criteria.append(f"The '{elem_text}' link has been clicked")
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
                        element_desc += f" ({position})"
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

    def refine_steps_with_gpt(self, steps):
        """
        Uses OpenAI GPT-4 to refine and structure the extracted steps.
        The prompt instructs GPT to output a JSON array where each element has:
        'title', 'description', 'ui_elements', 'inputs', and 'success_criteria'.
        """

        prompt = (
            "Extract the onboarding steps from the following data. For each step, provide:\n"
            "1. A clear title summarizing the step\n"
            "2. A detailed description explaining exactly what to do\n"
            "3. Any UI elements to look for (buttons, fields, menus)\n"
            "4. Any specific inputs or selections that need to be made\n"
            "5. Success criteria for completing the step\n\n"
            "Here is the extracted data:\n\n"
            f"{self.generate_structured_output(steps)}\n\n"
            "Output the result as a JSON array where each element is an object with the keys: "
            "'title', 'description', 'ui_elements', 'inputs', 'success_criteria'."
        )
        try:
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts structured onboarding steps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            refined_output = response.choices[0].message.content
            try:
                refined_steps = json.loads(refined_output)
                return refined_steps
            except json.JSONDecodeError:
                print("Failed to parse JSON from GPT output, returning raw output.")
                return refined_output
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return steps


#####################################
# Main Processing Function          #
#####################################

def process_video(video_path):
    analyzer = OnboardingVideoAnalyzer(video_path)
    structured_output = analyzer.process()
    return structured_output


if __name__ == "__main__":
    video_path = "recording.webm"  # Ensure your recording file is here
    output = process_video(video_path)
    # Save the refined steps as JSON
    with open("onboarding_steps.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Processing complete. Results saved to onboarding_steps.json")
