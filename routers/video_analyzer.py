# routers/video_analyzer.py
import os
import cv2
import numpy as np
import json
import re
import logging
from typing import List, Dict, Any, Optional

from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
import models_loader

logger = logging.getLogger(__name__)

class OnboardingVideoAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.temp_dir = "temp_processing"
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.audio_path = os.path.join(self.temp_dir, "audio.wav")

        # Create temporary directories
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        logger.info("Initializing OnboardingVideoAnalyzer")
        
        # Get models from loader - will raise appropriate exceptions if models aren't available
        self.transcription_model = models_loader.get_model("transcription_model")
        if self.transcription_model is None:
            raise ValueError("Transcription model not loaded. Please ensure models are loaded first.")
            
        self.nlp_summarizer = models_loader.get_model("nlp_summarizer")
        self.sentence_model = models_loader.get_model("sentence_model")
        self.ui_detector = models_loader.get_model("ui_detector")
        self.cursor_tracker = models_loader.get_model("cursor_tracker")
        
        logger.info("Analyzer initialized successfully")

    def process(self):
        """Main processing function."""
        logger.info(f"Processing video: {self.video_path}")

        # Extract audio and frames
        self.extract_audio()
        self.extract_frames()

        # Transcribe audio
        transcript = self.transcribe_audio()

        # Analyze frames (detect UI elements and cursor actions)
        frames_data = self.analyze_frames()

        # Segment into meaningful steps
        steps = self.segment_into_steps(transcript, frames_data)

        # Refine steps with GPT (if API key available)
        refined_steps = self.refine_steps_with_gpt(steps)

        return refined_steps

    def extract_audio(self):
        """Extract audio from the video file with better error handling."""
        logger.info(f"Extracting audio from {self.video_path}")
        
        try:
            # First try with moviepy
            from moviepy.editor import VideoFileClip
            try:
                video = VideoFileClip(self.video_path)
                audio = video.audio
                
                # Check if audio exists
                if audio is None:
                    logger.warning("No audio track found in video file")
                    # Create an empty audio file as fallback
                    with open(self.audio_path, 'w') as f:
                        f.write("")
                    self.video_duration = video.duration
                    self.fps = video.fps
                    video.close()
                    return
                    
                audio.write_audiofile(self.audio_path, verbose=False, logger=None)
                self.video_duration = video.duration
                self.fps = video.fps
                video.close()
                logger.info("Audio extraction complete")
                
            except Exception as moviepy_error:
                logger.warning(f"MoviePy error: {str(moviepy_error)}")
                logger.info("Trying alternative approach with FFmpeg directly...")
                
                # Alternative approach using ffmpeg directly
                import subprocess
                
                # Get video duration using ffprobe
                duration_cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-show_entries", "format=duration", 
                    "-of", "default=noprint_wrappers=1:nokey=1", 
                    self.video_path
                ]
                
                try:
                    result = subprocess.run(duration_cmd, capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        self.video_duration = float(result.stdout.strip())
                    else:
                        self.video_duration = 0
                        logger.warning("Could not determine video duration, using 0")
                except Exception as e:
                    logger.warning(f"Error getting duration: {str(e)}")
                    self.video_duration = 0
                
                # Get fps using ffprobe
                fps_cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-select_streams", "v:0", 
                    "-show_entries", "stream=r_frame_rate", 
                    "-of", "default=noprint_wrappers=1:nokey=1", 
                    self.video_path
                ]
                
                try:
                    result = subprocess.run(fps_cmd, capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        # Parse frame rate which might be in the format "num/den"
                        fr_parts = result.stdout.strip().split('/')
                        if len(fr_parts) == 2:
                            self.fps = float(fr_parts[0]) / float(fr_parts[1])
                        else:
                            self.fps = float(result.stdout.strip())
                    else:
                        self.fps = 30  # Default to 30 fps
                        logger.warning("Could not determine video FPS, using 30")
                except Exception as e:
                    logger.warning(f"Error getting FPS: {str(e)}")
                    self.fps = 30
                
                # Extract audio using ffmpeg
                extract_cmd = [
                    "ffmpeg",
                    "-i", self.video_path,
                    "-vn",  # No video
                    "-acodec", "pcm_s16le",  # PCM audio format
                    "-ar", "44100",  # 44.1 kHz sample rate
                    "-ac", "2",  # 2 channels (stereo)
                    "-y",  # Overwrite output file if it exists
                    self.audio_path
                ]
                
                try:
                    subprocess.run(extract_cmd, check=True, capture_output=True)
                    logger.info("Audio extraction with FFmpeg complete")
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg audio extraction failed: {e.stderr.decode() if e.stderr else str(e)}")
                    
                    # Create an empty audio file as last resort
                    with open(self.audio_path, 'w') as f:
                        f.write("")
                    logger.warning("Created empty audio file as fallback")
                    
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            # Create an empty file so processing can continue
            with open(self.audio_path, 'w') as f:
                f.write("")
            logger.warning("Created empty audio file due to extraction error")
            
            # Set default values for video properties
            self.video_duration = 0
            self.fps = 30

    def extract_frames(self, frame_interval: int = 5):
        """Extract frames at regular intervals with timestamps."""
        logger.info(f"Extracting frames from {self.video_path}")
        
        try:
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
            logger.info(f"Extracted {len(self.frame_data)} frames")
            return self.frame_data
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise

    def detect_motion(self, prev_frame, curr_frame):
        """Detect motion between two frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        motion_score = np.sum(diff) / (prev_gray.shape[0] * prev_gray.shape[1] * 255)
        return motion_score

    def transcribe_audio(self):
        """Transcribe the audio using Whisper with word timestamps."""
        logger.info(f"Transcribing audio from {self.audio_path}")
        
        try:
            # Check if audio file exists and has content
            if not os.path.exists(self.audio_path) or os.path.getsize(self.audio_path) == 0:
                logger.warning("Audio file is empty or missing, returning empty transcript")
                self.transcript_segments = []
                self.transcript_text = ""
                return self.transcript_segments
                
            # Check if model is available
            if self.transcription_model is None:
                logger.error("Transcription model not available")
                self.transcript_segments = []
                self.transcript_text = ""
                return self.transcript_segments
            
            # Try transcription
            try:
                result = self.transcription_model.transcribe(
                    self.audio_path,
                    verbose=False,
                    word_timestamps=True
                )
                
                self.transcript_segments = result["segments"]
                self.transcript_text = result["text"]
                
                logger.info(f"Transcription complete: {len(self.transcript_segments)} segments")
                return self.transcript_segments
                
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                self.transcript_segments = []
                self.transcript_text = ""
                return self.transcript_segments
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            self.transcript_segments = []
            self.transcript_text = ""
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

        # 1. Speech markers from transcript
        for segment in transcript_segments:
            text = segment['text'].lower()
            print(text)
            step_markers = ["step", "first", "second", "third", "fourth", "fifth", "next", "then", "click on", "open", "navigate"]
            for marker in step_markers:
                if marker in text:
                    potential_steps.append({
                        'type': 'speech_marker',
                        'timestamp': segment['start'],
                        'segment': segment
                    })
                    break

        # 2. UI interactions (clicks)
        for frame in frames_data:
            if frame['click_detected'] and frame['clicked_element'] is not None:
                potential_steps.append({
                    'type': 'ui_interaction',
                    'timestamp': frame['timestamp'],
                    'frame': frame
                })

        # 3. Screen transitions (change in UI elements)
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
        merged_steps = []
        for step in potential_steps:
            if not merged_steps or step['timestamp'] - merged_steps[-1]['timestamp'] > 2.0:
                merged_steps.append(step)
            else:
                if step['type'] == 'speech_marker' and merged_steps[-1]['type'] != 'speech_marker':
                    merged_steps[-1] = step

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
            score = (1 if mentioned else 0) + (2 if interacted else 0)
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
                # Try to choose a corresponding action type
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
            if interaction_details and not any(detail.lower() in description.lower() for detail in interaction_details):
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
        return ' '.join(sentences)

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

    def describe_position(self, x, y, w, h, img_width: int = 1920, img_height: int = 1080):
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

    def process_openai_response_to_json(self, openai_response):
        """
        Processes the raw text response from OpenAI and converts it to JSON.
        Handles errors gracefully.
        
        Args:
            openai_response (str): The raw text response from OpenAI.
            
        Returns:
            dict or list: The parsed JSON data if successful, or None if parsing fails.
        """
        logger.info("Processing OpenAI response to JSON")
        
        if not openai_response:
            logger.warning("Empty response received, returning None")
            return None
        
        # Try to extract JSON if response contains markdown code blocks
        if "```json" in openai_response:
            try:
                json_content = openai_response.split("```json")[1].split("```")[0].strip()
                parsed_data = json.loads(json_content)
                logger.info(f"Successfully parsed JSON from markdown code block")
                return parsed_data
            except (IndexError, json.JSONDecodeError) as e:
                logger.error(f"Failed to extract JSON from markdown: {str(e)}")
                # Continue to try parsing the whole response
        
        # Try to parse the entire response as JSON
        try:
            parsed_data = json.loads(openai_response)
            logger.info(f"Successfully parsed JSON from response")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {str(e)}")
            
            # Last resort: try to find anything that looks like JSON in the response
            try:
                # Look for content between curly braces
                import re
                json_pattern = r'(\{.*\}|\[.*\])'
                matches = re.search(json_pattern, openai_response, re.DOTALL)
                if matches:
                    potential_json = matches.group(0)
                    parsed_data = json.loads(potential_json)
                    logger.warning(f"Extracted JSON using regex pattern")
                    return parsed_data
            except (re.error, json.JSONDecodeError) as e:
                logger.error(f"All JSON parsing attempts failed: {str(e)}")
        
        logger.error("Could not convert response to JSON")
        return None

    def refine_steps_with_gpt(self, steps):
        """
        Uses OpenAI GPT-4 (or equivalent) to refine and structure the extracted steps.
        Handles missing API key gracefully.
        """
        logger.info("Refining steps with GPT")
        
        # If no steps or API key, return the steps as is
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not steps:
            logger.warning("No API key or steps, skipping GPT refinement")
            return steps
        
        # Generate structured output for prompt
        structured_output = self.generate_structured_output(steps)
        
        prompt = (
            "Extract the onboarding steps from the following data. For each step, provide:\n"
            "1. A clear title summarizing the step\n"
            "2. A detailed description explaining exactly what to do\n"
            "3. Any UI elements to look for (buttons, fields, menus)\n"
            "4. Any specific inputs or selections that need to be made\n"
            "5. Success criteria for completing the step\n\n"
            "Here is the extracted data:\n\n"
            f"{structured_output}\n\n"
            "Format your response as a valid JSON array of step objects."
        )
        
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # change to your desired model
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts structured onboarding steps in JSON data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            raw_output = response.choices[0].message.content
            
            # Process the raw output to JSON using the new function
            refined_steps = self.process_openai_response_to_json(raw_output)
            
            if refined_steps:
                logger.info(f"Successfully refined steps with GPT: {len(refined_steps)} steps")
                print(f"Steps are: {refined_steps}")
                return refined_steps
            else:
                logger.warning("Could not parse GPT output to JSON, returning original steps")
                return steps
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return steps