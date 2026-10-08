import os
import cv2
import base64
import requests

from .config import config

class Generator:
    def __init__(self):
        # Ollama configuration from environment variable or RAGConfig
        self.base_url = os.environ.get("OLLAMA_BASE_URL", config.ollama_base_url)
        self.text_model = os.environ.get("OLLAMA_TEXT_MODEL", config.llm_text_model)
        self.vision_model = os.environ.get("OLLAMA_VISION_MODEL", config.llm_vision_model)
        print(f"Generator initialized. Using Ollama at {self.base_url}")
        print(f"  - Text model: {self.text_model}")
        print(f"  - Vision model: {self.vision_model}")

    def _call_ollama(self, model: str, prompt: str, images: list = None) -> str:
        """
        Send a prompt (optionally with base64 images) to the Ollama API
        and return the generated text.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            payload["images"] = images  # List of base64-encoded strings

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return (
                f"**Error: Could not connect to Ollama at `{self.base_url}`.**\n\n"
                "Please make sure Ollama is running. You can start it by running:\n"
                "```\nollama serve\n```\n"
                f"And pull the required model:\n```\nollama pull {model}\n```"
            )
        except requests.exceptions.Timeout:
            return "**Error: Ollama request timed out.** The model may still be loading. Please try again."
        except Exception as e:
            return f"**Generation Error:** Failed to get response from Ollama: {str(e)}"

    def _extract_key_frames(self, video_path: str, num_frames: int = 4) -> list:
        """
        Extract evenly-spaced key frames from a video file.
        Returns a list of base64-encoded JPEG image strings.
        """
        import numpy as np
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video for frame extraction: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []

        frame_indices = set(
            int(i) for i in np.linspace(0, max(0, total_frames - 1), num_frames)
        )

        base64_frames = []
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in frame_indices:
                # Encode frame to JPEG bytes, then base64
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if success:
                    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                    base64_frames.append(b64)
            current_frame += 1

        cap.release()
        print(f"Extracted {len(base64_frames)} key frames from video.")
        return base64_frames

    def generate_report(self, anomaly_data: dict, context_docs: list) -> str:
        """
        Constructs a prompt and gets a text-only response from the local LLM (Llama3).
        """
        context_text = "\n\n".join([
            f"- {d['document']['content']} (Source: {d['document'].get('source', 'Unknown')})"
            for d in context_docs
        ])

        prompt = f"""You are an AI assistant tasked with generating a formal security incident report.

INCIDENT DETAILS:
Type: {anomaly_data.get('type', 'Unknown')}
Location: {anomaly_data.get('location', 'Unknown')}
Time: {anomaly_data.get('timestamp', 'Unknown')}
Description: {anomaly_data.get('description', 'Anomaly detected in video feed.')}

RELEVANT POLICIES & PROCEDURES (CONTEXT):
{context_text}

INSTRUCTIONS:
Using the incident details and the relevant policies provided above, write a comprehensive incident report based strictly on the facts presented. If the policy demands specific actions, outline them clearly. Do NOT make up new facts.

The report should include:
- Incident Summary
- Location & Timestamp Verification
- Relevant Policy Citation
- Recommended Immediate Actions (based on policy)
- Conclusion

REPORT:
"""
        return self._call_ollama(self.text_model, prompt)

    def generate_report_from_video(self, video_path: str, anomaly_data: dict, context_docs: list) -> str:
        """
        Extracts key frames from the video, then sends them to the vision model (LLaVA)
        along with the prompt to generate a multimodal incident report.
        """
        context_text = "\n\n".join([
            f"- {d['document']['content']} (Source: {d['document'].get('source', 'Unknown')})"
            for d in context_docs
        ])

        print(f"Extracting key frames from video: {video_path}")
        base64_frames = self._extract_key_frames(video_path, num_frames=4)

        if not base64_frames:
            print("Warning: No frames extracted. Falling back to text-only report.")
            return self.generate_report(anomaly_data, context_docs)

        prompt = f"""You are a professional security operations center analyst reporting an incident strictly based on visual facts derived from video evidence. Do not hallucinate.

INCIDENT METADATA:
Type: {anomaly_data.get('type', 'Unknown')}
Location: {anomaly_data.get('location', 'Unknown')}
Time: {anomaly_data.get('timestamp', 'Unknown')}
Description/User Notes: {anomaly_data.get('description', 'Anomaly detected.')}

RELEVANT POLICIES & PROCEDURES (CONTEXT):
{context_text}

INSTRUCTIONS:
1. Carefully examine the provided video frames to understand what happened.
2. Using the visual evidence from the frames and the contextual policies above, write a comprehensive incident report.
3. If the video frames contradict the user notes, rely on the objective visual evidence.
4. If the policy demands specific administrative or security actions, outline them clearly.
5. Do NOT make up new facts that are not visibly present in the frames.

The report should include:
- Incident Summary (Describe what visually occurred based on the frames)
- Location & Timestamp Verification
- Relevant Policy Citation
- Recommended Immediate Actions (based on policy and visual severity)
- Conclusion

REPORT:
"""
        print(f"Sending {len(base64_frames)} frames to vision model '{self.vision_model}'...")
        return self._call_ollama(self.vision_model, prompt, images=base64_frames)
