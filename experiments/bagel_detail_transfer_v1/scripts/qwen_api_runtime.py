import base64
import io
import json
import os
import time
from urllib.error import HTTPError
from urllib import request


class QwenAPI:
    is_api_client = True

    def __init__(self):
        self.api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise RuntimeError('DASHSCOPE_API_KEY is required and is never read from a source file')
        self.endpoint = os.environ.get(
            'DASHSCOPE_BASE_URL',
            'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        )
        self.model = os.environ.get('QWEN_API_MODEL', 'qwen-vl-max')
        self.timeout = int(os.environ.get('QWEN_API_TIMEOUT', '180'))

    @staticmethod
    def _image_data(image):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')

    def respond(self, images, prompts, limit=4096):
        results = []
        for image, prompt in zip(images, prompts, strict=True):
            content = []
            if image is not None:
                content.append({'type': 'image_url', 'image_url': {'url': self._image_data(image)}})
            content.append({'type': 'text', 'text': prompt})
            payload = json.dumps({
                'model': self.model,
                'messages': [{'role': 'user', 'content': content}],
                'temperature': 0,
                'max_tokens': min(limit, int(os.environ.get('QWEN_API_MAX_TOKENS', '2048'))),
            }).encode()
            last_error = None
            for attempt in range(4):
                try:
                    req = request.Request(self.endpoint, data=payload, headers={
                        'Authorization': 'Bearer ' + self.api_key,
                        'Content-Type': 'application/json',
                    })
                    with request.urlopen(req, timeout=self.timeout) as response:
                        value = json.loads(response.read())
                    results.append(value['choices'][0]['message']['content'])
                    break
                except HTTPError as error:
                    body = error.read().decode('utf-8', errors='replace')[:1000]
                    last_error = RuntimeError(f'HTTP {error.code}: {body}')
                    if error.code in (400, 401, 403, 404):
                        break
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                except Exception as error:
                    last_error = error
                    if attempt < 3:
                        time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f'Qwen API request failed after retries: {last_error}') from last_error
        return results
