"""
HybridPdM - Hugging Face Spaces 진입점.

HF Spaces는 관례상 `app.py` 의 `demo` 객체를 자동으로 launch한다.
로컬 개발/ngrok 터널용 엔트리는 `gradio_app.py` 를 그대로 유지하며,
본 파일은 `demo` 를 재노출하는 얇은 래퍼 역할만 한다.

로컬 실행:
    python app.py          # HF Spaces 환경 시뮬레이션
    python gradio_app.py   # 기존 엔트리 (동일 동작)
"""
from __future__ import annotations

import os

from gradio_app import demo

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    _port_env = os.getenv("GRADIO_SERVER_PORT")
    server_port = int(_port_env) if _port_env else 7860
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
