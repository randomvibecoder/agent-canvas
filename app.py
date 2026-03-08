import asyncio
import base64
import io
import json
import uuid
from contextlib import asynccontextmanager

import aiohttp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NANO_GPT_API_URL = os.getenv(
    "NANO_GPT_API_URL", "https://nano-gpt.com/api/v1/chat/completions"
)
NANO_GPT_API_KEY = os.getenv("NANO_GPT_API_KEY", "")
MODEL_NAME = "minimax/minimax-m2.5"

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

canvas_elements: dict[str, dict] = {}
connected_clients: set[WebSocket] = set()
chat_lock = asyncio.Lock()

display_messages: list[dict] = []
llm_messages: list[dict] = []


class UserChatRequest(BaseModel):
    content: str
    display_content: str | None = None


def summarize_element(element: dict) -> dict:
    return {
        "x": element.get("x", 0),
        "y": element.get("y", 0),
        "width": element.get("width", 0),
        "height": element.get("height", 0),
        "angle": element.get("angle", 0),
    }


async def broadcast(payload: dict) -> None:
    if not connected_clients:
        return

    stale_clients: list[WebSocket] = []
    message = json.dumps(payload)

    for client in list(connected_clients):
        try:
            await client.send_text(message)
        except Exception:
            stale_clients.append(client)

    for client in stale_clients:
        connected_clients.discard(client)


async def broadcast_canvas_state() -> None:
    await broadcast({"type": "canvas_update", "elements": canvas_elements})


async def broadcast_chat_state() -> None:
    await broadcast({"type": "chat_update", "messages": display_messages})


def execute_matplotlib(code: str) -> str:
    plt.close("all")
    try:
        exec(code, {"np": np, "plt": plt}, {})
        figures = plt.get_fignums()
        if not figures:
            return "Error: matplotlib code did not create a figure"
        fig = plt.figure(figures[0])
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        plt.close("all")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"
    except Exception as exc:
        plt.close("all")
        return f"Error executing matplotlib: {exc}"


async def call_nano_gpt(messages: list[dict], tools: list[dict] | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {NANO_GPT_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            NANO_GPT_API_URL, json=payload, headers=headers
        ) as response:
            return await response.json()


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "canvas_tool",
        "description": "Create or manipulate images on a shared 800x600 canvas. Use matplotlib Python code to create an image, then position it. The tool returns concise JSON summaries, not image data.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create",
                        "move",
                        "resize",
                        "rotate",
                        "center",
                        "delete",
                        "list",
                    ],
                    "description": "Canvas action to perform",
                },
                "python_code": {
                    "type": "string",
                    "description": "Matplotlib Python code. Use only plt and np. NEVER call plt.close() or plt.show() - just create the plot and leave it open. Example: plt.plot([1,2,3],[1,4,9]); plt.title('My Plot')  # no plt.close() or plt.show()",
                },
                "image_id": {
                    "type": "string",
                    "description": "Target image id for non-create actions",
                },
                "x": {
                    "type": "number",
                    "description": "Left position in pixels",
                },
                "y": {
                    "type": "number",
                    "description": "Top position in pixels",
                },
                "width": {
                    "type": "number",
                    "description": "Image width in pixels",
                },
                "height": {
                    "type": "number",
                    "description": "Image height in pixels",
                },
                "angle": {
                    "type": "number",
                    "description": "Rotation angle in degrees",
                },
            },
            "required": ["action"],
        },
    },
}


SYSTEM_PROMPT = (
    "You control a shared "
    + str(CANVAS_WIDTH)
    + "x"
    + str(CANVAS_HEIGHT)
    + " canvas.\n\n"
    + "You have one tool: canvas_tool.\n\n"
    + "Use it to:\n"
    + "- create matplotlib images\n"
    + "- move, resize, rotate, center, delete, and list canvas items\n\n"
    + "CANVAS LAYOUT RULES - follow these EXACTLY:\n\n"
    + "IMPORTANT: Canvas is 1920x1080. Always set figsize=(19.2, 10.8) in matplotlib.\n\n"
    + "1. First image ever:\n"
    + "   - Width: 1920px, Height: 1080px\n"
    + "   - Position: x=0, y=0\n\n"
    + "2. Second image (2 images total):\n"
    + "   - Split vertically into two equal panes with 10px gap\n"
    + "   - Left pane: x=0, y=0, width=950px, height=1080px\n"
    + "   - Right pane: x=960, y=0, width=950px, height=1080px\n"
    + "   - Move existing image to left (x=0, y=0)\n"
    + "   - Create new image in right pane (x=960, y=0)\n\n"
    + "3. Third image (3 images total):\n"
    + "   - Left half split vertically (two stacked panes)\n"
    + "   - Left top: x=0, y=0, width=950px, height=530px\n"
    + "   - Left bottom: x=0, y=540, width=950px, height=530px\n"
    + "   - Right (new): x=960, y=0, width=950px, height=1080px\n\n"
    + "4. Fourth image (4 images total):\n"
    + "   - 4 equal quadrants with 10px gaps\n"
    + "   - Top-left: x=0, y=0, width=950px, height=530px\n"
    + "   - Top-right: x=960, y=0, width=950px, height=530px\n"
    + "   - Bottom-left: x=0, y=540, width=950px, height=530px\n"
    + "   - Bottom-right: x=960, y=540, width=950px, height=530px\n\n"
    + "Always recalculate positions when adding new images.\n\n"
    + "CRITICAL matplotlib rules - follow these exactly:\n"
    + "- ALWAYS use figsize=(19.2, 10.8) for full resolution\n"
    + "- NEVER call plt.close() or plt.show() - they will prevent image creation\n"
    + "- NEVER use fig.clf() or fig.clear()\n"
    + "- Just create the plot and leave it open\n"
    + "- Example: plt.figure(figsize=(19.2,10.8)); plt.plot([1,2,3],[1,4,9]); plt.title('My Plot')  # no plt.close(), no plt.show()\n\n"
    + "Important rules:\n"
    + "- The tool result is the source of truth for image ids and geometry.\n"
    + "- After create, use the returned image_id for move/resize/rotate/delete.\n"
    + "- Don't call list after create - the create result tells you everything.\n"
    + "- After tool use, answer briefly and clearly."
)


def list_canvas_summary() -> dict:
    return {
        "canvas": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
        "items": {
            image_id: summarize_element(element)
            for image_id, element in canvas_elements.items()
        },
    }


async def run_canvas_tool(arguments: dict) -> str:
    action = arguments.get("action", "")
    image_id = arguments.get("image_id") or str(uuid.uuid4())

    if action == "create":
        python_code = arguments.get("python_code", "")
        if not python_code:
            return json.dumps({"status": "error", "message": "python_code is required"})

        image_url = execute_matplotlib(python_code)
        if not image_url.startswith("data:image"):
            return json.dumps({"status": "error", "message": image_url})

        element = {
            "image_url": image_url,
            "x": arguments.get("x", 100),
            "y": arguments.get("y", 100),
            "width": arguments.get("width", 400),
            "height": arguments.get("height", 300),
            "angle": arguments.get("angle", 0),
        }
        canvas_elements[image_id] = element
        await broadcast_canvas_state()
        return json.dumps(
            {
                "status": "created",
                "image_id": image_id,
                **summarize_element(element),
            }
        )

    if action == "list":
        return json.dumps(list_canvas_summary())

    if image_id not in canvas_elements:
        return json.dumps(
            {
                "status": "error",
                "message": f"image '{image_id}' not found",
                **list_canvas_summary(),
            }
        )

    element = canvas_elements[image_id]

    if action == "move":
        element["x"] = arguments.get("x", element["x"])
        element["y"] = arguments.get("y", element["y"])
        await broadcast_canvas_state()
        return json.dumps(
            {"status": "moved", "image_id": image_id, **summarize_element(element)}
        )

    if action == "resize":
        element["width"] = arguments.get("width", element["width"])
        element["height"] = arguments.get("height", element["height"])
        await broadcast_canvas_state()
        return json.dumps(
            {"status": "resized", "image_id": image_id, **summarize_element(element)}
        )

    if action == "rotate":
        element["angle"] = arguments.get("angle", element["angle"])
        await broadcast_canvas_state()
        return json.dumps(
            {"status": "rotated", "image_id": image_id, **summarize_element(element)}
        )

    if action == "center":
        element["x"] = (CANVAS_WIDTH - element["width"]) / 2
        element["y"] = (CANVAS_HEIGHT - element["height"]) / 2
        await broadcast_canvas_state()
        return json.dumps(
            {"status": "centered", "image_id": image_id, **summarize_element(element)}
        )

    if action == "delete":
        del canvas_elements[image_id]
        await broadcast_canvas_state()
        return json.dumps({"status": "deleted", "image_id": image_id})

    return json.dumps({"status": "error", "message": f"unknown action '{action}'"})


async def run_chat_turn(user_content: str, display_content: str | None = None) -> None:
    display = display_content if display_content else user_content
    display_messages.append({"role": "user", "content": display})
    llm_messages.append({"role": "user", "content": user_content})
    await broadcast_chat_state()

    for _ in range(8):
        response = await call_nano_gpt(
            [{"role": "system", "content": SYSTEM_PROMPT}, *llm_messages],
            [TOOL_DEFINITION],
        )

        choices = response.get("choices") or []
        if not choices:
            display_messages.append(
                {"role": "error", "content": f"Model error: {json.dumps(response)}"}
            )
            await broadcast_chat_state()
            return

        message = choices[0].get("message", {})
        assistant_content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []

        assistant_message = {"role": "assistant", "content": assistant_content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        llm_messages.append(assistant_message)

        if assistant_content:
            display_messages.append({"role": "assistant", "content": assistant_content})
            await broadcast_chat_state()

        if not tool_calls:
            return

        for tool_call in tool_calls:
            raw_arguments = tool_call.get("function", {}).get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
                pretty_arguments = json.dumps(arguments, indent=2)
            except json.JSONDecodeError:
                arguments = None
                pretty_arguments = raw_arguments

            display_messages.append({"role": "tool_call", "content": pretty_arguments})
            await broadcast_chat_state()

            if arguments is None:
                tool_result = json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid JSON arguments: {raw_arguments}",
                    }
                )
            else:
                tool_result = await run_canvas_tool(arguments)

            llm_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": tool_result,
                }
            )
            display_messages.append({"role": "tool_result", "content": tool_result})
            await broadcast_chat_state()

    display_messages.append(
        {
            "role": "error",
            "content": "Stopped after too many tool iterations.",
        }
    )
    await broadcast_chat_state()


HOMEPAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Canvas - AI-Powered Data Visualization</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated background */
        .bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            background: 
                radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(168, 85, 247, 0.12) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(56, 189, 248, 0.08) 0%, transparent 60%);
        }
        
        .grid-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            background-image: 
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 60px 60px;
        }
        
        .container {
            position: relative;
            z-index: 1;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
        }
        
        .hero {
            text-align: center;
            max-width: 800px;
        }
        
        .badge {
            display: inline-block;
            padding: 8px 16px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 50px;
            font-size: 13px;
            color: #a5b4fc;
            margin-bottom: 24px;
            letter-spacing: 0.5px;
        }
        
        h1 {
            font-size: clamp(2.5rem, 6vw, 4rem);
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 24px;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 50%, #67e8f9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            font-size: clamp(1.1rem, 2vw, 1.4rem);
            color: #9ca3af;
            margin-bottom: 48px;
            line-height: 1.6;
        }
        
        .cta-group {
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 16px 32px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.3s ease;
            cursor: pointer;
            border: none;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: #e0e0e0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 24px;
            margin-top: 80px;
            max-width: 900px;
            width: 100%;
        }
        
        .feature {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 28px;
            text-align: left;
            transition: all 0.3s ease;
        }
        
        .feature:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateY(-4px);
        }
        
        .feature-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 16px;
        }
        
        .feature h3 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #fff;
        }
        
        .feature p {
            font-size: 14px;
            color: #9ca3af;
            line-height: 1.5;
        }
        
        .footer {
            position: absolute;
            bottom: 24px;
            font-size: 13px;
            color: #6b7280;
        }
        
        @media (max-width: 600px) {
            .features {
                grid-template-columns: 1fr;
            }
            .btn {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="bg"></div>
    <div class="grid-bg"></div>
    
    <div class="container">
        <div class="hero">
            <div class="badge">AI-Powered Visualization</div>
            <h1>Turn Your Data Into Stunning Visualizations</h1>
            <p class="subtitle">
                Upload your data files, describe what you want to see, 
                and watch AI create beautiful charts instantly. 
                No coding required.
            </p>
            
            <div class="cta-group">
                <a href="/canvas" class="btn btn-primary">
                    <span>Try Now</span>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                </a>
            </div>
        </div>
        
        <div class="features">
            <div class="feature">
                <div class="feature-icon">&#x1F4C1;</div>
                <h3>Upload Data Files</h3>
                <p>Simply upload .txt files with your data - finances, metrics, or any numbers. AI understands context instantly.</p>
            </div>
            <div class="feature">
                <div class="feature-icon">&#x1F4CA;</div>
                <h3>Natural Language</h3>
                <p>Describe what you want in plain English. "Show my electric bills over 10 years" - that's all it takes.</p>
            </div>
            <div class="feature">
                <div class="feature-icon">&#x2728;</div>
                <h3>Beautiful Charts</h3>
                <p>AI generates professional matplotlib visualizations with perfect formatting, colors, and labels.</p>
            </div>
        </div>
        
        <div class="footer">
            Built with AI
        </div>
    </div>
</body>
</html>
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Canvas</title>
    <script src="https://unpkg.com/fabric@5.3.0/dist/fabric.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            background: #0f0f0f;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }
        
        /* Full-screen canvas */
        .canvas-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 1;
            background-color: #0f0f0f;
            background-image: 
                linear-gradient(to right, #222 1px, transparent 1px),
                linear-gradient(to bottom, #222 1px, transparent 1px);
            background-size: 50px 50px;
        }
        
        .canvas-container canvas {
            width: 100vw !important;
            height: 100vh !important;
        }
        
        /* Floating chat panel */
        .chat-panel {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: min(600px, 90vw);
            z-index: 10;
            background: #1a1a1a;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        /* Collapsed state - just header */
        .chat-panel.collapsed {
            height: 44px;
            background: rgba(26, 26, 26, 0.7);
        }
        
        .chat-panel.collapsed .chat-body {
            height: 0;
            opacity: 0;
            overflow: hidden;
        }
        
        .chat-panel.collapsed .collapse-icon {
            transform: rotate(0deg);
        }
        
        /* Expanded state */
        .chat-panel.expanded {
            height: 60vh;
            max-height: 500px;
        }
        
        .chat-panel.expanded .chat-body {
            height: calc(100% - 44px);
            opacity: 1;
        }
        
        .chat-panel.expanded .collapse-icon {
            transform: rotate(180deg);
        }
        
        /* Chat header (always visible) */
        .chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: #222;
            cursor: pointer;
            user-select: none;
            height: 44px;
        }
        
        .chat-header:hover {
            background: #2a2a2a;
        }
        
        .chat-title {
            font-size: 15px;
            font-weight: 600;
            color: #fff;
        }
        
        .collapse-icon {
            font-size: 20px;
            color: #888;
            transition: transform 0.3s ease;
        }
        
        /* Chat body (messages + input) */
        .chat-body {
            height: calc(60vh - 44px);
            max-height: 456px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: all 0.3s ease;
            opacity: 1;
        }
        
        /* Messages area */
        .messages {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .messages::-webkit-scrollbar {
            width: 6px;
        }
        .messages::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        .messages::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 3px;
        }
        
        /* Individual messages */
        .message {
            max-width: 90%;
            padding: 10px 12px;
            border-radius: 12px;
            white-space: normal;
            line-height: 1.4;
            word-break: break-word;
            font-size: 14px;
        }
        
        /* Markdown content */
        .message p { margin: 0 0 8px 0; }
        .message p:last-child { margin-bottom: 0; }
        .message code {
            background: rgba(0,0,0,0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 13px;
        }
        .message pre {
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 8px 0;
        }
        .message pre code {
            background: none;
            padding: 0;
        }
        .message ul, .message ol {
            margin: 8px 0;
            padding-left: 20px;
        }
        .message li { margin: 4px 0; }
        .message h1, .message h2, .message h3 { margin: 12px 0 8px 0; }
        .message h1:first-child, .message h2:first-child, .message h3:first-child { margin-top: 0; }
        .message a { color: #67e8f9; }
        .message strong { font-weight: 600; }
        .message em { font-style: italic; }
        .message blockquote {
            border-left: 3px solid #6366f1;
            margin: 8px 0;
            padding-left: 12px;
            color: #9ca3af;
        }
        
        .message.user {
            align-self: flex-end;
            background: #2563eb;
            color: #fff;
        }
        
        .message.assistant {
            align-self: flex-start;
            background: #2a2a2a;
            border: 1px solid #333;
            color: #e0e0e0;
        }
        
        .message.tool_call,
        .message.tool_result,
        .message.error {
            align-self: flex-start;
            width: 100%;
            max-width: 100%;
            font-family: "SFMono-Regular", Consolas, monospace;
            font-size: 11px;
            border-radius: 8px;
            cursor: pointer;
            position: relative;
        }
        
        .message.tool_call .tool-content,
        .message.tool_result .tool-content,
        .message.error .tool-content {
            max-height: 60px;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        
        .message.tool_call.collapsed .tool-content,
        .message.tool_result.collapsed .tool-content,
        .message.error.collapsed .tool-content {
            max-height: 22px;
            overflow: hidden;
        }
        
        .message.tool_call .label,
        .message.tool_result .label,
        .message.error .label {
            cursor: pointer;
        }
        
        .message.tool_call {
            background: #1a3322;
            border: 1px solid #2d5a3d;
            color: #a8e6b3;
        }
        
        .message.tool_result {
            background: #1a1d24;
            border: 1px solid #3a4553;
            color: #a8b8c8;
        }
        
        .message.error {
            background: #2d1a1a;
            border: 1px solid #5a2d2d;
            color: #e6a8a8;
        }
        
        .label {
            display: block;
            margin-bottom: 6px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.7;
            color: inherit;
        }
        
        /* Input row */
        .input-row {
            display: flex;
            gap: 8px;
            padding: 12px;
            background: #1a1a1a;
            border-top: 1px solid #333;
        }
        
        .input-row input {
            flex: 1;
            min-width: 0;
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid #333;
            background: #252525;
            color: #e0e0e0;
            font-size: 14px;
        }
        
        .input-row input::placeholder {
            color: #666;
        }
        
        .input-row input:focus {
            outline: none;
            border-color: #2563eb;
        }
        
        .input-row button {
            border: 0;
            border-radius: 10px;
            padding: 0 16px;
            background: #2563eb;
            color: white;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .input-row button:hover {
            background: #1d4ed8;
        }
        
        .input-row button:disabled {
            opacity: 0.55;
            cursor: wait;
        }
        
        /* Upload button */
        .upload-btn {
            background: #333;
            border: none;
            border-radius: 8px;
            padding: 0 12px;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .upload-btn:hover {
            background: #444;
        }
        
        /* File tag inside input */
        .file-tag {
            display: flex;
            align-items: center;
            gap: 4px;
            background: #333;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            color: #e0e0e0;
            white-space: nowrap;
            max-width: 150px;
        }
        
        .file-icon {
            font-size: 12px;
        }
        
        .file-name {
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Thinking indicator */
        .thinking {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: bold;
            letter-spacing: 2px;
            background: linear-gradient(90deg, #fff 0%, #fff 50%, #0ff 50%, #0ff 100%);
            background-size: 200% 100%;
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            color: transparent;
            animation: thinking-sweep 2s linear infinite;
        }
        
        @keyframes thinking-sweep {
            0% { background-position: 100% 50%; }
            100% { background-position: -100% 50%; }
        }
        
        .spinner {
            width: 12px;
            height: 12px;
            border: 2px solid #444;
            border-top-color: #2563eb;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Canvas wrapper for responsiveness */
        .canvas-wrapper {
            position: relative;
        }
        
        @media (max-width: 700px) {
            .chat-panel {
                width: 95vw;
                bottom: 10px;
            }
            
            .chat-panel.expanded {
                height: 55vh;
                max-height: 450px;
            }
        }
    </style>
</head>
<body>
    <!-- Full-screen canvas -->
    <div class="canvas-container">
        <div class="canvas-wrapper">
            <canvas id="canvas" width="1920" height="1080"></canvas>
        </div>
    </div>
    
    <!-- Floating chat panel -->
    <div class="chat-panel collapsed" id="chat-panel">
        <div class="chat-header" onclick="toggleChat()">
            <span class="chat-title">Chat</span>
            <span class="collapse-icon">−</span>
        </div>
        <div class="chat-body">
            <div class="messages" id="messages">
                <div id="thinking" class="thinking" style="display: none;">
                    <span class="spinner"></span> Thinking...
                </div>
            </div>
            <div class="input-row">
                <input type="file" id="file-input" accept=".txt" style="display: none;" />
                <button id="upload-button" class="upload-btn" title="Upload .txt file">📎</button>
                <div id="file-tag" class="file-tag" style="display: none;">
                    <span class="file-icon">📄</span>
                    <span id="file-name" class="file-name"></span>
                </div>
                <input id="chat-input" type="text" placeholder="Ask for a chart, move or resize images..." />
                <button id="send-button">Send</button>
            </div>
        </div>
    </div>

    <script>
        const messagesEl = document.getElementById('messages');
        const inputEl = document.getElementById('chat-input');
        const buttonEl = document.getElementById('send-button');
        const thinkingEl = document.getElementById('thinking');
        const chatPanel = document.getElementById('chat-panel');
        const canvasEl = document.getElementById('canvas');
        const fileInputEl = document.getElementById('file-input');
        const uploadBtnEl = document.getElementById('upload-button');
        const fileTagEl = document.getElementById('file-tag');
        const fileNameEl = document.getElementById('file-name');
        
        let uploadedFile = null;
        
        uploadBtnEl.addEventListener('click', () => {
            fileInputEl.click();
        });
        
        fileInputEl.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            uploadedFile = file;
            fileNameEl.textContent = file.name;
            fileTagEl.style.display = 'flex';
            fileInputEl.value = '';
        });
        
        const canvas = new fabric.Canvas('canvas', {
            backgroundColor: '#1a1a1a',
            selection: false,
        });

        // Draw grid on canvas (aesthetic only)
        function drawGrid() {
            const gridSize = 50;
            const gridColor = '#2a2a2a';
            for (let i = 0; i <= canvas.width; i += gridSize) {
                canvas.add(new fabric.Line([i, 0, i, canvas.height], { stroke: gridColor, selectable: false, evented: false }));
            }
            for (let j = 0; j <= canvas.height; j += gridSize) {
                canvas.add(new fabric.Line([0, j, canvas.width, j], { stroke: gridColor, selectable: false, evented: false }));
            }
            canvas.renderAll();
        }
        drawGrid();

        let messages = [];
        let chatExpanded = false;

        function toggleChat() {
            chatExpanded = !chatExpanded;
            if (chatExpanded) {
                chatPanel.classList.remove('collapsed');
                chatPanel.classList.add('expanded');
            } else {
                chatPanel.classList.remove('expanded');
                chatPanel.classList.add('collapsed');
            }
        }

        function prettyContent(content) {
            try {
                // Try to parse as JSON first
                const parsed = JSON.parse(content);
                return JSON.stringify(parsed, null, 2);
            } catch {
                // Render as markdown
                return marked.parse(content);
            }
        }

        function renderMessages() {
            // Keep thinking indicator, remove old messages
            const thinkingEl = document.getElementById('thinking');
            messagesEl.innerHTML = '';
            
            for (const message of messages) {
                const wrapper = document.createElement('div');
                wrapper.className = `message ${message.role}`;

                if (message.role === 'tool_call' || message.role === 'tool_result' || message.role === 'error') {
                    // Add collapsible wrapper
                    const contentWrapper = document.createElement('div');
                    contentWrapper.className = 'tool-content';
                    
                    const label = document.createElement('span');
                    label.className = 'label';
                    if (message.role === 'tool_call') label.textContent = 'Tool Call';
                    else if (message.role === 'tool_result') label.textContent = 'Tool Result';
                    else label.textContent = 'Error';
                    
                    // Click on label toggles collapse
                    label.onclick = function(e) {
                        e.stopPropagation();
                        wrapper.classList.toggle('collapsed');
                    };
                    
                    const pre = document.createElement('div');
                    pre.textContent = prettyContent(message.content || '');
                    
                    contentWrapper.appendChild(label);
                    contentWrapper.appendChild(pre);
                    wrapper.appendChild(contentWrapper);
                    
                    // Start collapsed for tool results
                    if (message.role === 'tool_result') {
                        wrapper.classList.add('collapsed');
                    }
                } else {
                    // Regular message - render markdown
                    const contentDiv = document.createElement('div');
                    contentDiv.innerHTML = prettyContent(message.content || '');
                    wrapper.appendChild(contentDiv);
                }
                
                messagesEl.appendChild(wrapper);
            }
            
            // Re-add thinking indicator
            messagesEl.appendChild(thinkingEl);
            
            requestAnimationFrame(() => {
                messagesEl.scrollTop = messagesEl.scrollHeight;
            });
        }

        function applyElementToObject(obj, element) {
            obj.set({
                left: element.x || 0,
                top: element.y || 0,
                angle: element.angle || 0,
                selectable: false,
                evented: false,
            });
            if (obj.width && obj.height) {
                obj.scaleX = (element.width || obj.width) / obj.width;
                obj.scaleY = (element.height || obj.height) / obj.height;
            }
            obj.setCoords();
        }

        function syncCanvas(elements) {
            const existing = new Map(canvas.getObjects().map((obj) => [obj.canvasId, obj]));
            const nextIds = new Set(Object.keys(elements));

            for (const [id, obj] of existing.entries()) {
                if (!nextIds.has(id)) {
                    canvas.remove(obj);
                }
            }

            Object.entries(elements).forEach(([id, element]) => {
                const existingObj = existing.get(id);

                if (!existingObj) {
                    fabric.Image.fromURL(element.image_url, (img) => {
                        img.canvasId = id;
                        applyElementToObject(img, element);
                        canvas.add(img);
                        canvas.renderAll();
                    }, { crossOrigin: 'anonymous' });
                    return;
                }

                applyElementToObject(existingObj, element);
            });

            canvas.renderAll();
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

            socket.onmessage = (event) => {
                const payload = JSON.parse(event.data);
                if (payload.type === 'init') {
                    messages = payload.messages || [];
                    renderMessages();
                    syncCanvas(payload.elements || {});
                    return;
                }
                if (payload.type === 'chat_update') {
                    messages = payload.messages || [];
                    renderMessages();
                    // Hide thinking if we received an assistant response
                    const hasAssistant = messages.some(m => m.role === 'assistant');
                    if (hasAssistant) {
                        thinkingEl.style.display = 'none';
                    }
                    return;
                }
                if (payload.type === 'canvas_update') {
                    syncCanvas(payload.elements || {});
                }
            };

            socket.onclose = () => {
                setTimeout(connectWebSocket, 1000);
            };
        }

        async function sendMessage() {
            let content = inputEl.value.trim();
            if (!content && !uploadedFile) {
                return;
            }
            
            let hasFile = false;
            let fileName = '';
            let contentToSend = content;
            
            // If file is uploaded, prepend to message but track separately
            if (uploadedFile) {
                try {
                    const fileContent = await uploadedFile.text();
                    fileName = uploadedFile.name;
                    // Send full content to LLM but don't show in chat
                    contentToSend = `[File: ${fileName}]\n\n${fileContent}\n\n${content}`;
                    hasFile = true;
                } catch (err) {
                    messages.push({ role: 'error', content: 'Failed to read file: ' + err.message });
                    renderMessages();
                    return;
                }
                uploadedFile = null;
                fileTagEl.style.display = 'none';
            }
            
            if (!content && !hasFile) {
                return;
            }

            // Add user message - if has a file, show filename + user text
            let displayContent = hasFile ? '[File: ' + fileName + '] ' + content : content;
            messages.push({ role: 'user', content: displayContent });
            renderMessages();
            inputEl.value = '';
            buttonEl.disabled = true;
            thinkingEl.style.display = 'flex';

            // Auto-expand chat when sending
            if (!chatExpanded) {
                toggleChat();
            }

            try {
                // Send full content to API (includes file content if attached)
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: contentToSend, display_content: displayContent }),
                });

                if (!response.ok) {
                    const text = await response.text();
                    messages.push({ role: 'error', content: text || `Request failed: ${response.status}` });
                    renderMessages();
                } else {
                    inputEl.value = '';
                }
            } catch (error) {
                messages.push({ role: 'error', content: String(error) });
                renderMessages();
            } finally {
                buttonEl.disabled = false;
                thinkingEl.style.display = 'none';
                inputEl.focus();
            }
        }

        buttonEl.addEventListener('click', sendMessage);
        inputEl.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });

        // Expose toggleChat globally
        window.toggleChat = toggleChat;

        connectWebSocket();
        inputEl.focus();
    </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(HOMEPAGE_TEMPLATE)


@app.get("/canvas")
async def canvas_page() -> HTMLResponse:
    return HTMLResponse(HTML_TEMPLATE)


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/state")
async def get_state() -> JSONResponse:
    return JSONResponse({"messages": display_messages, "elements": canvas_elements})


@app.post("/api/reset")
async def reset() -> JSONResponse:
    display_messages.clear()
    llm_messages.clear()
    canvas_elements.clear()
    await broadcast_canvas_state()
    await broadcast_chat_state()
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "init",
                    "messages": display_messages,
                    "elements": canvas_elements,
                }
            )
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)


@app.post("/api/chat")
async def chat(request: UserChatRequest) -> JSONResponse:
    content = request.content.strip()
    if not content:
        return JSONResponse({"error": "content is required"}, status_code=400)

    display_content = request.display_content if request.display_content else content

    async with chat_lock:
        await run_chat_turn(content, display_content)

    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6473)
