"""
ComfyUI 自定义节点：通感 AIGC 生图 API

"""

import time
import json
import requests


class TongganAIGCNode:
    """
    调用通感 AIGC 生图 API。
    提交任务 + 自动轮询 + 输出图片 URL 列表，不下载图片。
    """

    CATEGORY = "image/generation"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_urls",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入生图提示词...",
                }),
                "resolution": (["1k", "2k", "4k"], {
                    "default": "1k",
                }),
                "aspectRatio": (["auto", "1:1", "3:4", "4:3", "9:16", "16:9"], {
                    "default": "auto",
                }),
                "inputFiles": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "参考图片 URL，多个用英文逗号分隔；不填则传空数组",
                }),
                "app_id": ("STRING", {
                    "default": "",
                    "placeholder": "App ID，如 app_abcdefgh",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "API Key，如 sk-...",
                }),
                "base_url": ("STRING", {
                    "default": "https://www.tongganagent.cn/api/v2/ai-creations",
                    "placeholder": "只填到 /ai-creations，不要带后缀",
                }),
                "poll_interval": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "number",
                }),
                "max_attempts": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "display": "number",
                }),
            }
        }

    def generate(
        self,
        prompt,
        resolution,
        aspectRatio,
        inputFiles,
        app_id,
        api_key,
        base_url,
        poll_interval,
        max_attempts,
    ):
        # 构建请求体
        task_id = int(time.time() * 1000) % 1000000
        body = {
            "taskId": task_id,
            "prompt": prompt,
            "resolution": resolution,
            "modelName": "GG",
            "modelVersion": "3.1",
        }

        if aspectRatio == "auto":
            body["aspectRatio"] = ""
        else:
            body["aspectRatio"] = aspectRatio

        if inputFiles and inputFiles.strip():
            urls = [u.strip() for u in inputFiles.split(",") if u.strip()]
            body["inputFiles"] = [{"url": url} for url in urls]
        else:
            body["inputFiles"] = []

        # 构建 URL
        base = base_url.strip().rstrip("/")
        for suffix in ("/tencent-aigc-image/status", "/tencent-aigc-image"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        submit_url = f"{base}/tencent-aigc-image"
        status_url = f"{base}/tencent-aigc-image/status"

        headers = {
            "X-App-Id": app_id.strip(),
            "X-Api-Key": api_key.strip(),
            "Content-Type": "application/json",
        }

        # 提交任务
        resp = requests.post(
            submit_url,
            json=body,
            headers=headers,
            timeout=30,
            allow_redirects=False,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 200:
            raise RuntimeError(
                f"[Tonggan AIGC] 提交任务失败: {result.get('message', '未知错误')}"
            )

        tencent_task_id = result["data"]["tencentTaskId"]

        # 轮询查询状态
        image_urls = []

        for _ in range(max_attempts):
            time.sleep(poll_interval)

            resp = requests.get(
                status_url,
                params={"taskId": tencent_task_id},
                headers={"X-App-Id": app_id.strip(), "X-Api-Key": api_key.strip()},
                timeout=30,
                allow_redirects=False,
            )
            resp.raise_for_status()
            status_result = resp.json()

            if status_result.get("code") != 200:
                continue

            data = status_result.get("data", {})
            status = data.get("status")

            if status == "FINISH":
                image_urls = data.get("imageUrls", [])
                break

            elif status == "FAIL":
                err_msg = data.get("message", "未知错误")
                err_code = data.get("errCode", "N/A")
                raise RuntimeError(
                    f"[Tonggan AIGC] 任务失败: {err_msg} (errCode={err_code})"
                )

            elif status in ("WAITING", "PROCESSING"):
                continue

        else:
            raise RuntimeError(
                f"[Tonggan AIGC] 轮询超时，任务仍在处理中 "
                f"(已等待 {max_attempts * poll_interval} 秒)"
            )

        if not image_urls:
            return ("[]",)

        # 输出 URL 列表
        return (json.dumps(image_urls, ensure_ascii=False),)


# ==================== 节点注册 ====================
NODE_CLASS_MAPPINGS = {
    "TongganAIGCNode": TongganAIGCNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TongganAIGCNode": "🎨 Tonggan AIGC Image",
}
