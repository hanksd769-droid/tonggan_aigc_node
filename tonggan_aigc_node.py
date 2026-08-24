"""
ComfyUI 自定义节点：通感 AIGC 生图 API + 图片上传

功能：
1. TongganAIGCNode: 提交生图任务 → 自动轮询状态 → 输出图片 URL 列表（纯文本）
2. TongganImageUploadNode: 本地图片 → 上传七牛云 → 输出图片 URL（支持失败自动重试）

安装：将本文件保存到 ComfyUI/custom_nodes/tonggan_aigc_node.py，重启 ComfyUI
"""

import time
import json
import requests
import io
import numpy as np
from PIL import Image


# ==================== 节点一：生图 ====================
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
            return ("",)

        # 输出 URL 列表（纯文本，换行分隔）
        return ("\n".join(image_urls),)


# ==================== 节点二：图片上传（带重试） ====================
class TongganImageUploadNode:
    """
    将 ComfyUI 图片上传至通感平台，获取可访问的 URL。
    流程：获取七牛云 Token → 上传图片 → 返回 URL
    支持失败自动重试：网络异常、Token 获取失败、七牛云上传失败均会重试，
    每次重试重新获取 Token，避免 Token 过期导致反复失败。
    """

    CATEGORY = "image/upload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_url",)
    FUNCTION = "upload"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "app_id": ("STRING", {
                    "default": "",
                    "placeholder": "App ID，如 app_abcdefgh",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "API Key，如 sk-...",
                }),
                "token_url": ("STRING", {
                    "default": "http://admin-dev.tongganagent.cn/api/v2/assets/get-uploadQN-token-passthrough",
                    "placeholder": "获取上传Token的接口地址",
                }),
                "retry_times": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "上传失败后的重试次数（0 = 不重试）",
                }),
                "retry_interval": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "number",
                    "tooltip": "重试等待秒数，逐次递增（第n次重试等待 n×该值 秒）",
                }),
            }
        }

    def upload(self, image, app_id, api_key, token_url, retry_times, retry_interval):
        # image 是 torch tensor (B, H, W, C)，取第一张
        img_tensor = image[0]  # (H, W, C)
        img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # 转为 PNG bytes
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        file_bytes = buf.getvalue()

        # 带重试的上传：首次尝试 + retry_times 次重试
        last_error = None
        for attempt in range(1, retry_times + 2):
            try:
                url = self._upload_once(file_bytes, app_id, api_key, token_url)
                if attempt > 1:
                    print(f"[Tonggan Upload] 第 {attempt - 1} 次重试成功")
                return (url,)
            except Exception as e:
                last_error = e
                if attempt <= retry_times:
                    wait = retry_interval * attempt
                    print(
                        f"[Tonggan Upload] 第 {attempt} 次上传失败: {e}，"
                        f"{wait} 秒后进行第 {attempt} 次重试..."
                    )
                    time.sleep(wait)
                else:
                    print(f"[Tonggan Upload] 第 {attempt} 次上传失败: {e}，已达最大重试次数")

        raise RuntimeError(
            f"[Tonggan Upload] 上传失败（已重试 {retry_times} 次）: {last_error}"
        )

    def _upload_once(self, file_bytes, app_id, api_key, token_url):
        """单次完整上传流程：获取 Token → 上传七牛云 → 返回 URL。失败抛异常由上层重试。"""

        # 1. 获取七牛云上传 Token
        headers = {
            "X-App-Id": app_id.strip(),
            "X-Api-Key": api_key.strip(),
            "Content-Type": "application/json",
        }
        resp = requests.post(token_url, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 200:
            raise RuntimeError(f"获取Token失败: {result.get('message', '未知错误')}")

        data = result["data"]
        uptoken = data["uptoken"]
        upload_url = data["uploadUrl"]
        key = data["key"]

        # 2. 上传图片到七牛云（multipart/form-data）
        files = {"file": ("image.png", file_bytes, "image/png")}
        form_data = {
            "token": uptoken,
            "key": key,
        }
        resp = requests.post(upload_url, files=files, data=form_data, timeout=60)

        # 先检查状态码，非200时直接抛响应内容
        if resp.status_code != 200:
            raise RuntimeError(
                f"七牛云上传失败 HTTP {resp.status_code}: {resp.reason}，"
                f"响应内容: {resp.text[:800]}"
            )

        upload_result = resp.json()

        # 3. 获取最终 URL
        # 优先使用七牛云返回的 url 字段，否则按规则拼接
        url = upload_result.get("url")
        if not url:
            url = f"https://img.tongganai.com/{key}"

        return url


# ==================== 节点注册 ====================
NODE_CLASS_MAPPINGS = {
    "TongganAIGCNode": TongganAIGCNode,
    "TongganImageUploadNode": TongganImageUploadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TongganAIGCNode": "🎨 Tonggan AIGC Image",
    "TongganImageUploadNode": "📤 Tonggan Image Upload",
}
