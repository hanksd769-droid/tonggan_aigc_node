"""
ComfyUI 自定义节点：通感 AIGC 生图 API

功能：提交生图任务 → 自动轮询状态 → 下载图片 → 输出 IMAGE
支持 inputFiles 传入逗号分隔字符串，自动转为 [{"url": "xxx"}, ...] 格式
保留空字符串字段（如 aspectRatio: ""），防止后端使用默认值

安装：将本文件保存到 ComfyUI/custom_nodes/tonggan_aigc_node.py，重启 ComfyUI
"""

import time
import json
import requests
import numpy as np
from PIL import Image
import torch
import io
import urllib.request


class TongganAIGCNode:
    """
    调用通感 AIGC 生图 API。
    提交任务 + 自动轮询 + 下载图片，一体化完成。
    """

    CATEGORY = "image/generation"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                "request_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": '接口请求参数（JSON 格式），如：\n'
                                   '{\n'
                                   '  "prompt": "测试，随便生",\n'
                                   '  "aspectRatio": "",\n'
                                   '  "resolution": "2k",\n'
                                   '  "inputFiles": "https://.../a.jpg, https://.../b.jpg"\n'
                                   '}',
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
        app_id,
        api_key,
        base_url,
        request_json,
        poll_interval,
        max_attempts,
    ):
        # ---------- 1. 解析并处理请求参数 ----------
        task_id = int(time.time() * 1000) % 1000000
        body = {"taskId": task_id}

        if request_json and request_json.strip():
            try:
                extra = json.loads(request_json.strip())
                if not isinstance(extra, dict):
                    raise RuntimeError("[Tonggan AIGC] request_json 必须是 JSON 对象")
                body.update(extra)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"[Tonggan AIGC] request_json 解析失败: {e}")

        # ---------- 2. 自动转换 inputFiles ----------
        if "inputFiles" in body:
            raw = body["inputFiles"]
            if isinstance(raw, str):
                urls = [u.strip() for u in raw.split(",") if u.strip()]
                body["inputFiles"] = [{"url": url} for url in urls]
            elif isinstance(raw, list):
                formatted = []
                for item in raw:
                    if isinstance(item, str):
                        formatted.append({"url": item})
                    elif isinstance(item, dict) and "url" in item:
                        formatted.append(item)
                    else:
                        print(f"[Tonggan AIGC] ⚠️ 跳过无效的 inputFiles 项: {item}")
                body["inputFiles"] = formatted

        # ---------- 3. 构建 URL（防呆：去掉误填后缀）----------
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

        print("=" * 60)
        print("[Tonggan AIGC] 🔍 调试信息：")
        print(f"  提交 URL: {submit_url}")
        print(f"  Body: {json.dumps(body, ensure_ascii=False, indent=2)}")
        print("=" * 60)

        # ---------- 4. 提交任务（禁用重定向，防止 301 导致 POST 变 GET）----------
        try:
            resp = requests.post(
                submit_url,
                json=body,
                headers=headers,
                timeout=30,
                allow_redirects=False,
            )
            print(f"[Tonggan AIGC] 原始响应码: {resp.status_code}")
            print(f"[Tonggan AIGC] 响应内容: {resp.text[:500]}")

            if resp.status_code in (301, 302, 307, 308):
                loc = resp.headers.get("Location", "未知")
                raise RuntimeError(
                    f"服务器返回 {resp.status_code} 重定向到 {loc}。\n"
                    f"请将 base_url 改为 HTTPS 地址，或联系管理员确认 API 地址。"
                )

            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            raise RuntimeError(f"[Tonggan AIGC] 提交任务网络异常: {e}")

        if result.get("code") != 200:
            raise RuntimeError(
                f"[Tonggan AIGC] 提交任务失败: {result.get('message', '未知错误')}"
            )

        tencent_task_id = result["data"]["tencentTaskId"]
        print(f"[Tonggan AIGC] ✅ 任务已提交 | tencentTaskId={tencent_task_id}")

        # ---------- 5. 轮询查询状态 ----------
        image_urls = []

        for attempt in range(max_attempts):
            time.sleep(poll_interval)

            try:
                resp = requests.get(
                    status_url,
                    params={"taskId": tencent_task_id},
                    headers={"X-App-Id": app_id.strip(), "X-Api-Key": api_key.strip()},
                    timeout=30,
                    allow_redirects=False,
                )
                resp.raise_for_status()
                status_result = resp.json()
            except Exception as e:
                print(f"[Tonggan AIGC] 查询状态网络异常: {e}")
                continue

            if status_result.get("code") != 200:
                print(f"[Tonggan AIGC] 查询状态失败: {status_result.get('message')}")
                continue

            data = status_result.get("data", {})
            status = data.get("status")

            if status == "FINISH":
                image_urls = data.get("imageUrls", [])
                print(f"[Tonggan AIGC] ✅ 任务完成，共 {len(image_urls)} 张图片")
                break

            elif status == "FAIL":
                err_msg = data.get("message", "未知错误")
                err_code = data.get("errCode", "N/A")
                raise RuntimeError(
                    f"[Tonggan AIGC] ❌ 任务失败: {err_msg} (errCode={err_code})"
                )

            elif status in ("WAITING", "PROCESSING"):
                print(f"[Tonggan AIGC] ⏳ 轮询 [{attempt + 1}/{max_attempts}] 状态: {status}")
                continue

            else:
                print(f"[Tonggan AIGC] ⚠️ 未知状态: {status}")

        else:
            raise RuntimeError(
                f"[Tonggan AIGC] ⏰ 轮询超时，任务仍在处理中 "
                f"(已等待 {max_attempts * poll_interval} 秒)"
            )

        if not image_urls:
            raise RuntimeError("[Tonggan AIGC] 任务完成但未返回图片 URL")

        # ---------- 6. 下载图片并转为 ComfyUI tensor ----------
        images = []
        for url in image_urls:
            try:
                tensor = self._download_image_to_tensor(url)
                images.append(tensor)
                print(f"[Tonggan AIGC] ✅ 已下载图片: {url}")
            except Exception as e:
                print(f"[Tonggan AIGC] ❌ 下载图片失败 {url}: {e}")

        if not images:
            raise RuntimeError("[Tonggan AIGC] 所有图片下载失败")

        if len(images) == 1:
            return (images[0],)
        return (torch.cat(images, dim=0),)

    def _download_image_to_tensor(self, url: str) -> torch.Tensor:
        """从 URL 下载图片并转为 ComfyUI IMAGE 格式 (B, H, W, C)"""
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (ComfyUI-TongganAIGC/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            image_data = response.read()

        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None, ...]  # (1, H, W, C)


# ==================== 节点注册 ====================
NODE_CLASS_MAPPINGS = {
    "TongganAIGCNode": TongganAIGCNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TongganAIGCNode": "🎨 Tonggan AIGC Image",
}