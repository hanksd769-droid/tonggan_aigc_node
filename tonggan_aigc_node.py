"""
ComfyUI 自定义节点：通感 AIGC 生图 API（修复版）

功能：提交生图任务 → 自动轮询状态 → 输出图片 URL 列表（JSON 字符串）
关键修复：显式传入 modelName/modelVersion，与 Coze 调用保持一致。

安装：将本文件保存到 ComfyUI/custom_nodes/tonggan_aigc_node.py，重启 ComfyUI
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

    STATUS_MAP = {
        "WAITING": "排队中",
        "PROCESSING": "处理中",
        "FINISH": "已完成",
        "FAIL": "失败",
    }

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
                    "placeholder": "参考图片 URL，多个用英文逗号分隔；不填则传空数组 []",
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
        print("\n" + "=" * 60)
        print("[Tonggan AIGC] 🚀 开始执行")
        print("=" * 60)

        # ---------- 1. 构建请求体 ----------
        task_id = int(time.time() * 1000) % 1000000
        body = {
            "taskId": task_id,
            "prompt": prompt,
            "resolution": resolution,
            "modelName": "GG",        # 关键：显式传入，与 Coze 保持一致
            "modelVersion": "3.1",    # 关键：显式传入，与 Coze 保持一致
        }

        # aspectRatio 处理
        if aspectRatio == "auto":
            body["aspectRatio"] = ""
        else:
            body["aspectRatio"] = aspectRatio

        # inputFiles 处理：不填图片时传 []，填了则转 [{"url": "xxx"}, ...]
        if inputFiles and inputFiles.strip():
            urls = [u.strip() for u in inputFiles.split(",") if u.strip()]
            if urls:
                body["inputFiles"] = [{"url": url} for url in urls]
                print(f"[Tonggan AIGC] 📎 传入参考图片，共 {len(urls)} 张")
            else:
                body["inputFiles"] = []
                print("[Tonggan AIGC] 📎 inputFiles 为空，传 []")
        else:
            body["inputFiles"] = []
            print("[Tonggan AIGC] 📎 inputFiles 未填写，传 []（文生图模式）")

        # ---------- 2. 构建 URL ----------
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

        # 打印完整请求信息
        print("\n" + "-" * 60)
        print("[Tonggan AIGC] 📤 提交任务信息：")
        print(f"  URL: {submit_url}")
        print(f"  Headers: {json.dumps(headers, ensure_ascii=False, indent=2)}")
        print(f"  Body: {json.dumps(body, ensure_ascii=False, indent=2)}")
        print("-" * 60 + "\n")

        # ---------- 3. 提交任务 ----------
        try:
            resp = requests.post(
                submit_url,
                json=body,
                headers=headers,
                timeout=30,
                allow_redirects=False,
            )
            print(f"[Tonggan AIGC] 📥 响应状态码: {resp.status_code}")
            print(f"[Tonggan AIGC] 📥 响应内容: {resp.text[:500]}")

            if resp.status_code in (301, 302, 307, 308):
                loc = resp.headers.get("Location", "未知")
                raise RuntimeError(
                    f"服务器返回 {resp.status_code} 重定向到 {loc}，"
                    f"请将 base_url 改为 HTTPS 地址。"
                )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"[Tonggan AIGC] ❌ 提交任务异常: {e}")
            raise RuntimeError(f"[Tonggan AIGC] 提交任务网络异常: {e}")

        if result.get("code") != 200:
            raise RuntimeError(
                f"[Tonggan AIGC] 提交任务失败: {result.get('message', '未知错误')}"
            )

        tencent_task_id = result["data"]["tencentTaskId"]
        print(f"[Tonggan AIGC] ✅ 提交成功，腾讯云任务ID: {tencent_task_id}")

        # ---------- 4. 轮询查询状态 ----------
        image_urls = []
        last_status = None

        print(f"[Tonggan AIGC] ⏳ 开始轮询，最多 {max_attempts} 次，间隔 {poll_interval} 秒")

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
                print(f"[Tonggan AIGC] ⚠️ 查询状态异常 (第{attempt + 1}次): {e}")
                continue

            if status_result.get("code") != 200:
                print(f"[Tonggan AIGC] ⚠️ 查询状态失败 (第{attempt + 1}次): {status_result.get('message')}")
                continue

            data = status_result.get("data", {})
            status = data.get("status")
            status_cn = self.STATUS_MAP.get(status, status)

            if status != last_status:
                print(f"[Tonggan AIGC] 📊 状态变更: {status} ({status_cn})")
                last_status = status

            if status == "FINISH":
                image_urls = data.get("imageUrls", [])
                print(f"[Tonggan AIGC] ✅ 任务完成！共 {len(image_urls)} 张图片")
                for i, url in enumerate(image_urls, 1):
                    print(f"    图片 {i}: {url}")
                break

            elif status == "FAIL":
                err_msg = data.get("message", "未知错误")
                err_code = data.get("errCode", "N/A")
                print(f"[Tonggan AIGC] ❌ 任务失败: {err_msg} (errCode={err_code})")
                raise RuntimeError(
                    f"[Tonggan AIGC] 任务失败: {err_msg} (errCode={err_code})"
                )

            elif status in ("WAITING", "PROCESSING"):
                if (attempt + 1) % 5 == 0 or attempt == 0:
                    print(f"[Tonggan AIGC] ⏳ 轮询中... 第 {attempt + 1}/{max_attempts} 次 | 状态: {status_cn} | 已等待 {(attempt + 1) * poll_interval} 秒")
                continue

            else:
                print(f"[Tonggan AIGC] ⚠️ 未知状态: {status}")

        else:
            raise RuntimeError(
                f"[Tonggan AIGC] 轮询超时，任务仍在处理中 "
                f"(已等待 {max_attempts * poll_interval} 秒)"
            )

        if not image_urls:
            print("[Tonggan AIGC] ⚠️ 任务完成但未返回图片 URL")
            return ("[]",)

        # ---------- 5. 输出 URL 列表 ----------
        urls_json = json.dumps(image_urls, ensure_ascii=False)
        print(f"\n[Tonggan AIGC] 📤 最终输出: {urls_json}\n")
        return (urls_json,)


# ==================== 节点注册 ====================
NODE_CLASS_MAPPINGS = {
    "TongganAIGCNode": TongganAIGCNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TongganAIGCNode": "🎨 Tonggan AIGC Image",
}
