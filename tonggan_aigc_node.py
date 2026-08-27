"""
ComfyUI 自定义节点：通感 AIGC 生图 API + 图片上传（融合版）

功能：
1. TongganAIGCNode:
   - 支持 IMAGE 直接连线输入参考图（节点内部自动上传并取得 URL）
   - 支持 url1～url14 文本连线输入，以及 inputFiles 批量 URL 文本
   - 提交生图任务并自动轮询
   - 自动下载生成结果，输出 ComfyUI IMAGE
   - 同时输出图片 URL、提交任务响应、最终任务状态响应
2. TongganImageUploadNode: 本地图片 → 上传七牛云 → 输出图片 URL（支持失败自动重试）

安装：将本文件保存到 ComfyUI/custom_nodes/tonggan_aigc_node/tonggan_aigc_node.py，重启 ComfyUI
"""

import io
import json
import re
import time

import numpy as np
import requests
import torch
from PIL import Image


# ==================== 通用工具 ====================
def _json_or_none(resp):
    try:
        return resp.json()
    except ValueError:
        return None


def _pretty_json(payload, max_length=5000):
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(payload)
    return text[:max_length]


def _extract_server_task_id(payload):
    if not isinstance(payload, dict):
        return None

    data = payload.get("data")
    if isinstance(data, dict):
        return (
            data.get("tencentTaskId")
            or data.get("taskId")
            or data.get("id")
        )

    return payload.get("tencentTaskId") or payload.get("taskId")


def _raise_api_error(
    prefix,
    stage,
    local_task_id=None,
    server_task_id=None,
    resp=None,
    payload=None,
    exc=None,
    extra=None,
):
    lines = [f"[{prefix}] {stage}"]

    if local_task_id is not None:
        lines.append(f"本地 taskId: {local_task_id}")

    if server_task_id:
        lines.append(f"服务端 tencentTaskId: {server_task_id}")

    if resp is not None:
        lines.append(f"HTTP: {resp.status_code} {resp.reason}")

    if isinstance(payload, dict):
        lines.append(f"API code: {payload.get('code', 'N/A')}")
        lines.append(f"API message: {payload.get('message', 'N/A')}")
        lines.append(f"API 响应: {_pretty_json(payload, 1500)}")
    elif payload is not None:
        lines.append(f"API JSON: {repr(payload)[:1500]}")

    if exc is not None:
        lines.append(f"异常: {type(exc).__name__}: {exc}")

    if extra:
        lines.append(str(extra))

    if resp is not None and payload is None:
        text = (resp.text or "").strip()
        if text:
            lines.append(f"响应内容: {text[:1500]}")

    raise RuntimeError("\n".join(lines))


def _split_urls(value):
    """支持英文逗号、换行、制表符分隔 URL。"""
    if value is None:
        return []
    if not isinstance(value, str):
        value = str(value)
    return [part.strip() for part in re.split(r"[,\r\n\t]+", value) if part.strip()]


def _collect_input_urls(input_files, url_values):
    urls = []
    urls.extend(_split_urls(input_files))

    for value in url_values:
        urls.extend(_split_urls(value))

    # 去重并保持原始顺序
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def _image_tensor_to_png_bytes(image_tensor):
    """将 ComfyUI 单张 IMAGE tensor (H, W, C) 转为 PNG bytes。"""
    tensor = image_tensor.detach().cpu()
    image_np = (tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)

    if image_np.ndim != 3:
        raise RuntimeError(f"不支持的图片张量形状: {image_np.shape}")

    channels = image_np.shape[-1]
    if channels == 4:
        pil_image = Image.fromarray(image_np, mode="RGBA").convert("RGB")
    elif channels == 3:
        pil_image = Image.fromarray(image_np, mode="RGB")
    elif channels == 1:
        pil_image = Image.fromarray(image_np[:, :, 0], mode="L").convert("RGB")
    else:
        raise RuntimeError(f"不支持的图片通道数: {channels}")

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def _upload_image_bytes(file_bytes, app_id, api_key, token_url):
    """获取七牛云 Token → 上传图片 → 返回可访问 URL。"""
    headers = {
        "X-App-Id": app_id.strip(),
        "X-Api-Key": api_key.strip(),
        "Content-Type": "application/json",
    }

    # 1. 获取上传 Token
    try:
        resp = requests.post(token_url.strip(), headers=headers, timeout=30)
    except requests.RequestException as exc:
        _raise_api_error("Tonggan Upload", "获取上传 Token 网络异常", exc=exc)

    result = _json_or_none(resp)

    if not 200 <= resp.status_code < 300:
        _raise_api_error(
            "Tonggan Upload",
            "获取上传 Token HTTP 错误",
            resp=resp,
            payload=result,
        )

    if not isinstance(result, dict):
        _raise_api_error(
            "Tonggan Upload",
            "获取上传 Token 响应不是 JSON",
            resp=resp,
            payload=result,
        )

    if result.get("code") != 200:
        _raise_api_error(
            "Tonggan Upload",
            "获取上传 Token 业务失败",
            resp=resp,
            payload=result,
        )

    data = result.get("data")
    if not isinstance(data, dict):
        _raise_api_error(
            "Tonggan Upload",
            "获取上传 Token 响应缺少 data",
            resp=resp,
            payload=result,
        )

    missing_keys = [k for k in ("uptoken", "uploadUrl", "key") if not data.get(k)]
    if missing_keys:
        _raise_api_error(
            "Tonggan Upload",
            f"获取上传 Token 响应缺少字段: {', '.join(missing_keys)}",
            resp=resp,
            payload=result,
        )

    uptoken = data["uptoken"]
    upload_url = data["uploadUrl"]
    key = data["key"]

    # 2. 上传到七牛云
    files = {"file": ("image.png", file_bytes, "image/png")}
    form_data = {
        "token": uptoken,
        "key": key,
    }

    try:
        upload_resp = requests.post(
            upload_url,
            files=files,
            data=form_data,
            timeout=60,
        )
    except requests.RequestException as exc:
        _raise_api_error(
            "Tonggan Upload",
            "七牛云上传网络异常",
            exc=exc,
            extra=f"上传 key: {key}",
        )

    if upload_resp.status_code != 200:
        _raise_api_error(
            "Tonggan Upload",
            "七牛云上传失败",
            resp=upload_resp,
            extra=f"上传 key: {key}",
        )

    upload_result = _json_or_none(upload_resp)
    if not isinstance(upload_result, dict):
        _raise_api_error(
            "Tonggan Upload",
            "七牛云上传响应不是 JSON",
            resp=upload_resp,
            payload=upload_result,
            extra=f"上传 key: {key}",
        )

    # 3. 获取最终 URL
    url = upload_result.get("url")
    if not url:
        url = f"https://img.tongganai.com/{key}"

    return url


def _download_image_tensor(image_url):
    """下载单张图片 URL，并转换为 ComfyUI IMAGE tensor。"""
    try:
        resp = requests.get(
            image_url,
            timeout=60,
            headers={"User-Agent": "ComfyUI-TongganAIGC/1.0"},
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"下载图片网络异常: {image_url}；{type(exc).__name__}: {exc}") from exc

    if not 200 <= resp.status_code < 300:
        raise RuntimeError(
            f"下载图片失败: {image_url}；"
            f"HTTP {resp.status_code} {resp.reason}；"
            f"响应内容: {resp.text[:500]}"
        )

    try:
        pil_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"下载内容不是有效图片: {image_url}；{exc}") from exc

    image_np = np.asarray(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np)[None,]


def _download_image_batch(image_urls):
    tensors = [_download_image_tensor(url) for url in image_urls]
    if len(tensors) == 1:
        return tensors[0]

    shapes = {tuple(tensor.shape[1:3]) for tensor in tensors}
    if len(shapes) > 1:
        print(
            "[Tonggan AIGC] 多张生成图片尺寸不一致，ComfyUI 无法直接组成批次，"
            "本次仅输出第一张；完整 URL 仍会保留在 url 输出中。"
        )
        return tensors[0]

    return torch.cat(tensors, dim=0)


# ==================== 节点一：生图 ====================
class TongganAIGCNode:
    """
    调用通感 AIGC 生图 API。

    输入：
    - image：可选 IMAGE 直连参考图，节点内部自动上传
    - url1～url14：可选 URL 连线输入
    - inputFiles：可选批量 URL 文本，支持逗号或换行分隔

    输出：
    - image：下载后的 ComfyUI IMAGE
    - url：生成图片 URL，换行分隔
    - submit_response：提交生图 API 的原始响应
    - status_response：最终查询 API 的原始响应
    """

    CATEGORY = "image/generation"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "url", "submit_response", "status_response")
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        optional_url_inputs = {
            f"url{i}": ("STRING", {
                "forceInput": True,
                "tooltip": f"参考图片 URL {i}",
            })
            for i in range(1, 15)
        }

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入生图提示词...",
                }),
                "inputFiles": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "参考图片 URL，多个用英文逗号或换行分隔；不填则传空数组",
                }),
                "resolution": (["1k", "2k", "4k"], {
                    "default": "1k",
                }),
                "aspectRatio": (["auto", "1:1", "3:4", "4:3", "9:16", "16:9"], {
                    "default": "auto",
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
                "token_url": ("STRING", {
                    "default": "http://admin-dev.tongganagent.cn/api/v2/assets/get-uploadQN-token-passthrough",
                    "placeholder": "image 直连参考图时使用的上传 Token 接口",
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
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "直接连接参考图片；批量输入时默认使用第一张"}),
                **optional_url_inputs,
            },
        }

    def generate(
        self,
        prompt,
        inputFiles,
        resolution,
        aspectRatio,
        app_id,
        api_key,
        base_url,
        token_url,
        poll_interval,
        max_attempts,
        image=None,
        url1=None,
        url2=None,
        url3=None,
        url4=None,
        url5=None,
        url6=None,
        url7=None,
        url8=None,
        url9=None,
        url10=None,
        url11=None,
        url12=None,
        url13=None,
        url14=None,
    ):
        task_id = int(time.time() * 1000)

        # 收集文本 URL 和 14 个 URL 输入点
        url_values = (
            url1, url2, url3, url4, url5, url6, url7,
            url8, url9, url10, url11, url12, url13, url14,
        )
        reference_urls = _collect_input_urls(inputFiles, url_values)

        # 如果有 IMAGE 直连输入，先自动上传并取得 URL
        if image is not None:
            if not token_url or not token_url.strip():
                raise RuntimeError(
                    "[Tonggan AIGC] 已连接 image 参考图，但 token_url 为空，无法上传参考图"
                )

            batch_size = int(image.shape[0]) if hasattr(image, "shape") else 1
            if batch_size > 1:
                print(
                    f"[Tonggan AIGC] image 输入包含 {batch_size} 张图片，"
                    "当前默认使用第一张作为参考图"
                )

            try:
                image_bytes = _image_tensor_to_png_bytes(image[0])
                uploaded_url = _upload_image_bytes(
                    image_bytes,
                    app_id,
                    api_key,
                    token_url,
                )
                reference_urls.insert(0, uploaded_url)
            except Exception as exc:
                _raise_api_error(
                    "Tonggan AIGC",
                    "image 直连参考图上传失败",
                    local_task_id=task_id,
                    exc=exc,
                )

        # 去重并保持顺序
        reference_urls = list(dict.fromkeys(reference_urls))

        body = {
            "taskId": task_id,
            "prompt": prompt,
            "resolution": resolution,
            "modelName": "GG",
            "modelVersion": "3.1",
            "aspectRatio": "" if aspectRatio == "auto" else aspectRatio,
            "inputFiles": [{"url": url} for url in reference_urls],
        }

        # 构建 API URL
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

        # 1. 提交生图任务
        try:
            resp = requests.post(
                submit_url,
                json=body,
                headers=headers,
                timeout=30,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            _raise_api_error(
                "Tonggan AIGC",
                "提交任务网络异常",
                local_task_id=task_id,
                exc=exc,
            )

        submit_result = _json_or_none(resp)
        server_task_id = _extract_server_task_id(submit_result)

        if not 200 <= resp.status_code < 300:
            _raise_api_error(
                "Tonggan AIGC",
                "提交任务 HTTP 错误",
                local_task_id=task_id,
                server_task_id=server_task_id,
                resp=resp,
                payload=submit_result,
            )

        if not isinstance(submit_result, dict):
            _raise_api_error(
                "Tonggan AIGC",
                "提交任务响应不是 JSON",
                local_task_id=task_id,
                resp=resp,
                payload=submit_result,
            )

        if submit_result.get("code") != 200:
            _raise_api_error(
                "Tonggan AIGC",
                "提交任务业务失败",
                local_task_id=task_id,
                server_task_id=server_task_id,
                resp=resp,
                payload=submit_result,
            )

        submit_data = submit_result.get("data")
        if not isinstance(submit_data, dict) or not submit_data.get("tencentTaskId"):
            _raise_api_error(
                "Tonggan AIGC",
                "提交任务响应缺少 tencentTaskId",
                local_task_id=task_id,
                server_task_id=server_task_id,
                resp=resp,
                payload=submit_result,
            )

        tencent_task_id = submit_data["tencentTaskId"]

        # 2. 轮询任务状态
        final_status_result = None
        last_status_result = None
        image_urls = []

        for attempt in range(1, max_attempts + 1):
            time.sleep(poll_interval)

            try:
                resp = requests.get(
                    status_url,
                    params={"taskId": tencent_task_id},
                    headers={
                        "X-App-Id": app_id.strip(),
                        "X-Api-Key": api_key.strip(),
                    },
                    timeout=30,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                _raise_api_error(
                    "Tonggan AIGC",
                    f"查询任务状态网络异常（第 {attempt} 次）",
                    local_task_id=task_id,
                    server_task_id=tencent_task_id,
                    exc=exc,
                )

            status_result = _json_or_none(resp)
            last_status_result = status_result

            if not 200 <= resp.status_code < 300:
                _raise_api_error(
                    "Tonggan AIGC",
                    f"查询任务状态 HTTP 错误（第 {attempt} 次）",
                    local_task_id=task_id,
                    server_task_id=tencent_task_id,
                    resp=resp,
                    payload=status_result,
                )

            if not isinstance(status_result, dict):
                _raise_api_error(
                    "Tonggan AIGC",
                    f"查询任务状态响应不是 JSON（第 {attempt} 次）",
                    local_task_id=task_id,
                    server_task_id=tencent_task_id,
                    resp=resp,
                    payload=status_result,
                )

            if status_result.get("code") != 200:
                _raise_api_error(
                    "Tonggan AIGC",
                    f"查询任务状态业务失败（第 {attempt} 次）",
                    local_task_id=task_id,
                    server_task_id=tencent_task_id,
                    resp=resp,
                    payload=status_result,
                )

            data = status_result.get("data")
            if not isinstance(data, dict):
                _raise_api_error(
                    "Tonggan AIGC",
                    f"查询任务状态响应缺少 data（第 {attempt} 次）",
                    local_task_id=task_id,
                    server_task_id=tencent_task_id,
                    resp=resp,
                    payload=status_result,
                )

            status = data.get("status")

            if status == "FINISH":
                image_urls = data.get("imageUrls") or []
                final_status_result = status_result
                break

            if status == "FAIL":
                _raise_api_error(
                    "Tonggan AIGC",
                    "任务执行失败",
                    local_task_id=task_id,
                    server_task_id=tencent_task_id,
                    resp=resp,
                    payload=status_result,
                )

            if status in ("WAITING", "PROCESSING"):
                continue

            _raise_api_error(
                "Tonggan AIGC",
                f"查询任务状态返回未知状态: {status}",
                local_task_id=task_id,
                server_task_id=tencent_task_id,
                resp=resp,
                payload=status_result,
            )

        else:
            _raise_api_error(
                "Tonggan AIGC",
                f"轮询超时，任务仍在处理中（已等待 {max_attempts * poll_interval} 秒）",
                local_task_id=task_id,
                server_task_id=tencent_task_id,
                payload=last_status_result,
            )

        if not image_urls:
            _raise_api_error(
                "Tonggan AIGC",
                "任务已完成，但响应中没有 imageUrls",
                local_task_id=task_id,
                server_task_id=tencent_task_id,
                payload=final_status_result,
            )

        # 3. 下载生成图片，转为 ComfyUI IMAGE
        try:
            image_tensor = _download_image_batch(image_urls)
        except Exception as exc:
            _raise_api_error(
                "Tonggan AIGC",
                "下载生成图片失败",
                local_task_id=task_id,
                server_task_id=tencent_task_id,
                exc=exc,
                payload=final_status_result,
            )

        return (
            image_tensor,
            "\n".join(image_urls),
            _pretty_json(submit_result),
            _pretty_json(final_status_result),
        )


# ==================== 节点二：图片上传（带重试） ====================
class TongganImageUploadNode:
    """
    将 ComfyUI 图片上传至通感平台，获取可访问的 URL。
    流程：获取七牛云 Token → 上传图片 → 返回 URL。
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
        file_bytes = _image_tensor_to_png_bytes(image[0])

        last_error = None
        for attempt in range(1, retry_times + 2):
            try:
                url = _upload_image_bytes(file_bytes, app_id, api_key, token_url)
                if attempt > 1:
                    print(f"[Tonggan Upload] 第 {attempt - 1} 次重试成功")
                return (url,)
            except Exception as exc:
                last_error = exc
                if attempt <= retry_times:
                    wait = retry_interval * attempt
                    print(
                        f"[Tonggan Upload] 第 {attempt} 次上传失败: {exc}，"
                        f"{wait} 秒后进行第 {attempt} 次重试..."
                    )
                    time.sleep(wait)
                else:
                    print(f"[Tonggan Upload] 第 {attempt} 次上传失败: {exc}，已达最大重试次数")

        raise RuntimeError(
            f"[Tonggan Upload] 上传失败（已重试 {retry_times} 次）: {last_error}"
        )


# ==================== 节点注册 ====================
NODE_CLASS_MAPPINGS = {
    "TongganAIGCNode": TongganAIGCNode,
    "TongganImageUploadNode": TongganImageUploadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TongganAIGCNode": "🎨 Tonggan AIGC Image",
    "TongganImageUploadNode": "📤 Tonggan Image Upload",
}
