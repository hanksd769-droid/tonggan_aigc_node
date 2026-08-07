# ComfyUI-Tonggan-AIGC

ComfyUI 自定义节点，用于调用通感（Tonggan）AIGC 生图 API。支持提交生图任务、自动轮询任务状态、下载生成图片并输出为 ComfyUI 可用的 IMAGE 格式。

## 功能特性

- **提交 + 轮询一体化**：一个节点完成提交任务、自动轮询等待、下载图片全流程
- **inputFiles 智能转换**：支持传入逗号分隔的 URL 字符串，自动转为 `[{"url": "xxx"}, ...]` 格式
- **保留空值字段**：空字符串（如 `"aspectRatio": ""`）会原样提交，避免后端使用默认值
- **URL 防呆处理**：自动修正误填的 base_url 后缀，防止路径拼接错误
- **301 重定向防护**：禁用自动重定向，防止 HTTP → HTTPS 跳转导致 POST 变 GET 而鉴权失败

## 安装

1. 将 `tonggan_aigc_node.py` 复制到 ComfyUI 的自定义节点目录：

   ```bash
   cp tonggan_aigc_node.py /path/to/ComfyUI/custom_nodes/