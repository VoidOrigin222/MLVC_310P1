# Windows 三路低延时对比界面

## HTML 控制台（推荐）

启动本地服务：

```powershell
python tools/ui/web_ui.py
```

或双击 `run_web_ui.bat`，然后打开 <http://127.0.0.1:8765>。浏览器页面提供
RTSP 地址、启停、同带宽/同质量调参和三路统计；实际视频由本机弹出的三个
`ffplay` 窗口播放。这样保留了低延时 RTSP 播放，同时使用 HTML 做控制面板。

启动：

```powershell
python tools/ui/mlvc_compare.py
```

界面分别启动三个 `ffplay` 窗口：原图、H.264、MLVC。默认原图地址为
`rtsp://192.168.5.3:8554/camera-original`，MLVC 地址为
`rtsp://192.168.5.11:8554/mlvc`，可按实际 MediaMTX 路径修改。

勾选“使用原图→ffmpeg pipe”时，Windows 端从原图 RTSP 读取，使用
`libx264` 零延时参数编码，再通过 stdout pipe 交给 ffplay，不产生中间文件。
播放使用 `nobuffer`、`low_delay`、`framedrop`、`analyzeduration=0`。

面板显示压缩率（倍）、bpp 和带宽（KB/s），并通过 `drawtext` 文件每秒刷新到
三个播放画面。`同带宽`根据输入码率计算 bpp；`同质量`根据 bpp 反算码率。
Windows PATH 需要有 Python、ffmpeg 和 ffplay，也可以在界面填完整路径。
