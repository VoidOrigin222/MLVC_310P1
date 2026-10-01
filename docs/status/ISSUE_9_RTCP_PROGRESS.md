# Issue #9 RTCP progress

本阶段先完成 RTCP 控制面的协议编解码，代码位于
`common/include/mlvc/transport/rtcp_session.h` 和
`common/src/transport/rtcp_session.cc`。

已实现：

- RFC 3550 Sender Report、Receiver Report 和 SDES CNAME；
- Generic NACK、PLI、FIR、TMMBR/TMMBR FCI；
- `MLVC` RTCP APP subtype 0，16 字节控制 envelope 和 4 字节对齐 TLV；
- compound RTCP 的首包 SR/RR、CNAME、长度、padding、SSRC 和字段范围校验；
- COMMAND 的 `ATOMIC`、非零 transaction ID 和控制版本校验。
- `RtcpUdpEndpoint` 单独负责 RTCP UDP bind/send/timeout receive，并在收发边界验证
  compound RTCP；它使用显式本地和远端端口，不隐式假设 RTP 端口。
- `RtpMlvcSender` 与 `RtpMessageReceiver` 已接入配对 UDP 端口。发送端绑定 RTP 偶数端口和
  相邻 RTCP 端口，按默认 5 秒间隔发 SR；接收端在 RTP 端口加 1 处接收 SR，并回送包含丢包、
  序号、jitter、LSR/DLSR 的 RR。普通数据配置中的 `*_transport_port` 仍填写 RTP 端口。
- RTP sender stats 加入 RTCP SR/RR 收发计数。`test_rtp_rtcp_session` 检查 localhost 上 RTP
  媒体接收以及 SR/RR 往返。
- 接收端会把持续超过 20 ms 的 RTP 序号缺口聚合为 Generic NACK；发送端维护最多 2048 个
  RTP 包的有界历史缓存，按原序号重发，并经过现有 pacer/发送队列。重复 NACK 在 100 ms 内
  不会重复排队。

`tools/cpp/tests/test_rtcp_session.cc` 已覆盖报文往返、TLV padding、反馈消息和畸形长度。
已接入 PLI/FIR：发送端 RTCP 线程只记录针对本媒体 SSRC 的请求，编码线程在下一帧开始前消费，
因此随机访问 I 帧及 reset-reference 标志不会在编码中途改变。MLVC APP COMMAND 已在同一帧边界消费，
支持 TLV 1 固定 Q、TLV 3 GOP、TLV 4 请求随机访问，并尊重 `apply_after_frame_id`；transaction ID 在会话内去重。
板上正式 CMake 构建和 `test_rtcp_session`、`test_rtp_rtcp_session` 均通过，产物部署到
`/root/workplace/MLVC_310P1_aipp/build_issue9_rtcp`。

RTCP UDP endpoint 现在支持等效 SRTCP 认证：设置同一个 `MLVC_RTCP_KEY` 后，所有 RTCP
报文附加 HMAC-SHA256 截断标签和 64 位发送索引；接收端先验签，再做 64 包重放窗口校验，
最后才交给 RTCP 解析器。未设置密钥时保留兼容的明文模式。正式运行时应在编码端和解码端
启动环境中设置相同的随机密钥，例如 `export MLVC_RTCP_KEY='...'`，不要把密钥写入仓库。

本机 Windows 使用 MinGW 编译并运行了协议单元测试；由于本机 RTCP socket 实现面向 POSIX，
UDP endpoint 的真实收发仍需在 Linux/Ascend 设备上验证。
